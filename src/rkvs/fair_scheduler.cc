#include "rkvs/fair_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace rkvs {

LatestFairScheduler::LatestFairScheduler(
    std::vector<SchedulerSourceConfig> sources, int workers,
    Processor processor, ResultCallback result_callback)
    : worker_count_(workers), processor_(std::move(processor)),
      result_callback_(std::move(result_callback)) {
  if (workers < 1 || workers > 3) throw std::invalid_argument("workers must be 1..3");
  if (!processor_) throw std::invalid_argument("scheduler processor is required");
  for (auto& source : sources) {
    if (source.source_id.empty() || source.target_fps < 1)
      throw std::invalid_argument("invalid scheduler source");
    if (sources_.count(source.source_id))
      throw std::invalid_argument("duplicate scheduler source");
    order_.push_back(source.source_id);
    SourceSlot slot;
    slot.config = std::move(source);
    sources_.emplace(slot.config.source_id, std::move(slot));
  }
  if (sources_.empty()) throw std::invalid_argument("scheduler has no sources");
  max_in_flight_per_source_ = std::max(
      1, (worker_count_ + static_cast<int>(sources_.size()) - 1) /
             static_cast<int>(sources_.size()));
}

LatestFairScheduler::~LatestFairScheduler() { stop(); }

void LatestFairScheduler::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) return;
  running_ = true;
  for (int worker = 0; worker < worker_count_; ++worker)
    threads_.emplace_back(&LatestFairScheduler::worker_loop, this, worker);
}

void LatestFairScheduler::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
    running_ = false;
    for (auto& item : sources_) item.second.pending.reset();
  }
  cv_.notify_all();
  for (auto& thread : threads_) if (thread.joinable()) thread.join();
  threads_.clear();
}

void LatestFairScheduler::submit(SharedFrame frame) {
  if (!frame) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = sources_.find(frame->source_id);
    if (found == sources_.end() || !running_) return;
    auto& slot = found->second;
    ++slot.stats.submitted;
    if (slot.pending) ++slot.stats.replaced;
    slot.pending = std::move(frame);
    slot.stats.queue_depth = 1;
  }
  cv_.notify_one();
}

void LatestFairScheduler::pause() {
  std::lock_guard<std::mutex> lock(mutex_);
  paused_ = true;
  // Discard pre-pause frames so resume never produces a historical burst.
  for (auto& item : sources_) {
    if (item.second.pending) ++item.second.stats.replaced;
    item.second.pending.reset();
    item.second.stats.queue_depth = 0;
  }
}

void LatestFairScheduler::resume() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    paused_ = false;
    for (auto& item : sources_) item.second.next_due_ns = 0;
  }
  cv_.notify_all();
}

void LatestFairScheduler::update_sources(
    const std::vector<SchedulerSourceConfig>& sources) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (sources.size() != sources_.size())
    throw std::invalid_argument("scheduler source topology change requires restart");
  for (const auto& source : sources) {
    auto found = sources_.find(source.source_id);
    if (found == sources_.end())
      throw std::invalid_argument("scheduler source topology change requires restart");
    if (source.target_fps < 1)
      throw std::invalid_argument("invalid scheduler target fps");
    // Motion gating may report the same state for every input frame. Resetting
    // next_due_ns on every identical update bypasses the configured idle FPS
    // and accidentally schedules one detection per motion result.
    if (found->second.config.target_fps != source.target_fps) {
      found->second.config.target_fps = source.target_fps;
      found->second.next_due_ns = 0;
    }
  }
  cv_.notify_all();
}

bool LatestFairScheduler::paused() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return paused_;
}

std::map<std::string, SchedulerSourceStats> LatestFairScheduler::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<std::string, SchedulerSourceStats> result;
  for (const auto& item : sources_) {
    auto copy = item.second.stats;
    copy.in_flight = copy.in_flight_count > 0;
    if (item.second.first_completed_ns && copy.completed > 1) {
      const uint64_t elapsed = monotonic_ns() - item.second.first_completed_ns;
      if (elapsed) copy.detection_fps = (copy.completed - 1) * 1e9 / elapsed;
    }
    auto samples = item.second.latency_samples;
    if (!samples.empty()) {
      std::sort(samples.begin(), samples.end());
      auto percentile = [&](double fraction) {
        const size_t index = std::min(samples.size() - 1,
            static_cast<size_t>(fraction * static_cast<double>(samples.size() - 1)));
        return samples[index];
      };
      copy.latency_p50_ms = percentile(0.50);
      copy.latency_p95_ms = percentile(0.95);
    }
    result[item.first] = copy;
  }
  return result;
}

void LatestFairScheduler::worker_loop(int worker_index) {
  while (true) {
    SharedFrame frame;
    std::string source_id;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (running_ && !frame) {
        if (!paused_) {
          const uint64_t now = monotonic_ns();
          for (size_t offset = 0; offset < order_.size(); ++offset) {
            const size_t index = (cursor_ + offset) % order_.size();
            auto& slot = sources_.at(order_[index]);
            if (slot.pending &&
                slot.stats.in_flight_count < max_in_flight_per_source_ &&
                now >= slot.next_due_ns) {
              frame = std::move(slot.pending);
              source_id = order_[index];
              ++slot.stats.in_flight_count;
              slot.stats.in_flight = true;
              slot.stats.queue_depth = 0;
              const uint64_t period = 1000000000ULL /
                  static_cast<uint64_t>(slot.config.target_fps);
              slot.next_due_ns = now + period;
              cursor_ = (index + 1) % order_.size();
              break;
            }
          }
        }
        if (!frame) cv_.wait_for(lock, std::chrono::milliseconds(2));
      }
      if (!running_) return;
    }
    try {
      ResultBatch result = processor_(std::move(frame), worker_index);
      const double latency_ms = result.timings.capture_to_result_ms;
      if (result_callback_) result_callback_(std::move(result));
      std::lock_guard<std::mutex> lock(mutex_);
      auto& slot = sources_.at(source_id);
      ++slot.stats.completed;
      if (!slot.first_completed_ns) slot.first_completed_ns = monotonic_ns();
      slot.latency_samples.push_back(latency_ms);
      if (slot.latency_samples.size() > 512)
        slot.latency_samples.erase(slot.latency_samples.begin(),
                                   slot.latency_samples.begin() + 128);
      if (slot.stats.in_flight_count > 0) --slot.stats.in_flight_count;
      slot.stats.in_flight = slot.stats.in_flight_count > 0;
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& slot = sources_.at(source_id);
      ++slot.stats.errors;
      if (slot.stats.in_flight_count > 0) --slot.stats.in_flight_count;
      slot.stats.in_flight = slot.stats.in_flight_count > 0;
    }
    cv_.notify_all();
  }
}

}  // namespace rkvs
