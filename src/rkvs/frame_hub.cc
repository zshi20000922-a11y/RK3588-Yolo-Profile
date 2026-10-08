#include "rkvs/frame_hub.hpp"

#include <chrono>
#include <stdexcept>

namespace rkvs {

LatestFrameHub::Consumer::Consumer(LatestFrameHub* hub, std::string name)
    : hub_(hub), name_(std::move(name)) {}

SharedFrame LatestFrameHub::Consumer::take(const std::string& source_id) {
  return hub_ ? hub_->take_after(source_id, &cursors_[source_id], 0, false)
              : nullptr;
}

SharedFrame LatestFrameHub::Consumer::wait_take(const std::string& source_id,
                                                 int timeout_ms) {
  return hub_ ? hub_->take_after(source_id, &cursors_[source_id], timeout_ms,
                                 true)
              : nullptr;
}

std::vector<SharedFrame> LatestFrameHub::Consumer::take_ready() {
  return hub_ ? hub_->take_ready_after(&cursors_) : std::vector<SharedFrame>{};
}

LatestFrameHub::Consumer LatestFrameHub::subscribe(std::string name) {
  if (name.empty()) throw std::invalid_argument("consumer name is empty");
  return Consumer(this, std::move(name));
}

void LatestFrameHub::publish(SharedFrame frame) {
  if (!frame || frame->source_id.empty()) return;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    auto& slot = slots_[frame->source_id];
    slot.frame = std::move(frame);
    ++slot.generation;
    ++slot.publishes;
  }
  cv_.notify_all();
}

void LatestFrameHub::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    for (auto& item : slots_) item.second.frame.reset();
  }
  cv_.notify_all();
}

uint64_t LatestFrameHub::overwritten(const std::string& source_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = slots_.find(source_id);
  return it == slots_.end() || it->second.publishes == 0
             ? 0
             : it->second.publishes - 1;
}

SharedFrame LatestFrameHub::take_after(const std::string& source_id,
                                        uint64_t* cursor, int timeout_ms,
                                        bool wait) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto ready = [&] {
    auto it = slots_.find(source_id);
    return closed_ ||
           (it != slots_.end() && it->second.frame &&
            it->second.generation > *cursor);
  };
  if (wait && !ready())
    cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
  auto it = slots_.find(source_id);
  if (it == slots_.end() || !it->second.frame ||
      it->second.generation <= *cursor)
    return nullptr;
  *cursor = it->second.generation;
  return it->second.frame;
}

std::vector<SharedFrame> LatestFrameHub::take_ready_after(
    std::map<std::string, uint64_t>* cursors) {
  std::vector<SharedFrame> result;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& item : slots_) {
    uint64_t& cursor = (*cursors)[item.first];
    if (item.second.frame && item.second.generation > cursor) {
      cursor = item.second.generation;
      result.push_back(item.second.frame);
    }
  }
  return result;
}

}  // namespace rkvs
