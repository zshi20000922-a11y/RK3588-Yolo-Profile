#include "rkvs/byte_tracker.hpp"
#include "rkvs/config.hpp"
#include "rkvs/fair_scheduler.hpp"
#include "rkvs/frame_hub.hpp"
#include "rkvs/json.hpp"
#include "rkvs/rknn_detector.hpp"
#include "rkvs/unix_servers.hpp"
#ifdef RKVS_HAS_V4L2
#include "rkvs/v4l2_source.hpp"
#endif
#ifdef RKVS_HAS_FRAME_DIFF
#include "rkvs/rga_motion.hpp"
#endif
#ifdef RKVS_HAS_RTSP_INPUT
#include "rkvs/rtsp_mpp_source.hpp"
#endif
#ifdef RKVS_HAS_MOSAIC
#include "rkvs/mosaic_compositor.hpp"
#endif
#ifdef RKVS_HAS_DISPLAY
#include "rkvs/wayland_display.hpp"
#endif
#ifdef RKVS_HAS_MPP_ENCODER
#include "rkvs/mpp_encoder.hpp"
#endif
#ifdef RKVS_HAS_UDP_OUTPUT
#include "rkvs/udp_output.hpp"
#endif
#ifdef RKVS_HAS_RTSP_OUTPUT
#include "rkvs/rtsp_output.hpp"
#endif

#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<bool> running{true};
void signal_handler(int) { running = false; }

std::string field(const std::string& json, const std::string& name) {
  const std::string marker = "\"" + name + "\"";
  auto position = json.find(marker);
  if (position == std::string::npos) return {};
  position = json.find(':', position + marker.size());
  if (position == std::string::npos) return {};
  position = json.find('"', position + 1);
  if (position == std::string::npos) return {};
  const auto end = json.find('"', position + 1);
  return end == std::string::npos ? std::string{} : json.substr(position + 1, end - position - 1);
}

void ensure_parent(const std::string& path) {
  const auto slash = path.find_last_of('/');
  if (slash == std::string::npos || slash == 0) return;
  const std::string directory = path.substr(0, slash);
  if (mkdir(directory.c_str(), 0755) < 0 && errno != EEXIST)
    throw std::runtime_error("cannot create socket directory: " + directory);
}

long rss_kb() {
  std::ifstream statm("/proc/self/statm");
  long total_pages = 0, resident_pages = 0;
  statm >> total_pages >> resident_pages;
  const long page_kb = sysconf(_SC_PAGESIZE) / 1024;
  return resident_pages * page_kb;
}

class TimingWindows {
 public:
  void add(const rkvs::ResultBatch& batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& sample = samples_[batch.source_id];
    ++totals_[batch.source_id];
    sample.push_back(batch.timings);
    if (sample.size() > 512) sample.pop_front();
  }

  std::string json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << '{';
    bool first_source = true;
    for (const auto& item : samples_) {
      if (!first_source) out << ',';
      first_source = false;
      out << '"' << rkvs::json_escape(item.first) << "\":{\"samples\":"
          << totals_.at(item.first) << ',';
      write_metric(out, "queue", item.second, [](const auto& t) { return t.queue_ms; }, true);
      write_metric(out, "rga", item.second, [](const auto& t) { return t.rga_ms; }, false);
      write_metric(out, "inference", item.second, [](const auto& t) { return t.inference_ms; }, false);
      write_metric(out, "output_sync", item.second, [](const auto& t) { return t.output_sync_ms; }, false);
      write_metric(out, "postprocess", item.second, [](const auto& t) { return t.postprocess_ms; }, false);
      write_metric(out, "tracking", item.second, [](const auto& t) { return t.tracking_ms; }, false);
      write_metric(out, "capture_to_result", item.second,
                   [](const auto& t) { return t.capture_to_result_ms; }, false);
      out << '}';
    }
    out << '}';
    return out.str();
  }

 private:
  template <typename Getter>
  static void write_metric(std::ostringstream& out, const char* name,
                           const std::deque<rkvs::StageTimings>& samples,
                           Getter getter, bool first) {
    std::vector<double> values;
    values.reserve(samples.size());
    double sum = 0;
    for (const auto& sample : samples) {
      const double value = getter(sample);
      values.push_back(value);
      sum += value;
    }
    std::sort(values.begin(), values.end());
    auto percentile = [&](double fraction) {
      if (values.empty()) return 0.0;
      const size_t index = std::min(values.size() - 1,
          static_cast<size_t>(fraction * static_cast<double>(values.size() - 1)));
      return values[index];
    };
    if (!first) out << ',';
    out << '"' << name << "\":{\"avg\":" << (values.empty() ? 0.0 : sum / values.size())
        << ",\"p50\":" << percentile(0.50)
        << ",\"p95\":" << percentile(0.95) << '}';
  }

  mutable std::mutex mutex_;
  std::map<std::string, std::deque<rkvs::StageTimings>> samples_;
  std::map<std::string, uint64_t> totals_;
};

void validate_build_features(const rkvs::ServiceConfig& config) {
  for (const auto& source : config.sources) {
#ifndef RKVS_HAS_V4L2
    if (source.type == rkvs::SourceType::kV4l2)
      throw std::runtime_error("config requires V4L2 input, but this binary was built without RKVS_WITH_V4L2");
#endif
#ifndef RKVS_HAS_RTSP_INPUT
    if (source.type == rkvs::SourceType::kRtsp)
      throw std::runtime_error("config requires RTSP input, but this binary was built without RKVS_WITH_RTSP_INPUT");
#endif
  }
#ifndef RKVS_HAS_FRAME_DIFF
  if (config.motion.enabled)
    throw std::runtime_error("config enables frame-diff motion, but this binary was built without RKVS_WITH_FRAME_DIFF");
#endif
#ifndef RKVS_HAS_DISPLAY
  if (config.display.enabled)
    throw std::runtime_error("config enables display, but this binary was built without RKVS_WITH_DISPLAY");
#endif
  if (!config.outputs.empty()) {
#ifndef RKVS_HAS_MOSAIC
    throw std::runtime_error("config enables stream outputs, but this binary was built without mosaic composition");
#endif
#ifndef RKVS_HAS_MPP_ENCODER
    throw std::runtime_error("config enables stream outputs, but this binary was built without RKVS_WITH_MPP_ENCODE");
#endif
  }
  for (const auto& output : config.outputs) {
#ifndef RKVS_HAS_UDP_OUTPUT
    if (output.udp_enabled)
      throw std::runtime_error("config enables UDP output, but this binary was built without RKVS_WITH_UDP_OUTPUT");
#endif
#ifndef RKVS_HAS_RTSP_OUTPUT
    if (output.rtsp_enabled)
      throw std::runtime_error("config enables RTSP output, but this binary was built without RKVS_WITH_RTSP_OUTPUT");
#endif
  }
}
}  // namespace

int main(int argc, char** argv) {
  std::string config_path = "/userdata/rk-vision-service/config.yaml";
  if (argc == 3 && std::string(argv[1]) == "--config") config_path = argv[2];
  else if (argc != 1) {
    std::cerr << "usage: rk_vision_service [--config FILE]\n";
    return 2;
  }
  try {
    auto config = rkvs::ConfigLoader::load(config_path);
    validate_build_features(config);
    ensure_parent(config.results_socket);
    ensure_parent(config.control_socket);
    rkvs::RknnDetectorPool detector(config.model);
    rkvs::JsonlBroadcastServer results(config.results_socket);
    results.start();
    rkvs::LatestFrameHub frame_hub;
#ifdef RKVS_HAS_MOSAIC
    std::unique_ptr<rkvs::RgaMosaicCompositor> compositor;
#endif
#ifdef RKVS_HAS_DISPLAY
    std::unique_ptr<rkvs::WaylandDisplay> display;
#endif
#ifdef RKVS_HAS_MPP_ENCODER
    rkvs::EncodedPacketBus packet_bus;
    std::vector<std::pair<rkvs::StreamOutputConfig,
                          std::unique_ptr<rkvs::MppEncoder>>> encoders;
    std::vector<std::unique_ptr<rkvs::RgaMosaicCompositor>> focus_compositors;
    std::map<std::string, rkvs::RgaMosaicCompositor*> focus_index;
#endif
#ifdef RKVS_HAS_UDP_OUTPUT
    std::vector<std::unique_ptr<rkvs::UdpMpegTsOutput>> udp_outputs;
#endif
#ifdef RKVS_HAS_RTSP_OUTPUT
    std::unique_ptr<rkvs::RtspOutputServer> rtsp_output;
#endif

    std::map<std::string, std::unique_ptr<rkvs::ByteTracker>> trackers;
    std::map<std::string, std::mutex> tracker_mutexes;
    std::mutex config_mutex;
    std::map<std::string, int> source_detect_fps;
    std::vector<rkvs::SchedulerSourceConfig> schedules;
    for (const auto& source : config.sources) {
      trackers.emplace(source.id, std::make_unique<rkvs::ByteTracker>(config.tracker));
      tracker_mutexes.try_emplace(source.id);
      source_detect_fps[source.id] = source.detect_fps;
      schedules.push_back({source.id, source.detect_fps});
    }
    std::mutex motion_mutex;
    std::map<std::string, rkvs::MotionResult> latest_motion;
    std::map<std::string, bool> motion_active;
    TimingWindows timing_windows;
    TimingWindows motion_timing_windows;
    rkvs::LatestFairScheduler scheduler(
        schedules, detector.size(),
        [&](rkvs::SharedFrame frame, int worker) {
          auto batch = detector.process(std::move(frame), worker);
          if (config.tracker.enabled) {
            const auto begin = rkvs::monotonic_ns();
            std::lock_guard<std::mutex> lock(tracker_mutexes.at(batch.source_id));
            batch.detections = trackers.at(batch.source_id)->update(
                batch.detections, batch.capture_ts_ns);
            batch.timings.tracking_ms =
                static_cast<double>(rkvs::monotonic_ns() - begin) / 1e6;
          }
          return batch;
        },
        [&](rkvs::ResultBatch batch) {
          timing_windows.add(batch);
          {
            std::lock_guard<std::mutex> lock(motion_mutex);
            auto motion = latest_motion.find(batch.source_id);
            if (motion != latest_motion.end()) batch.motion = motion->second;
          }
#ifdef RKVS_HAS_MOSAIC
          if (compositor) compositor->update_result(batch);
#ifdef RKVS_HAS_MPP_ENCODER
          for (auto& focus : focus_compositors) focus->update_result(batch);
#endif
#endif
          if (batch.timings.capture_to_result_ms <= config.stale_result_ms)
            results.publish(rkvs::to_json(batch));
        });

#ifdef RKVS_HAS_FRAME_DIFF
    std::unique_ptr<rkvs::RgaMotionProcessor> motion_processor;
    std::unique_ptr<rkvs::LatestFairScheduler> motion_scheduler;
    if (config.motion.enabled) {
      motion_processor = std::make_unique<rkvs::RgaMotionProcessor>(config.motion);
      std::vector<rkvs::SchedulerSourceConfig> motion_schedules;
      for (const auto& source : config.sources) {
        // At the capture-rate target, an equal periodic deadline can alias
        // with V4L2 delivery and skip a frame even when processing finishes
        // within budget. Use an effectively unthrottled scheduler and let the
        // camera cadence be the limiter.
        const int target = config.motion.fps >= source.fps
                               ? 1000 : config.motion.fps;
        motion_schedules.push_back({source.id, target});
      }
      motion_scheduler = std::make_unique<rkvs::LatestFairScheduler>(
          motion_schedules, std::min(2, static_cast<int>(config.sources.size())),
          [&](rkvs::SharedFrame frame, int) {
            return motion_processor->process(std::move(frame));
          },
          [&](rkvs::ResultBatch batch) {
            motion_timing_windows.add(batch);
            bool active = false;
            {
              std::lock_guard<std::mutex> lock(motion_mutex);
              active = batch.motion.active || batch.motion.global_change;
              latest_motion[batch.source_id] = std::move(batch.motion);
              motion_active[batch.source_id] = active;
            }
            rkvs::MotionConfig motion_config;
            std::map<std::string, int> detect_fps_copy;
            std::vector<rkvs::SourceConfig> source_copy;
            {
              std::lock_guard<std::mutex> lock(config_mutex);
              motion_config = config.motion;
              detect_fps_copy = source_detect_fps;
              source_copy = config.sources;
            }
            if (motion_config.gate_detection) {
              std::vector<rkvs::SchedulerSourceConfig> gated;
              for (const auto& source : source_copy) {
                int fps = detect_fps_copy[source.id];
                bool source_active = true;
                {
                  std::lock_guard<std::mutex> lock(motion_mutex);
                  auto found = motion_active.find(source.id);
                  if (found != motion_active.end()) source_active = found->second;
                }
                if (!source_active)
                  fps = std::min(fps, motion_config.idle_detect_fps);
                gated.push_back({source.id, std::max(1, fps)});
              }
              scheduler.update_sources(gated);
            }
          });
    }
#endif

    std::map<std::string, rkvs::IVideoSource*> source_index;
    std::vector<std::unique_ptr<rkvs::IVideoSource>> sources;
    for (const auto& source_config : config.sources) {
#ifdef RKVS_HAS_V4L2
      if (source_config.type == rkvs::SourceType::kV4l2) {
        auto source = std::make_unique<rkvs::V4L2Source>(source_config);
        source_index[source_config.id] = source.get();
        sources.push_back(std::move(source));
        continue;
      }
#endif
#ifdef RKVS_HAS_RTSP_INPUT
      if (source_config.type == rkvs::SourceType::kRtsp) {
        auto source = std::make_unique<rkvs::RtspMppSource>(source_config);
        source_index[source_config.id] = source.get();
        sources.push_back(std::move(source));
        continue;
      }
#endif
      throw std::runtime_error(std::string("source module absent for ") +
                               rkvs::to_string(source_config.type));
    }

#ifdef RKVS_HAS_MOSAIC
    const bool has_mosaic_output = std::any_of(
        config.outputs.begin(), config.outputs.end(),
        [](const auto& value) { return value.kind == "mosaic"; });
    // Focus-only WebRTC must not create an unused full-frame compositor.  At
    // 4K60 this extra RGA consumer competes with both focus pipelines and can
    // eventually stall the media stack under motion-heavy scenes.
    if (config.display.enabled || has_mosaic_output) {
      std::vector<std::string> mosaic_sources = config.display.sources;
      if (mosaic_sources.empty())
        for (const auto& source : config.sources) mosaic_sources.push_back(source.id);
      rkvs::DisplayConfig mosaic_config = config.display;
      if (!config.display.enabled) {
        auto output = std::find_if(config.outputs.begin(), config.outputs.end(),
            [](const auto& value) { return value.kind == "mosaic"; });
        if (output != config.outputs.end()) {
          mosaic_config.width = output->width;
          mosaic_config.height = output->height;
          mosaic_config.fps = output->fps;
        }
      }
      compositor = std::make_unique<rkvs::RgaMosaicCompositor>(
          mosaic_config, mosaic_sources, &frame_hub);
    }
#endif
#ifdef RKVS_HAS_DISPLAY
    if (config.display.enabled) {
      if (config.display.backend != "wayland")
        throw std::runtime_error("this build currently provides Wayland display only");
      display = std::make_unique<rkvs::WaylandDisplay>(
          config.display.width, config.display.height, config.display.fps, true);
    }
#endif

#ifdef RKVS_HAS_MPP_ENCODER
    for (const auto& output : config.outputs) {
      encoders.emplace_back(output,
          std::make_unique<rkvs::MppEncoder>(output, &packet_bus));
#ifdef RKVS_HAS_UDP_OUTPUT
      if (output.udp_enabled) {
        auto sink = std::make_unique<rkvs::UdpMpegTsOutput>(output);
        auto* sink_pointer = sink.get();
        packet_bus.subscribe([sink_pointer](rkvs::SharedPacket packet) {
          sink_pointer->submit(std::move(packet));
        });
        udp_outputs.push_back(std::move(sink));
      }
#endif
      if (output.kind == "focus") {
        rkvs::DisplayConfig focus;
        focus.enabled = false;
        focus.width = output.width;
        focus.height = output.height;
        focus.fps = output.fps;
        focus.sources = {output.source};
        auto focus_compositor = std::make_unique<rkvs::RgaMosaicCompositor>(
            focus, focus.sources, &frame_hub);
        focus_compositor->set_roi_mode("auto");
        focus_index[output.source] = focus_compositor.get();
        focus_compositors.push_back(std::move(focus_compositor));
      }
    }
#endif
#ifdef RKVS_HAS_RTSP_OUTPUT
    if (std::any_of(config.outputs.begin(), config.outputs.end(),
                    [](const auto& output) { return output.rtsp_enabled; })) {
      rtsp_output = std::make_unique<rkvs::RtspOutputServer>(
          config.rtsp_port, config.outputs, [&] {
            for (auto& encoder : encoders) encoder.second->request_idr();
          });
      auto* server = rtsp_output.get();
      packet_bus.subscribe([server](rkvs::SharedPacket packet) {
        server->submit(std::move(packet));
      });
    }
#endif

    rkvs::UnixControlServer control(config.control_socket, [&](const std::string& request) {
      const std::string command = field(request, "command");
      if (command == "pause_detection") {
        scheduler.pause();
        return std::string("{\"ok\":true,\"paused\":true}");
      }
      if (command == "resume_detection") {
        scheduler.resume();
        return std::string("{\"ok\":true,\"paused\":false}");
      }
      if (command == "enable_source" || command == "disable_source") {
        const std::string id = field(request, "source_id");
        auto source = source_index.find(id);
        if (source == source_index.end())
          return std::string("{\"ok\":false,\"error\":\"unknown source\"}");
        source->second->set_enabled(command == "enable_source");
        return std::string("{\"ok\":true}");
      }
      if (command == "set_roi_mode") {
        const std::string id = field(request, "source_id");
        const std::string mode = field(request, "mode");
        auto focus = focus_index.find(id);
        if (focus == focus_index.end())
          return std::string("{\"ok\":false,\"error\":\"unknown ROI source\"}");
        if (!focus->second->set_roi_mode(mode))
          return std::string("{\"ok\":false,\"error\":\"ROI mode must be full or auto\"}");
        return std::string("{\"ok\":true}");
      }
      if (command == "reload_soft_config") {
        auto replacement = rkvs::ConfigLoader::load(config_path);
        std::lock_guard<std::mutex> lock(config_mutex);
        if (replacement.topology_fingerprint() != config.topology_fingerprint())
          return std::string("{\"ok\":false,\"restart_required\":true}");
        detector.update_soft_params(replacement.model);
        for (auto& tracker : trackers) tracker.second->update_config(replacement.tracker);
        std::map<std::string, int> replacement_detect_fps;
        std::vector<rkvs::SchedulerSourceConfig> replacement_schedules;
        for (const auto& source : replacement.sources) {
          replacement_detect_fps[source.id] = source.detect_fps;
          replacement_schedules.push_back({source.id, source.detect_fps});
        }
        scheduler.update_sources(replacement_schedules);
        source_detect_fps = std::move(replacement_detect_fps);
        config.model = replacement.model;
        config.tracker = replacement.tracker;
        config.motion = replacement.motion;
#ifdef RKVS_HAS_FRAME_DIFF
        if (motion_processor) motion_processor->update_config(replacement.motion);
#endif
        return std::string("{\"ok\":true}");
      }
      if (command == "shutdown") {
        running = false;
        return std::string("{\"ok\":true}");
      }
      if (command == "request_idr") {
#ifdef RKVS_HAS_MPP_ENCODER
        for (auto& encoder : encoders) encoder.second->request_idr();
        return std::string("{\"ok\":true}");
#else
        return std::string("{\"ok\":false,\"error\":\"encoder module absent\"}");
#endif
      }
      if (command == "set_focus") {
        return std::string("{\"ok\":false,\"restart_required\":true,"
                           "\"error\":\"focus source hot switch is not enabled in this build\"}");
      }
      if (command == "status") {
        std::ostringstream response;
        response << "{\"ok\":true,\"paused\":" << (scheduler.paused() ? "true" : "false")
                 << ",\"result_clients\":" << results.clients()
                 << ",\"rss_kb\":" << rss_kb()
                 << ",\"npu_cores\":[";
        const auto cores = detector.cores();
        for (size_t index = 0; index < cores.size(); ++index) {
          if (index) response << ',';
          response << cores[index];
        }
        response << ']';
#ifdef RKVS_HAS_MOSAIC
        if (compositor)
          response << ",\"mosaic_frames\":" << compositor->frames()
                   << ",\"mosaic_dropped\":" << compositor->dropped()
                   << ",\"mosaic_rga_ms\":" << compositor->last_rga_ms()
                   << ",\"mosaic_idle\":" << (compositor->idle() ? "true" : "false");
#endif
#ifdef RKVS_HAS_DISPLAY
        if (display)
          response << ",\"displayed\":" << display->displayed()
                   << ",\"display_dropped\":" << display->dropped();
#endif
#ifdef RKVS_HAS_MPP_ENCODER
        bool media_idle = !compositor || compositor->idle();
        for (const auto& focus : focus_compositors)
          media_idle = media_idle && focus->idle();
        response << ",\"media_idle\":" << (media_idle ? "true" : "false")
                 << ",\"focus_idle\":[";
        for (size_t index = 0; index < focus_compositors.size(); ++index) {
          if (index) response << ',';
          response << (focus_compositors[index]->idle() ? "true" : "false");
        }
        response << ']';
        response << ",\"roi\":[";
        for (size_t index = 0; index < focus_compositors.size(); ++index) {
          if (index) response << ',';
          response << focus_compositors[index]->roi_status_json();
        }
        response << ']';
        response << ",\"encoders\":[";
        for (size_t index = 0; index < encoders.size(); ++index) {
          if (index) response << ',';
          response << "{\"id\":\"" << rkvs::json_escape(encoders[index].first.id)
                   << "\",\"encoded\":" << encoders[index].second->encoded()
                   << ",\"dropped\":" << encoders[index].second->dropped()
                   << ",\"mpp_ms\":" << encoders[index].second->last_encode_ms()
                   << '}';
        }
        response << ']';
#endif
#ifdef RKVS_HAS_UDP_OUTPUT
        response << ",\"udp_outputs\":[";
        for (size_t index = 0; index < udp_outputs.size(); ++index) {
          if (index) response << ',';
          response << "{\"packets\":" << udp_outputs[index]->packets()
                   << ",\"dropped\":" << udp_outputs[index]->dropped() << '}';
        }
        response << ']';
#endif
        response << ",\"pipeline_ms\":" << timing_windows.json();
#ifdef RKVS_HAS_FRAME_DIFF
        if (motion_processor)
          response << ",\"motion_pipeline_ms\":" << motion_timing_windows.json();
#endif
#ifdef RKVS_HAS_RTSP_OUTPUT
        if (rtsp_output)
          response << ",\"rtsp_packets\":" << rtsp_output->packets()
                   << ",\"rtsp_dropped\":" << rtsp_output->dropped();
#endif
        response << ",\"sources\":[";
        bool first = true;
        const auto scheduler_stats = scheduler.stats();
        for (const auto& source : sources) {
          if (!first) response << ',';
          first = false;
          const auto status = source->status();
          const auto scheduled = scheduler_stats.at(status.id);
          response << "{\"id\":\"" << rkvs::json_escape(status.id)
                   << "\",\"online\":" << (status.online ? "true" : "false")
                   << ",\"enabled\":" << (status.enabled ? "true" : "false")
                   << ",\"capture_fps\":" << status.fps
                   << ",\"frames\":" << status.frames
                   << ",\"dropped\":" << status.dropped
                   << ",\"timeouts\":" << status.timeouts
                   << ",\"reconnects\":" << status.reconnects
                   << ",\"source_errors\":" << status.errors
                   << ",\"last_error\":\"" << rkvs::json_escape(status.last_error) << '"'
                   << ",\"detection_completed\":" << scheduled.completed
                   << ",\"detection_errors\":" << scheduled.errors
                   << ",\"detection_fps\":" << scheduled.detection_fps
                   << ",\"detection_replaced\":" << scheduled.replaced
                   << ",\"in_flight\":" << scheduled.in_flight_count
                   << ",\"latency_p50_ms\":" << scheduled.latency_p50_ms
                   << ",\"latency_p95_ms\":" << scheduled.latency_p95_ms
                   << ",\"queue_depth\":" << scheduled.queue_depth << '}';
        }
        response << "]}";
        return response.str();
      }
      return std::string("{\"ok\":false,\"error\":\"unknown command\"}");
    });

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    scheduler.start();
#ifdef RKVS_HAS_FRAME_DIFF
    if (motion_scheduler) motion_scheduler->start();
#endif
    control.start();
#ifdef RKVS_HAS_UDP_OUTPUT
    for (auto& output : udp_outputs) output->start();
#endif
#ifdef RKVS_HAS_RTSP_OUTPUT
    if (rtsp_output) rtsp_output->start();
#endif
#ifdef RKVS_HAS_MPP_ENCODER
    for (auto& encoder : encoders) encoder.second->start();
#endif
#ifdef RKVS_HAS_DISPLAY
    if (display) display->start();
#endif
#ifdef RKVS_HAS_MOSAIC
    if (compositor) compositor->start([&](rkvs::SharedFrame frame) {
#ifdef RKVS_HAS_DISPLAY
      if (display) display->submit(frame);
#endif
#ifdef RKVS_HAS_MPP_ENCODER
      for (auto& encoder : encoders)
        if (encoder.first.kind == "mosaic" &&
            encoder.first.width == frame->width && encoder.first.height == frame->height)
          encoder.second->submit(frame);
#endif
    });
#endif
#ifdef RKVS_HAS_MPP_ENCODER
    {
      size_t focus_index = 0;
      for (auto& encoder : encoders) {
        if (encoder.first.kind != "focus") continue;
        auto* encoder_pointer = encoder.second.get();
        focus_compositors[focus_index++]->start(
            [encoder_pointer](rkvs::SharedFrame frame) {
              encoder_pointer->submit(std::move(frame));
            });
      }
    }
#endif
    for (auto& source : sources)
      source->start([&](rkvs::SharedFrame frame) {
        frame_hub.publish(frame);
        scheduler.submit(frame);
#ifdef RKVS_HAS_FRAME_DIFF
        if (motion_scheduler) motion_scheduler->submit(frame);
#endif
      });
    std::cerr << "rk_vision_service started with " << sources.size()
              << " source(s), " << detector.size() << " RKNN context(s)\n";
    while (running) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (auto& source : sources) source->stop();
#ifdef RKVS_HAS_MOSAIC
    if (compositor) compositor->stop();
#endif
#ifdef RKVS_HAS_MPP_ENCODER
    for (auto& focus : focus_compositors) focus->stop();
    for (auto& encoder : encoders) encoder.second->stop();
#endif
#ifdef RKVS_HAS_UDP_OUTPUT
    for (auto& output : udp_outputs) output->stop();
#endif
#ifdef RKVS_HAS_RTSP_OUTPUT
    if (rtsp_output) rtsp_output->stop();
#endif
#ifdef RKVS_HAS_DISPLAY
    if (display) display->stop();
#endif
    frame_hub.close();
    scheduler.stop();
#ifdef RKVS_HAS_FRAME_DIFF
    if (motion_scheduler) motion_scheduler->stop();
#endif
    control.stop();
    results.stop();
  } catch (const std::exception& error) {
    std::cerr << "rk_vision_service: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
