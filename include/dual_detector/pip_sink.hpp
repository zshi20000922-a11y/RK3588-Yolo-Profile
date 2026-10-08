#pragma once

#include "types.hpp"
#include "v4l2_camera.hpp"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace dual {

struct PipConfig {
  std::string target_ip;
  std::string camera_id = "cam0";
  bool overlay_target = true;
  int udp_port = 5600;
  int rtsp_feed_port = 7000;
  int width = 1280;
  int height = 720;
  int fps = 60;
  int bitrate = 8000000;
};

// A zero-copy 4K -> RGA compositor -> MPP encoder. The capture observer and
// detection callback only replace latest-value slots; all expensive work is
// confined to one independent thread and can never block YOLO.
class PipSink {
 public:
  explicit PipSink(PipConfig config);
  ~PipSink();
  PipSink(const PipSink&) = delete;
  bool start();
  void submit(std::shared_ptr<FrameLease> frame);
  void update_detections(const DetectionBatch& batch);
  uint64_t frames() const { return frames_; }
  uint64_t dropped() const { return dropped_; }
  double rga_ms() const { return rga_ms_.load(); }
  double mpp_ms() const { return mpp_ms_.load(); }

 private:
  struct Impl;
  void loop();
  PipConfig config_;
  std::unique_ptr<Impl> impl_;
  std::atomic<bool> running_{false};
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::shared_ptr<FrameLease> latest_;
  Detection target_;
  bool has_target_ = false;
  uint64_t target_update_ns_ = 0;
  std::atomic<uint64_t> frames_{0}, dropped_{0};
  std::atomic<double> rga_ms_{0}, mpp_ms_{0};
};

}  // namespace dual
