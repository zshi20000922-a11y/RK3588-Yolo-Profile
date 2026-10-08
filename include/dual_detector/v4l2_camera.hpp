#pragma once

#include "types.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dual {

class V4L2Camera;

class FrameLease {
 public:
  FrameLease(V4L2Camera* owner, uint32_t index, FrameHandle frame);
  FrameLease(std::function<void(uint32_t)> release, uint32_t index, FrameHandle frame);
  ~FrameLease();
  FrameLease(const FrameLease&) = delete;
  FrameLease& operator=(const FrameLease&) = delete;
  const FrameHandle& frame() const { return frame_; }
 private:
  V4L2Camera* owner_;
  uint32_t index_;
  FrameHandle frame_;
  std::function<void(uint32_t)> release_;
};

class V4L2Camera {
 public:
  struct Stats {
    uint64_t dropped = 0, timeouts = 0, reconnects = 0;
    uint64_t captured = 0, first_capture_ts_ns = 0, last_capture_ts_ns = 0;
    int width = 0, height = 0, requested_fps = 0;
  };
  V4L2Camera(std::string id, std::string device, int width, int height,
             int fps, unsigned buffers);
  ~V4L2Camera();
  bool start();
  void stop();
  bool restart(int width, int height, int fps);
  std::shared_ptr<FrameLease> take_latest();
  void set_frame_observer(
      std::function<void(std::shared_ptr<FrameLease>)> observer);
  void requeue(uint32_t index);
  Stats stats() const;
  const std::string& id() const { return id_; }
  bool online() const { return online_; }

 private:
  struct Buffer { int dmabuf_fd = -1; };
  void loop();
  bool open_stream();
  void close_stream();
  void drain_requeues();

  std::string id_, device_;
  int requested_width_, requested_height_, fps_;
  int width_ = 0, height_ = 0, stride_ = 0;
  unsigned buffer_count_;
  int fd_ = -1;
  std::vector<Buffer> buffers_;
  std::atomic<bool> running_{false}, online_{false};
  std::thread thread_;
  mutable std::mutex mutex_;
  std::shared_ptr<FrameLease> latest_;
  std::function<void(std::shared_ptr<FrameLease>)> observer_;
  std::deque<uint32_t> requeue_;
  Stats stats_;
};

}  // namespace dual
