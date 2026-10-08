#pragma once

#include "rkvs/config.hpp"
#include "rkvs/video_source.hpp"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rkvs {

class V4L2Source final : public IVideoSource {
 public:
  explicit V4L2Source(SourceConfig config);
  ~V4L2Source() override;
  bool start(FrameCallback callback) override;
  void stop() override;
  void set_enabled(bool enabled) override;
  SourceStatus status() const override;

 private:
  struct Buffer { std::vector<int> fds; };
  struct LeaseState {
    std::mutex mutex;
    std::deque<uint32_t> requeue;
    bool accepting = false;
  };

  bool open_stream();
  void close_stream();
  void loop();
  void drain_requeues();
  void set_error(std::string message);

  SourceConfig config_;
  FrameCallback callback_;
  int fd_ = -1;
  int width_ = 0;
  int height_ = 0;
  uint32_t plane_count_ = 0;
  std::vector<uint32_t> plane_strides_;
  std::vector<uint32_t> plane_sizes_;
  std::vector<Buffer> buffers_;
  std::shared_ptr<LeaseState> leases_ = std::make_shared<LeaseState>();
  std::atomic<bool> running_{false};
  std::atomic<bool> enabled_{true};
  std::thread thread_;
  mutable std::mutex status_mutex_;
  SourceStatus status_;
  uint64_t first_frame_ts_ns_ = 0;
};

}  // namespace rkvs
