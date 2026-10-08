#pragma once
#include "types.hpp"
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <string>
#include <vector>

namespace dual {
class PreviewSink {
 public:
  PreviewSink();
  ~PreviewSink();
  bool start();
  void submit(DetectionBatch& batch);
  uint64_t submitted() const { return submitted_; }
  uint64_t processed() const { return processed_; }
  uint64_t written() const { return written_; }
 private:
  struct Task { std::string camera; uint64_t sequence; uint64_t capture_ts_ns; std::vector<Detection> detections; std::vector<uint8_t> image; };
  void process_loop();
  void writer_loop();
  int fd_=-1; int child_=-1; uint64_t last_write_ns_=0;
  unsigned write_index_=0;
  std::atomic<uint64_t> submitted_{0},processed_{0},written_{0};
  std::mutex mutex_;
  std::condition_variable cv_,task_cv_; std::atomic<bool> running_{false}; std::thread writer_,processor_;
  // Keep one newest preview task per camera. A single shared slot made the
  // busier camera continuously overwrite the other one and made the second
  // half of the preview appear frozen.
  std::unique_ptr<Task> cam0_task_, cam1_task_;
  bool prefer_cam0_=true;
  uint64_t cam0_sequence_=0, cam1_sequence_=0;
  std::vector<uint8_t> canvas_, pending_, cam0_, cam1_;
};
}
