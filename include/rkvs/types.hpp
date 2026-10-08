#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rkvs {

inline uint64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

enum class PixelFormat { kUnknown, kNv12, kNv21, kGray8, kRgb888, kRgba8888 };

struct FramePlane {
  int dmabuf_fd = -1;
  uint32_t offset = 0;
  uint32_t stride = 0;
  uint32_t size = 0;
};

class BufferLease {
 public:
  explicit BufferLease(std::function<void()> release = {})
      : release_(std::move(release)) {}
  ~BufferLease() {
    if (release_) release_();
  }
  BufferLease(const BufferLease&) = delete;
  BufferLease& operator=(const BufferLease&) = delete;

 private:
  std::function<void()> release_;
};

struct FrameRef {
  std::string source_id;
  uint64_t sequence = 0;
  uint64_t capture_ts_ns = 0;
  uint64_t receive_ts_ns = 0;
  PixelFormat format = PixelFormat::kUnknown;
  int width = 0;
  int height = 0;
  std::vector<FramePlane> planes;
  std::shared_ptr<BufferLease> lease;
};

using SharedFrame = std::shared_ptr<const FrameRef>;
using FrameCallback = std::function<void(SharedFrame)>;

struct Box {
  float left = 0;
  float top = 0;
  float right = 0;
  float bottom = 0;

  float width() const { return std::max(0.0f, right - left); }
  float height() const { return std::max(0.0f, bottom - top); }
  float area() const { return width() * height(); }
};

struct Detection {
  int class_id = -1;
  std::string label;
  float score = 0;
  Box box;
  int64_t track_id = -1;
  std::string track_state;
  bool predicted = false;
};

struct MotionRegion {
  Box box;
  uint32_t pixels = 0;
};

struct MotionResult {
  bool active = false;
  bool global_change = false;
  float ratio = 0;
  std::vector<MotionRegion> regions;
};

struct StageTimings {
  double decode_ms = 0;
  double queue_ms = 0;
  double rga_ms = 0;
  double inference_ms = 0;
  double output_sync_ms = 0;
  double postprocess_ms = 0;
  double tracking_ms = 0;
  double capture_to_result_ms = 0;
};

struct ResultBatch {
  std::string source_id;
  uint64_t sequence = 0;
  uint64_t capture_ts_ns = 0;
  int width = 0;
  int height = 0;
  std::vector<Detection> detections;
  MotionResult motion;
  StageTimings timings;
};

struct SourceStatus {
  std::string id;
  bool enabled = true;
  bool online = false;
  uint64_t frames = 0;
  uint64_t dropped = 0;
  uint64_t timeouts = 0;
  uint64_t reconnects = 0;
  uint64_t errors = 0;
  int width = 0;
  int height = 0;
  double fps = 0;
  std::string last_error;
};

struct LetterboxTransform {
  int source_width = 0;
  int source_height = 0;
  int target_width = 0;
  int target_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_left = 0;
  int pad_top = 0;
  float scale = 1;

  static LetterboxTransform make(int sw, int sh, int tw, int th) {
    LetterboxTransform t;
    t.source_width = sw;
    t.source_height = sh;
    t.target_width = tw;
    t.target_height = th;
    t.scale = std::min(static_cast<float>(tw) / sw,
                       static_cast<float>(th) / sh);
    t.resized_width = std::max(2, static_cast<int>(sw * t.scale) & ~1);
    t.resized_height = std::max(2, static_cast<int>(sh * t.scale) & ~1);
    t.pad_left = (tw - t.resized_width) / 2;
    t.pad_top = (th - t.resized_height) / 2;
    return t;
  }

  Box to_source(Box box) const {
    auto x = [&](float value) {
      return std::clamp((value - pad_left) / scale, 0.0f,
                        static_cast<float>(source_width - 1));
    };
    auto y = [&](float value) {
      return std::clamp((value - pad_top) / scale, 0.0f,
                        static_cast<float>(source_height - 1));
    };
    return {x(box.left), y(box.top), x(box.right), y(box.bottom)};
  }
};

}  // namespace rkvs
