#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace dual {

inline uint64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Letterbox {
  int source_width = 0, source_height = 0;
  int target_width = 640, target_height = 640;
  int resized_width = 0, resized_height = 0;
  int pad_left = 0, pad_top = 0;
  float scale = 1.0f;

  static Letterbox make(int sw, int sh, int tw = 640, int th = 640) {
    Letterbox l;
    l.source_width = sw; l.source_height = sh;
    l.target_width = tw; l.target_height = th;
    l.scale = std::min(static_cast<float>(tw) / sw,
                       static_cast<float>(th) / sh);
    l.resized_width = std::max(2, static_cast<int>(sw * l.scale) & ~1);
    l.resized_height = std::max(2, static_cast<int>(sh * l.scale) & ~1);
    l.pad_left = (tw - l.resized_width) / 2;
    l.pad_top = (th - l.resized_height) / 2;
    return l;
  }

  float source_x(float x) const {
    return std::clamp((x - pad_left) / scale, 0.0f,
                      static_cast<float>(source_width - 1));
  }
  float source_y(float y) const {
    return std::clamp((y - pad_top) / scale, 0.0f,
                      static_cast<float>(source_height - 1));
  }
};

struct FrameHandle {
  std::string camera_id;
  uint64_t sequence = 0;
  uint64_t capture_ts_ns = 0;
  uint64_t dequeue_ts_ns = 0;
  int dmabuf_fd = -1;
  int width = 0, height = 0, stride = 0;
};

struct Detection {
  int class_id = -1;
  std::string label;
  float score = 0;
  float left = 0, top = 0, right = 0, bottom = 0;
};

struct StageTimings {
  double capture_dequeue_ms = 0;
  double queue_wait_ms = 0;
  double rga_ms = 0;
  double inference_queue_ms = 0;
  double inference_ms = 0;
  double postprocess_queue_ms = 0;
  double postprocess_ms = 0;
  double capture_to_result_ms = 0;
};

struct DetectionBatch {
  std::string camera_id;
  uint64_t sequence = 0;
  uint64_t capture_ts_ns = 0;
  int worker_id = -1;
  int npu_core = -1;
  int width = 0, height = 0;
  std::vector<Detection> detections;
  StageTimings timings;
  uint64_t dropped_frames = 0, v4l2_timeouts = 0, reconnects = 0;
  std::vector<uint8_t> preview_rgb; // optional 640x360 debug image; never serialized
};

inline std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (unsigned char c : s) {
    switch (c) {
      case '\\': o << "\\\\"; break; case '"': o << "\\\""; break;
      case '\n': o << "\\n"; break; case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default: if (c >= 0x20) o << c;
    }
  }
  return o.str();
}

std::string to_json(const DetectionBatch& b);

}  // namespace dual
