#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rkvs {

enum class SourceType { kV4l2, kRtsp };
enum class RtspTransport { kUdp, kTcp };
enum class ModelFamily { kYolov5, kYolov8, kYolo11, kYolo26 };

struct SourceConfig {
  std::string id;
  SourceType type = SourceType::kV4l2;
  std::string uri;
  bool enabled = true;
  int width = 0;
  int height = 0;
  int fps = 30;
  int buffers = 8;
  int detect_fps = 6;
  RtspTransport transport = RtspTransport::kUdp;
};

struct ModelConfig {
  std::string path;
  std::string manifest;
  std::string labels;
  ModelFamily family = ModelFamily::kYolov5;
  int contexts = 2;
  bool auto_select_cores = true;
  float confidence = 0.25f;
  float nms_iou = 0.45f;
  int max_detections = 100;
  bool class_agnostic_nms = false;
  std::vector<int> class_filter;
};

struct TrackerConfig {
  bool enabled = true;
  float high_threshold = 0.50f;
  float low_threshold = 0.10f;
  float match_iou = 0.30f;
  int track_buffer = 30;
};

struct MotionConfig {
  bool enabled = true;
  std::string algorithm = "three_frame";
  // When the motion resolution matches the NV12 source, consume the exported
  // V4L2 DMA-BUF Y plane directly instead of making an RGA copy.
  bool direct_input = true;
  int width = 640;
  int height = 360;
  int fps = 30;
  int pixel_threshold = 24;
  int min_region_pixels = 180;
  float active_ratio = 0.002f;
  float global_change_ratio = 0.55f;
  bool gate_detection = false;
  int idle_detect_fps = 1;
  int mog2_history = 120;
  float mog2_var_threshold = 16.0f;
  bool mog2_detect_shadows = false;
};

struct DisplayConfig {
  bool enabled = true;
  std::string backend = "wayland";
  int width = 1920;
  int height = 1080;
  int fps = 30;
  std::vector<std::string> sources;
};

struct StreamOutputConfig {
  std::string id;
  std::string kind = "mosaic";
  std::string source;
  std::string codec = "h264";
  int width = 1920;
  int height = 1080;
  int fps = 30;
  int bitrate = 8000000;
  int gop = 10;
  bool udp_enabled = false;
  std::string udp_host;
  int udp_port = 0;
  bool rtsp_enabled = true;
  std::string rtsp_path = "/mosaic";
};

struct ServiceConfig {
  int version = 1;
  std::string results_socket = "/run/rk-vision-service/results.sock";
  std::string control_socket = "/run/rk-vision-service/control.sock";
  std::string log_dir = "/userdata/rk-vision-service/logs";
  int stale_result_ms = 250;
  int rtsp_port = 8554;
  std::vector<SourceConfig> sources;
  ModelConfig model;
  TrackerConfig tracker;
  MotionConfig motion;
  DisplayConfig display;
  std::vector<StreamOutputConfig> outputs;

  void validate() const;
  std::string topology_fingerprint() const;
};

class ConfigLoader {
 public:
  static ServiceConfig load(const std::string& path);
};

const char* to_string(SourceType value);
const char* to_string(ModelFamily value);

}  // namespace rkvs
