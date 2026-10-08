#pragma once

#include "rkvs/config.hpp"
#include "rkvs/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rkvs {

enum class TensorDataType { kInt8, kUint8, kFloat32 };
enum class TensorLayout { kNchw, kNhwc, kNc1hwc2, kFlat };

struct TensorSchema {
  std::string name;
  TensorDataType type = TensorDataType::kInt8;
  TensorLayout layout = TensorLayout::kNchw;
  std::vector<int> dims;
  int32_t zero_point = 0;
  float scale = 1.0f;
};

struct TensorView {
  TensorSchema schema;
  const void* data = nullptr;
  size_t bytes = 0;

  // Reads a logical NCHW element. NC1HWC2 uses dims [N,C1,H,W,C2].
  float at(int channel, int y, int x) const;
  float scalar(size_t index) const;
  size_t elements() const;
};

struct ModelSchema {
  int input_width = 0;
  int input_height = 0;
  int input_channels = 0;
  TensorLayout input_layout = TensorLayout::kNhwc;
  TensorDataType input_type = TensorDataType::kUint8;
  std::vector<TensorSchema> outputs;
};

struct DetectorParams {
  float confidence = 0.25f;
  float nms_iou = 0.45f;
  int max_detections = 100;
  bool class_agnostic_nms = false;
  std::vector<int> class_filter;
};

struct DetectorManifest {
  ModelFamily family = ModelFamily::kYolov5;
  std::string task = "detect";
  int input_width = 0;
  int input_height = 0;
  std::string input_format = "rgb";
  TensorLayout input_layout = TensorLayout::kNhwc;
  std::string quantization = "int8";
  std::string labels_path;
  DetectorParams postprocess;

  static DetectorManifest load(const std::string& path);
};

class IDetectorAdapter {
 public:
  virtual ~IDetectorAdapter() = default;
  virtual void validate(const ModelSchema& schema,
                        const DetectorManifest& manifest) const = 0;
  virtual std::vector<Detection> decode(
      const std::vector<TensorView>& outputs,
      const LetterboxTransform& transform,
      const DetectorParams& params) const = 0;
};

std::unique_ptr<IDetectorAdapter> make_detector_adapter(
    ModelFamily family, const std::vector<std::string>& labels);
std::vector<std::string> load_labels(const std::string& path);

}  // namespace rkvs
