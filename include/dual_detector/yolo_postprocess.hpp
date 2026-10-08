#pragma once

#include "types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace dual {

class YoloPostprocessor {
 public:
  explicit YoloPostprocessor(const std::string& labels_path);
  std::vector<Detection> decode(const int8_t* out0, const int8_t* out1,
      const int8_t* out2, const std::vector<int32_t>& zps,
      const std::vector<float>& scales, const Letterbox& letterbox,
      float confidence = 0.25f, float nms = 0.45f) const;
  std::vector<Detection> decode_native(const int8_t* out0, const int8_t* out1,
      const int8_t* out2, const std::vector<int32_t>& zps,
      const std::vector<float>& scales, const Letterbox& letterbox,
      float confidence = 0.25f, float nms = 0.45f) const;
 private:
  std::vector<Detection> decode_layout(const int8_t* out0,
      const int8_t* out1, const int8_t* out2,
      const std::vector<int32_t>& zps, const std::vector<float>& scales,
      const Letterbox& letterbox, float confidence, float nms,
      bool nc1hwc2) const;
  std::vector<std::string> labels_;
};

}  // namespace dual
