#pragma once

#include "rkvs/config.hpp"
#include "rkvs/types.hpp"

#include <cstdint>
#include <vector>

namespace rkvs {

class ThreeFrameDifference {
 public:
  explicit ThreeFrameDifference(MotionConfig config = {});
  void update_config(const MotionConfig& config);
  MotionResult update(const uint8_t* gray, int width, int height, int stride);
  void reset();

 private:
  MotionResult regions_from_mask(const std::vector<uint8_t>& mask,
                                 int width, int height) const;
  MotionConfig config_;
  std::vector<uint8_t> previous2_;
  std::vector<uint8_t> previous1_;
  std::vector<uint8_t> current_;
  std::vector<uint8_t> delta1_;
  std::vector<uint8_t> delta2_;
  std::vector<uint8_t> mask_;
  std::vector<uint8_t> opened_;
};

}  // namespace rkvs
