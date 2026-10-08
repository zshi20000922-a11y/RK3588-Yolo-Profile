#pragma once

#include "rkvs/config.hpp"
#include "rkvs/types.hpp"

#include <memory>

namespace rkvs {

class RgaMotionProcessor {
 public:
  explicit RgaMotionProcessor(MotionConfig config);
  ~RgaMotionProcessor();
  RgaMotionProcessor(const RgaMotionProcessor&) = delete;
  RgaMotionProcessor& operator=(const RgaMotionProcessor&) = delete;
  void update_config(const MotionConfig& config);
  ResultBatch process(SharedFrame frame);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rkvs
