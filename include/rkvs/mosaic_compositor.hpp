#pragma once

#include "rkvs/config.hpp"
#include "rkvs/frame_hub.hpp"
#include "rkvs/types.hpp"

#include <functional>
#include <memory>

namespace rkvs {

class RgaMosaicCompositor {
 public:
  using OutputCallback = std::function<void(SharedFrame)>;
  RgaMosaicCompositor(DisplayConfig config, std::vector<std::string> source_ids,
                      LatestFrameHub* hub);
  ~RgaMosaicCompositor();
  void start(OutputCallback callback);
  void stop();
  void update_result(const ResultBatch& result);
  uint64_t frames() const;
  uint64_t dropped() const;
  double last_rga_ms() const;
  bool idle() const;
  bool set_roi_mode(const std::string& mode);
  std::string roi_status_json() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rkvs
