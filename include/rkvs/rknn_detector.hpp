#pragma once

#include "rkvs/config.hpp"
#include "rkvs/types.hpp"

#include <memory>
#include <string>

namespace rkvs {

// Synchronous per-context processing. Multiple scheduler workers call distinct
// context indices, so RGA, NPU and CPU postprocessing overlap across contexts
// without sharing mutable RKNN state.
class RknnDetectorPool {
 public:
  explicit RknnDetectorPool(const ModelConfig& config);
  ~RknnDetectorPool();
  RknnDetectorPool(const RknnDetectorPool&) = delete;
  RknnDetectorPool& operator=(const RknnDetectorPool&) = delete;

  int size() const;
  std::vector<int> cores() const;
  void update_soft_params(const ModelConfig& config);
  ResultBatch process(SharedFrame frame, int worker_index);
  // Available only when RKVS_COLLECT_PERF was set before construction.
  std::string perf_detail(int worker_index = 0) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rkvs
