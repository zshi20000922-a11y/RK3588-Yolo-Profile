#pragma once

#include "types.hpp"
#include "yolo_postprocess.hpp"
#include "rknn_api.h"
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace dual {

// A job keeps one tensor-memory slot reserved from the beginning of RGA
// preprocessing until CPU postprocessing is complete.  This lets RGA fill one
// slot and the CPU decode another while the NPU is using the context's current
// slot, without ever copying the 4K source frame through the CPU.
struct PipelineJob {
  int worker_id = -1;
  int slot = -1;
  size_t camera_index = 0;
  DetectionBatch batch;
  Letterbox letterbox;
  uint64_t preprocess_queued_ns = 0;
  uint64_t inference_queued_ns = 0;
  uint64_t postprocess_queued_ns = 0;
  bool make_preview = false;
};

class RknnWorker {
 public:
  static constexpr int kTensorSlots = 2;

  RknnWorker(rknn_context context, int id, int core,
             const std::string& labels);
  ~RknnWorker();
  RknnWorker(const RknnWorker&) = delete;

  int id() const { return id_; }
  int try_acquire_slot();
  void release_slot(int slot);
  std::unique_ptr<PipelineJob> preprocess(const FrameHandle& frame,
                                          size_t camera_index, int slot,
                                          bool make_preview);
  void run(PipelineJob& job);
  void postprocess(PipelineJob& job);

 private:
  struct TensorSlot {
    std::atomic<bool> in_use{false};
    rknn_tensor_mem* input_mem = nullptr;
    int input_dmabuf_fd = -1;
    void* input_mapping = nullptr;
    size_t input_mapping_size = 0;
    std::vector<rknn_tensor_mem*> output_mems;
  };

  void check(int ret, const char* operation);
  TensorSlot& checked_slot(int slot);
  int id_ = -1;
  int input_width_ = 640, input_height_ = 640;
  rknn_context ctx_ = 0;
  rknn_input_output_num io_{};
  rknn_tensor_attr input_attr_{};
  std::vector<rknn_tensor_attr> output_attrs_;
  std::array<TensorSlot, kTensorSlots> slots_;
  std::vector<int32_t> zps_;
  std::vector<float> scales_;
  YoloPostprocessor post_;
};

class RknnContextPool {
 public:
  RknnContextPool(const std::string& model, const std::string& labels,
                  int count, bool independent_contexts);
  std::vector<std::unique_ptr<RknnWorker>> take_workers();
 private:
  std::vector<std::unique_ptr<RknnWorker>> workers_;
};

} // namespace dual
