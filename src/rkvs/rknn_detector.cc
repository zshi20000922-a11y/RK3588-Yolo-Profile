#include "rkvs/rknn_detector.hpp"

#include "rkvs/detector_adapter.hpp"

#include "im2d.h"
#include "rknn_api.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rkvs {
namespace {

struct DmaHeapAllocationData {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};
#define RKVS_DMA_HEAP_IOC_ALLOC _IOWR('H', 0x0, DmaHeapAllocationData)

double milliseconds(uint64_t begin, uint64_t end) {
  return static_cast<double>(end - begin) / 1e6;
}

void check(int result, const char* operation) {
  if (result < 0)
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(result));
}

std::vector<unsigned char> read_binary(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open RKNN model: " + path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

int allocate_dma(size_t size) {
  const char* heaps[] = {"/dev/dma_heap/cma", "/dev/dma_heap/system-uncached",
                         "/dev/dma_heap/system"};
  for (const char* path : heaps) {
    int heap = open(path, O_RDWR | O_CLOEXEC);
    if (heap < 0) continue;
    DmaHeapAllocationData allocation{};
    allocation.len = size;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    const int result = ioctl(heap, RKVS_DMA_HEAP_IOC_ALLOC, &allocation);
    close(heap);
    if (result == 0) return static_cast<int>(allocation.fd);
  }
  return -1;
}

TensorDataType data_type(rknn_tensor_type type) {
  if (type == RKNN_TENSOR_INT8) return TensorDataType::kInt8;
  if (type == RKNN_TENSOR_UINT8) return TensorDataType::kUint8;
  if (type == RKNN_TENSOR_FLOAT32) return TensorDataType::kFloat32;
  throw std::invalid_argument("RKNN tensor type is unsupported");
}

TensorLayout layout(const rknn_tensor_attr& attr) {
  if (attr.fmt == RKNN_TENSOR_NCHW) return TensorLayout::kNchw;
  if (attr.fmt == RKNN_TENSOR_NHWC) return TensorLayout::kNhwc;
  if (attr.fmt == RKNN_TENSOR_NC1HWC2) return TensorLayout::kNc1hwc2;
  if (attr.fmt == RKNN_TENSOR_UNDEFINED) return TensorLayout::kFlat;
  throw std::invalid_argument("RKNN tensor layout is unsupported");
}

TensorSchema schema_from(const rknn_tensor_attr& attr) {
  TensorSchema schema;
  schema.name = attr.name;
  schema.type = data_type(attr.type);
  schema.layout = layout(attr);
  schema.zero_point = attr.zp;
  schema.scale = attr.scale;
  for (uint32_t i = 0; i < attr.n_dims; ++i) schema.dims.push_back(attr.dims[i]);
  return schema;
}

}  // namespace

class RknnDetectorPool::Impl {
 public:
  struct Worker {
    rknn_context context = 0;
    int core = 0;
    rknn_tensor_attr input_attr{};
    std::vector<rknn_tensor_attr> output_attrs;
    rknn_tensor_mem* input_mem = nullptr;
    int input_fd = -1;
    void* input_mapping = nullptr;
    size_t input_size = 0;
    std::vector<rknn_tensor_mem*> output_mems;
    std::mutex mutex;

    ~Worker() {
      for (auto* memory : output_mems)
        if (memory && context) rknn_destroy_mem(context, memory);
      if (input_mem && context) rknn_destroy_mem(context, input_mem);
      if (input_mapping) munmap(input_mapping, input_size);
      if (input_fd >= 0) close(input_fd);
      if (context) rknn_destroy(context);
    }
  };

  explicit Impl(const ModelConfig& config)
      : config_(config), manifest_(DetectorManifest::load(config.manifest)),
        labels_(load_labels(config.labels)),
        adapter_(make_detector_adapter(config.family, labels_)) {
    if (manifest_.family != config.family)
      throw std::invalid_argument("service config and manifest family mismatch");
    auto model = read_binary(config.path);
    std::vector<rknn_context> contexts;
    try {
      rknn_context base = 0;
      const uint32_t flags = std::getenv("RKVS_COLLECT_PERF")
                                 ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
      check(rknn_init(&base, model.data(), model.size(), flags, nullptr), "rknn_init");
      contexts.push_back(base);
      for (int index = 1; index < config.contexts; ++index) {
        rknn_context duplicate = 0;
        check(rknn_dup_context(&base, &duplicate), "rknn_dup_context");
        contexts.push_back(duplicate);
      }
      for (int index = 0; index < config.contexts; ++index) {
        auto worker = std::make_unique<Worker>();
        worker->context = contexts[index];
        contexts[index] = 0;
        initialize_worker(worker.get(), index);
        workers_.push_back(std::move(worker));
      }
    } catch (...) {
      for (auto context : contexts) if (context) rknn_destroy(context);
      throw;
    }
  }

  void initialize_worker(Worker* worker, int index) {
    const rknn_core_mask masks[] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1,
                                    RKNN_NPU_CORE_2};
    worker->core = index;
    check(rknn_set_core_mask(worker->context, masks[index % 3]),
          "rknn_set_core_mask");
    rknn_input_output_num count{};
    check(rknn_query(worker->context, RKNN_QUERY_IN_OUT_NUM, &count,
                     sizeof(count)), "query input/output count");
    if (count.n_input != 1) throw std::invalid_argument("model must have one input");
    worker->input_attr.index = 0;
    check(rknn_query(worker->context, RKNN_QUERY_NATIVE_INPUT_ATTR,
                     &worker->input_attr, sizeof(worker->input_attr)),
          "query native input");
    if (worker->input_attr.fmt != RKNN_TENSOR_NHWC ||
        worker->input_attr.n_dims != 4 || worker->input_attr.dims[3] != 3 ||
        (worker->input_attr.type != RKNN_TENSOR_INT8 &&
         worker->input_attr.type != RKNN_TENSOR_UINT8))
      throw std::invalid_argument("RKNN native input must be NHWC 8-bit RGB");
    // The current Model Zoo YOLOv5 graph reports a native INT8 input but its
    // zero-copy API contract accepts the RGA-produced UINT8 RGB tensor. This
    // explicit binding matches Rockchip's pass-through example and avoids a
    // hidden runtime repack in rknn_run.
    worker->input_attr.type = RKNN_TENSOR_UINT8;

    worker->output_attrs.resize(count.n_output);
    for (uint32_t output = 0; output < count.n_output; ++output) {
      auto& attr = worker->output_attrs[output];
      attr.index = output;
      check(rknn_query(worker->context, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &attr,
                       sizeof(attr)), "query native output");
      if (index == 0 && std::getenv("RKVS_PRINT_TENSORS")) {
        std::cerr << "output[" << output << "] name=" << attr.name
                  << " type=" << attr.type << " fmt=" << attr.fmt
                  << " zp=" << attr.zp << " scale=" << attr.scale << " dims=";
        for (uint32_t d = 0; d < attr.n_dims; ++d) std::cerr << attr.dims[d] << (d + 1 == attr.n_dims ? '\n' : 'x');
      }
    }
    if (index == 0) {
      ModelSchema schema;
      schema.input_height = worker->input_attr.dims[1];
      schema.input_width = worker->input_attr.dims[2];
      schema.input_channels = worker->input_attr.dims[3];
      schema.input_layout = TensorLayout::kNhwc;
      schema.input_type = TensorDataType::kUint8;
      for (const auto& attr : worker->output_attrs)
        schema.outputs.push_back(schema_from(attr));
      // An end-to-end output may be emitted with UNDEFINED/flat format. It is
      // not accepted implicitly: YOLO26 remains unavailable until its exact
      // converted RKNN tensor protocol is represented and validated here.
      adapter_->validate(schema, manifest_);
      model_schema_ = std::move(schema);
    }

    worker->input_size = worker->input_attr.size_with_stride
                             ? worker->input_attr.size_with_stride
                             : worker->input_attr.size;
    worker->input_fd = allocate_dma(worker->input_size);
    if (worker->input_fd < 0) throw std::runtime_error("DMA heap allocation failed");
    worker->input_mapping = mmap(nullptr, worker->input_size,
                                 PROT_READ | PROT_WRITE, MAP_SHARED,
                                 worker->input_fd, 0);
    if (worker->input_mapping == MAP_FAILED) {
      worker->input_mapping = nullptr;
      throw std::runtime_error("mmap RKNN input failed");
    }
    std::memset(worker->input_mapping, 114, worker->input_size);
    worker->input_mem = rknn_create_mem_from_fd(
        worker->context, worker->input_fd, worker->input_mapping,
        worker->input_size, 0);
    if (!worker->input_mem) throw std::runtime_error("create RKNN input memory failed");
    for (const auto& attr : worker->output_attrs) {
      auto* memory = rknn_create_mem(worker->context,
          attr.size_with_stride ? attr.size_with_stride : attr.size);
      if (!memory) throw std::runtime_error("create RKNN output memory failed");
      worker->output_mems.push_back(memory);
    }
  }

  ResultBatch process(SharedFrame frame, int worker_index) {
    if (!frame || frame->planes.empty()) throw std::invalid_argument("frame has no plane");
    if (frame->format != PixelFormat::kNv12)
      throw std::invalid_argument("detector currently requires NV12 input");
    if (worker_index < 0 || worker_index >= static_cast<int>(workers_.size()))
      throw std::out_of_range("RKNN worker index");
    auto& worker = *workers_[worker_index];
    std::lock_guard<std::mutex> lock(worker.mutex);
    ResultBatch result;
    result.source_id = frame->source_id;
    result.sequence = frame->sequence;
    result.capture_ts_ns = frame->capture_ts_ns;
    result.width = frame->width;
    result.height = frame->height;
    const uint64_t preprocess_begin = monotonic_ns();
    result.timings.queue_ms = milliseconds(frame->receive_ts_ns, preprocess_begin);
    const int input_height = worker.input_attr.dims[1];
    const int input_width = worker.input_attr.dims[2];
    const auto transform = LetterboxTransform::make(
        frame->width, frame->height, input_width, input_height);
    const int source_stride = frame->planes[0].stride
                                  ? static_cast<int>(frame->planes[0].stride)
                                  : frame->width;
    rga_buffer_t source = wrapbuffer_fd(frame->planes[0].dmabuf_fd,
        frame->width, frame->height, RK_FORMAT_YCbCr_420_SP,
        source_stride, frame->height);
    rga_buffer_t destination = wrapbuffer_fd(worker.input_mem->fd,
        input_width, input_height, RK_FORMAT_RGB_888,
        worker.input_attr.w_stride ? static_cast<int>(worker.input_attr.w_stride) : input_width,
        worker.input_attr.h_stride ? static_cast<int>(worker.input_attr.h_stride) : input_height);
    rga_buffer_t pattern{};
    im_rect source_rect{0, 0, frame->width, frame->height};
    im_rect destination_rect{transform.pad_left, transform.pad_top,
                             transform.resized_width, transform.resized_height};
    im_rect pattern_rect{};
    const IM_STATUS rga_result = improcess(source, destination, pattern,
        source_rect, destination_rect, pattern_rect, IM_SYNC);
    if (rga_result != IM_STATUS_SUCCESS)
      throw std::runtime_error(std::string("RGA letterbox failed: ") +
                               imStrError(rga_result));
    const uint64_t preprocess_end = monotonic_ns();
    result.timings.rga_ms = milliseconds(preprocess_begin, preprocess_end);
    check(rknn_set_io_mem(worker.context, worker.input_mem, &worker.input_attr),
          "bind RKNN input");
    for (size_t output = 0; output < worker.output_mems.size(); ++output)
      check(rknn_set_io_mem(worker.context, worker.output_mems[output],
                            &worker.output_attrs[output]), "bind RKNN output");
    const uint64_t inference_begin = monotonic_ns();
    check(rknn_run(worker.context, nullptr), "rknn_run");
    const uint64_t inference_end = monotonic_ns();
    result.timings.inference_ms = milliseconds(inference_begin, inference_end);
    const uint64_t sync_begin = monotonic_ns();
    for (auto* memory : worker.output_mems)
      check(rknn_mem_sync(worker.context, memory, RKNN_MEMORY_SYNC_FROM_DEVICE),
            "sync RKNN output");
    const uint64_t sync_end = monotonic_ns();
    result.timings.output_sync_ms = milliseconds(sync_begin, sync_end);
    std::vector<TensorView> outputs;
    for (size_t output = 0; output < worker.output_mems.size(); ++output) {
      const auto& attr = worker.output_attrs[output];
      outputs.push_back({schema_from(attr), worker.output_mems[output]->virt_addr,
                         static_cast<size_t>(attr.size_with_stride
                                                 ? attr.size_with_stride
                                                 : attr.size)});
    }
    const uint64_t postprocess_begin = monotonic_ns();
    DetectorParams params;
    {
      std::lock_guard<std::mutex> config_lock(config_mutex_);
      params.confidence = config_.confidence;
      params.nms_iou = config_.nms_iou;
      params.max_detections = config_.max_detections;
      params.class_agnostic_nms = config_.class_agnostic_nms;
      params.class_filter = config_.class_filter;
    }
    result.detections = adapter_->decode(outputs, transform, params);
    const uint64_t end = monotonic_ns();
    result.timings.postprocess_ms = milliseconds(postprocess_begin, end);
    result.timings.capture_to_result_ms = milliseconds(frame->capture_ts_ns, end);
    return result;
  }

  std::string perf_detail(int worker_index) const {
    if (worker_index < 0 || worker_index >= static_cast<int>(workers_.size()))
      throw std::out_of_range("RKNN worker index");
    rknn_perf_detail detail{};
    check(rknn_query(workers_[worker_index]->context, RKNN_QUERY_PERF_DETAIL,
                     &detail, sizeof(detail)), "query RKNN perf detail");
    return detail.perf_data && detail.data_len
               ? std::string(detail.perf_data, detail.data_len) : std::string();
  }

  ModelConfig config_;
  std::mutex config_mutex_;
  DetectorManifest manifest_;
  std::vector<std::string> labels_;
  std::unique_ptr<IDetectorAdapter> adapter_;
  ModelSchema model_schema_;
  std::vector<std::unique_ptr<Worker>> workers_;
};

RknnDetectorPool::RknnDetectorPool(const ModelConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
RknnDetectorPool::~RknnDetectorPool() = default;
int RknnDetectorPool::size() const { return static_cast<int>(impl_->workers_.size()); }
std::vector<int> RknnDetectorPool::cores() const {
  std::vector<int> result;
  result.reserve(impl_->workers_.size());
  for (const auto& worker : impl_->workers_) result.push_back(worker->core);
  return result;
}
void RknnDetectorPool::update_soft_params(const ModelConfig& config) {
  std::lock_guard<std::mutex> lock(impl_->config_mutex_);
  if (config.path != impl_->config_.path || config.family != impl_->config_.family ||
      config.contexts != impl_->config_.contexts)
    throw std::invalid_argument("model topology change requires restart");
  impl_->config_.confidence = config.confidence;
  impl_->config_.nms_iou = config.nms_iou;
  impl_->config_.max_detections = config.max_detections;
  impl_->config_.class_agnostic_nms = config.class_agnostic_nms;
  impl_->config_.class_filter = config.class_filter;
}
ResultBatch RknnDetectorPool::process(SharedFrame frame, int worker_index) {
  return impl_->process(std::move(frame), worker_index);
}
std::string RknnDetectorPool::perf_detail(int worker_index) const {
  return impl_->perf_detail(worker_index);
}

}  // namespace rkvs
