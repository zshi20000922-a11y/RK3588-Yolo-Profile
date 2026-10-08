#include "rkvs/rga_motion.hpp"

#include "rkvs/frame_difference.hpp"
#include "im2d.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

namespace rkvs {
namespace {
struct DmaAllocation { uint64_t len; uint32_t fd; uint32_t flags; uint64_t heap_flags; };
#define RKVS_MOTION_DMA_ALLOC _IOWR('H', 0x0, DmaAllocation)
int allocate_dma(size_t size) {
  const char* heaps[] = {"/dev/dma_heap/cma", "/dev/dma_heap/system-uncached",
                         "/dev/dma_heap/system"};
  for (const char* name : heaps) {
    int heap = open(name, O_RDWR | O_CLOEXEC);
    if (heap < 0) continue;
    DmaAllocation request{};
    request.len = size;
    request.flags = O_RDWR | O_CLOEXEC;
    const int result = ioctl(heap, RKVS_MOTION_DMA_ALLOC, &request);
    close(heap);
    if (!result) return request.fd;
  }
  return -1;
}
}  // namespace

class RgaMotionProcessor::Impl {
 public:
  struct State {
    explicit State(const MotionConfig& config) : difference(config) {
      if (config.algorithm == "mog2")
        mog2 = cv::createBackgroundSubtractorMOG2(
            config.mog2_history, config.mog2_var_threshold,
            config.mog2_detect_shadows);
    }
    ~State() {
      if (mapping) munmap(mapping, size);
      if (fd >= 0) close(fd);
    }
    int fd = -1;
    void* mapping = nullptr;
    size_t size = 0;
    ThreeFrameDifference difference;
    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2;
    cv::Mat mog2_mask;
    std::mutex mutex;
  };

  explicit Impl(MotionConfig value) : config(std::move(value)) {}

  State& state(const std::string& id) {
    std::lock_guard<std::mutex> lock(states_mutex);
    auto found = states.find(id);
    if (found == states.end()) {
      auto created = std::make_unique<State>(config);
      if (!config.direct_input) {
        created->size = static_cast<size_t>(config.width) * config.height * 3 / 2;
        created->fd = allocate_dma(created->size);
        if (created->fd < 0) throw std::runtime_error("motion DMA allocation failed");
        created->mapping = mmap(nullptr, created->size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, created->fd, 0);
        if (created->mapping == MAP_FAILED) {
          created->mapping = nullptr;
          throw std::runtime_error("motion DMA mmap failed");
        }
      }
      found = states.emplace(id, std::move(created)).first;
    }
    return *found->second;
  }

  MotionConfig config;
  std::mutex config_mutex;
  std::mutex states_mutex;
  std::map<std::string, std::unique_ptr<State>> states;
};

RgaMotionProcessor::RgaMotionProcessor(MotionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
RgaMotionProcessor::~RgaMotionProcessor() = default;

void RgaMotionProcessor::update_config(const MotionConfig& config) {
  std::lock_guard<std::mutex> states_lock(impl_->states_mutex);
  std::lock_guard<std::mutex> config_lock(impl_->config_mutex);
  if (config.width != impl_->config.width || config.height != impl_->config.height ||
      config.algorithm != impl_->config.algorithm ||
      config.direct_input != impl_->config.direct_input ||
      config.mog2_history != impl_->config.mog2_history ||
      config.mog2_var_threshold != impl_->config.mog2_var_threshold ||
      config.mog2_detect_shadows != impl_->config.mog2_detect_shadows)
    throw std::invalid_argument("motion topology/model change requires restart");
  impl_->config = config;
  for (auto& item : impl_->states) item.second->difference.update_config(config);
}

ResultBatch RgaMotionProcessor::process(SharedFrame frame) {
  if (!frame || frame->planes.empty() || frame->format != PixelFormat::kNv12)
    throw std::invalid_argument("motion processor requires NV12 DMA-BUF");
  MotionConfig config;
  {
    std::lock_guard<std::mutex> lock(impl_->config_mutex);
    config = impl_->config;
  }
  ResultBatch result;
  result.source_id = frame->source_id;
  result.sequence = frame->sequence;
  result.capture_ts_ns = frame->capture_ts_ns;
  result.width = frame->width;
  result.height = frame->height;
  if (!config.enabled) return result;
  auto& state = impl_->state(frame->source_id);
  std::lock_guard<std::mutex> state_lock(state.mutex);
  const uint64_t begin = monotonic_ns();
  const int source_stride = frame->planes[0].stride
                                ? static_cast<int>(frame->planes[0].stride)
                                : frame->width;
  const bool direct = config.direct_input && frame->width == config.width &&
                      frame->height == config.height;
  const uint8_t* gray_data = nullptr;
  void* direct_mapping = nullptr;
  size_t direct_mapping_size = 0;
  dma_buf_sync sync{};
  if (direct) {
    const auto& plane = frame->planes[0];
    if (plane.offset >= plane.size)
      throw std::runtime_error("motion input DMA-BUF offset is invalid");
    direct_mapping_size = plane.size;
    direct_mapping = mmap(nullptr, direct_mapping_size, PROT_READ, MAP_SHARED,
                          plane.dmabuf_fd, 0);
    if (direct_mapping == MAP_FAILED)
      throw std::runtime_error("motion input DMA-BUF mmap failed");
    gray_data = static_cast<const uint8_t*>(direct_mapping) + plane.offset;
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    if (ioctl(plane.dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync) != 0)
      throw std::runtime_error("motion input DMA-BUF cache sync start failed");
  } else {
    if (config.direct_input)
      throw std::runtime_error("motion direct_input requires source-sized motion image");
    rga_buffer_t source = wrapbuffer_fd(frame->planes[0].dmabuf_fd,
        frame->width, frame->height, RK_FORMAT_YCbCr_420_SP,
        source_stride, frame->height);
    rga_buffer_t destination = wrapbuffer_fd(state.fd, config.width, config.height,
        RK_FORMAT_YCbCr_420_SP, config.width, config.height);
    rga_buffer_t pattern{};
    im_rect source_rect{0, 0, frame->width, frame->height};
    im_rect destination_rect{0, 0, config.width, config.height};
    im_rect pattern_rect{};
    const IM_STATUS status = improcess(source, destination, pattern, source_rect,
                                       destination_rect, pattern_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS)
      throw std::runtime_error(std::string("RGA motion resize failed: ") +
                               imStrError(status));
    gray_data = static_cast<const uint8_t*>(state.mapping);
  }
  const uint64_t resized = monotonic_ns();
  result.timings.rga_ms = static_cast<double>(resized - begin) / 1e6;
  if (config.algorithm == "mog2") {
    cv::Mat gray(config.height, config.width, CV_8UC1,
                 const_cast<uint8_t*>(gray_data),
                 direct ? source_stride : config.width);
    state.mog2->apply(gray, state.mog2_mask);
    cv::threshold(state.mog2_mask, state.mog2_mask, 200, 255, cv::THRESH_BINARY);
    cv::morphologyEx(state.mog2_mask, state.mog2_mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    const int active_pixels = cv::countNonZero(state.mog2_mask);
    result.motion.ratio = static_cast<float>(active_pixels) /
                          static_cast<float>(config.width * config.height);
    if (result.motion.ratio >= config.global_change_ratio) {
      result.motion.global_change = true;
    } else {
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(state.mog2_mask, contours, cv::RETR_EXTERNAL,
                       cv::CHAIN_APPROX_SIMPLE);
      for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < config.min_region_pixels) continue;
        const cv::Rect box = cv::boundingRect(contour);
        result.motion.regions.push_back({
            {static_cast<float>(box.x), static_cast<float>(box.y),
             static_cast<float>(box.x + box.width),
             static_cast<float>(box.y + box.height)},
            static_cast<uint32_t>(area)});
      }
      result.motion.active = result.motion.ratio >= config.active_ratio &&
                             !result.motion.regions.empty();
    }
  } else {
    result.motion = state.difference.update(
        gray_data, config.width, config.height,
        direct ? source_stride : config.width);
  }
  if (direct) {
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
    if (ioctl(frame->planes[0].dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync) != 0)
      throw std::runtime_error("motion input DMA-BUF cache sync end failed");
    munmap(direct_mapping, direct_mapping_size);
  }
  const float scale_x = static_cast<float>(frame->width) / config.width;
  const float scale_y = static_cast<float>(frame->height) / config.height;
  for (auto& region : result.motion.regions) {
    region.box.left *= scale_x;
    region.box.right *= scale_x;
    region.box.top *= scale_y;
    region.box.bottom *= scale_y;
  }
  result.timings.postprocess_ms =
      static_cast<double>(monotonic_ns() - resized) / 1e6;
  result.timings.capture_to_result_ms =
      static_cast<double>(monotonic_ns() - frame->capture_ts_ns) / 1e6;
  return result;
}

}  // namespace rkvs
