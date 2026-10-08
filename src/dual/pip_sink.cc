#include "pip_sink.hpp"
#include "im2d.h"
#include "rk_mpi.h"
#include "rk_venc_cfg.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <signal.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace dual {
namespace {
struct DmaHeapAllocationData {
  uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags;
};
#define PIP_DMA_HEAP_IOC_ALLOC _IOWR('H', 0x0, DmaHeapAllocationData)

int alloc_dmabuf(size_t size) {
  const char* heaps[] = {"/dev/dma_heap/cma", "/dev/dma_heap/system-uncached",
                         "/dev/dma_heap/system"};
  for (const char* path : heaps) {
    int heap = open(path, O_RDWR | O_CLOEXEC);
    if (heap < 0) continue;
    DmaHeapAllocationData data{};
    data.len = size; data.fd_flags = O_RDWR | O_CLOEXEC;
    int result = ioctl(heap, PIP_DMA_HEAP_IOC_ALLOC, &data);
    close(heap);
    if (result == 0) return static_cast<int>(data.fd);
  }
  return -1;
}

void check_mpp(MPP_RET result, const char* operation) {
  if (result != MPP_OK)
    throw std::runtime_error(std::string(operation) + ": " +
                             std::to_string(result));
}

void write_all(int fd, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  while (size) {
    ssize_t written = write(fd, bytes, size);
    if (written > 0) { bytes += written; size -= written; continue; }
    if (written < 0 && errno == EINTR) continue;
    throw std::runtime_error("PIP H264 pipe write failed");
  }
}

double milliseconds(uint64_t begin, uint64_t end) {
  return static_cast<double>(end - begin) / 1e6;
}
}  // namespace

struct PipSink::Impl {
  int output_fd = -1;
  int dmabuf_fd = -1;
  size_t dmabuf_size = 0;
  pid_t child = -1;
  MppCtx context = nullptr;
  MppApi* api = nullptr;
  MppEncCfg encoder_config = nullptr;
  MppBuffer buffer = nullptr;

  ~Impl() {
    if (output_fd >= 0) close(output_fd);
    if (child > 0) { kill(child, SIGTERM); waitpid(child, nullptr, 0); }
    if (context) mpp_destroy(context);
    if (encoder_config) mpp_enc_cfg_deinit(encoder_config);
    if (buffer) mpp_buffer_put(buffer);
    if (dmabuf_fd >= 0) close(dmabuf_fd);
  }
};

PipSink::PipSink(PipConfig config) : config_(std::move(config)) {}
PipSink::~PipSink() {
  running_ = false; cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  latest_.reset(); impl_.reset();
}

bool PipSink::start() {
  try {
    impl_ = std::make_unique<Impl>();
    impl_->dmabuf_size = config_.width * config_.height * 3 / 2;
    impl_->dmabuf_fd = alloc_dmabuf(impl_->dmabuf_size);
    if (impl_->dmabuf_fd < 0) throw std::runtime_error("PIP DMA allocation failed");
    MppBufferInfo info{};
    info.type = MPP_BUFFER_TYPE_DRM;
    info.size = impl_->dmabuf_size;
    info.fd = impl_->dmabuf_fd;
    check_mpp(mpp_buffer_import(&impl_->buffer, &info), "mpp_buffer_import");

    check_mpp(mpp_create(&impl_->context, &impl_->api), "mpp_create");
    check_mpp(mpp_init(impl_->context, MPP_CTX_ENC, MPP_VIDEO_CodingAVC),
              "mpp_init encoder");
    check_mpp(mpp_enc_cfg_init(&impl_->encoder_config), "mpp_enc_cfg_init");
    check_mpp(impl_->api->control(impl_->context, MPP_ENC_GET_CFG,
                                  impl_->encoder_config), "MPP_ENC_GET_CFG");
    auto cfg = impl_->encoder_config;
    mpp_enc_cfg_set_s32(cfg, "prep:width", config_.width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", config_.height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", config_.width);
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", config_.height);
    mpp_enc_cfg_set_s32(cfg, "prep:format", MPP_FMT_YUV420SP);
    mpp_enc_cfg_set_s32(cfg, "rc:mode", MPP_ENC_RC_MODE_CBR);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_target", config_.bitrate);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_min", config_.bitrate * 7 / 8);
    mpp_enc_cfg_set_s32(cfg, "rc:bps_max", config_.bitrate * 9 / 8);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", config_.fps);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_in_denom", 1);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", config_.fps);
    mpp_enc_cfg_set_s32(cfg, "rc:fps_out_denom", 1);
    mpp_enc_cfg_set_s32(cfg, "rc:gop", 10);
    mpp_enc_cfg_set_s32(cfg, "h264:profile", 100);
    mpp_enc_cfg_set_s32(cfg, "h264:level", 42);
    mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", 1);
    check_mpp(impl_->api->control(impl_->context, MPP_ENC_SET_CFG, cfg),
              "MPP_ENC_SET_CFG");
    MppEncHeaderMode header = MPP_ENC_HEADER_MODE_EACH_IDR;
    check_mpp(impl_->api->control(impl_->context, MPP_ENC_SET_HEADER_MODE,
                                  &header), "MPP_ENC_SET_HEADER_MODE");
    MppPollType timeout = MPP_POLL_BLOCK;
    impl_->api->control(impl_->context, MPP_SET_OUTPUT_TIMEOUT, &timeout);

    int pipes[2];
    if (pipe(pipes) < 0) throw std::runtime_error("PIP pipe failed");
    impl_->child = fork();
    if (impl_->child == 0) {
      close(pipes[1]);
      std::string command =
          "gst-launch-1.0 -q fdsrc fd=" + std::to_string(pipes[0]) +
          " do-timestamp=true ! video/x-h264,stream-format=byte-stream "
          "! h264parse config-interval=-1 ! "
          "video/x-h264,stream-format=byte-stream,alignment=au,framerate=" +
          std::to_string(config_.fps) + "/1 ! mpegtsmux alignment=7 "
          "! tee name=t "
          "t. ! queue max-size-buffers=2 leaky=downstream ! udpsink host=" +
          config_.target_ip + " port=" + std::to_string(config_.udp_port) +
          " sync=false async=false "
          "t. ! queue max-size-buffers=64 ! udpsink host=127.0.0.1 port=" +
          std::to_string(config_.rtsp_feed_port) + " sync=false async=false";
      execl("/bin/sh", "sh", "-c", command.c_str(), (char*)nullptr);
      _exit(127);
    }
    close(pipes[0]);
    if (impl_->child < 0) { close(pipes[1]); throw std::runtime_error("fork failed"); }
    impl_->output_fd = pipes[1];
    running_ = true;
    thread_ = std::thread(&PipSink::loop, this);
    return true;
  } catch (...) {
    impl_.reset();
    throw;
  }
}

void PipSink::submit(std::shared_ptr<FrameLease> frame) {
  if (!running_ || !frame || frame->frame().camera_id != config_.camera_id) return;
  std::shared_ptr<FrameLease> old;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    old = std::move(latest_);
    latest_ = std::move(frame);
  }
  if (old) ++dropped_;
  cv_.notify_one();
}

void PipSink::update_detections(const DetectionBatch& batch) {
  if (!config_.overlay_target || batch.camera_id != config_.camera_id) return;
  const Detection* best = nullptr;
  float best_value = 0;
  for (const auto& detection : batch.detections) {
    if (detection.class_id != 0 || detection.score < 0.40f) continue;
    // A large low-confidence false positive must not hide a smaller, clear
    // person. Target identity/tracking will replace this simple policy later.
    float value = detection.score;
    if (!best || value > best_value) { best = &detection; best_value = value; }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (best) {
    if (has_target_) {
      constexpr float alpha = 0.35f;
      target_.left += alpha * (best->left - target_.left);
      target_.top += alpha * (best->top - target_.top);
      target_.right += alpha * (best->right - target_.right);
      target_.bottom += alpha * (best->bottom - target_.bottom);
      target_.score = best->score;
    } else target_ = *best;
    has_target_ = true;
    target_update_ns_ = monotonic_ns();
  } else if (monotonic_ns() - target_update_ns_ > 500000000ULL) {
    has_target_ = false;
  }
}

void PipSink::loop() {
  try {
  while (running_) {
    std::shared_ptr<FrameLease> lease;
    Detection target;
    bool has_target = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return !running_ || latest_; });
      if (!running_) break;
      lease = std::move(latest_);
      has_target = has_target_ &&
                   monotonic_ns() - target_update_ns_ <= 500000000ULL;
      target = target_;
    }
    // Hardware-composition diagnostic for installations where no person is
    // currently in front of the camera. Removing the marker immediately
    // returns control to YOLO; production behavior is unchanged.
    if (!has_target && access("/run/rknn-pip-test", F_OK) == 0) {
      target.left = 1400; target.top = 600;
      target.right = 2400; target.bottom = 1600;
      has_target = true;
    }
    // Copy metadata because the lease is intentionally released immediately
    // after RGA; keeping a reference here would become dangling before MPP.
    const FrameHandle frame = lease->frame();
    uint64_t rga_begin = monotonic_ns();
    rga_buffer_t source = wrapbuffer_fd(frame.dmabuf_fd, frame.width,
        frame.height, RK_FORMAT_YCbCr_420_SP, frame.stride, frame.height);
    rga_buffer_t output = wrapbuffer_fd(impl_->dmabuf_fd, config_.width,
        config_.height, RK_FORMAT_YCbCr_420_SP, config_.width, config_.height);
    rga_buffer_t pattern{};
    im_rect source_rect{0, 0, frame.width, frame.height};
    im_rect output_rect{0, 0, config_.width, config_.height};
    im_rect pattern_rect{};
    IM_STATUS status = improcess(source, output, pattern, source_rect,
                                 output_rect, pattern_rect, IM_SYNC);
    if (status != IM_STATUS_SUCCESS) { ++dropped_; continue; }
    if (config_.overlay_target && has_target) {
      constexpr int pip_size = 400, margin = 20;
      float box_width = std::max(32.0f, target.right - target.left);
      float box_height = std::max(32.0f, target.bottom - target.top);
      // Crop exactly the detected object plus a small context margin. Do not
      // expand the longest side into a square: a tall person would otherwise
      // turn into an almost full-height/full-frame crop.
      float margin_x = box_width * 0.08f;
      float margin_y = box_height * 0.08f;
      int left = std::clamp(static_cast<int>(target.left - margin_x), 0,
                            frame.width - 2) & ~1;
      int top = std::clamp(static_cast<int>(target.top - margin_y), 0,
                           frame.height - 2) & ~1;
      int right = std::clamp(static_cast<int>(target.right + margin_x),
                             left + 2, frame.width) & ~1;
      int bottom = std::clamp(static_cast<int>(target.bottom + margin_y),
                              top + 2, frame.height) & ~1;
      int crop_width = right - left;
      int crop_height = bottom - top;
      im_rect crop_rect{left, top, crop_width, crop_height};
      // Preserve the target aspect ratio with a variable-size overlay. This
      // needs only one RGA blit after the full-view resize; extra synchronous
      // fill/border operations previously cut the display rate sharply.
      const float scale = std::min(static_cast<float>(pip_size) / crop_width,
                                   static_cast<float>(pip_size) / crop_height);
      int shown_width = std::max(2, static_cast<int>(crop_width * scale) & ~1);
      int shown_height = std::max(2, static_cast<int>(crop_height * scale) & ~1);
      im_rect pip_rect{config_.width - shown_width - margin, margin,
                       shown_width, shown_height};
      status = improcess(source, output, pattern, crop_rect, pip_rect,
                         pattern_rect, IM_SYNC);
      if (status != IM_STATUS_SUCCESS) { ++dropped_; continue; }
    }
    uint64_t rga_done = monotonic_ns();
    lease.reset();  // RGA is done with the 4K V4L2 DMA-BUF.

    MppFrame mpp_frame = nullptr;
    check_mpp(mpp_frame_init(&mpp_frame), "mpp_frame_init");
    mpp_frame_set_width(mpp_frame, config_.width);
    mpp_frame_set_height(mpp_frame, config_.height);
    mpp_frame_set_hor_stride(mpp_frame, config_.width);
    mpp_frame_set_ver_stride(mpp_frame, config_.height);
    mpp_frame_set_fmt(mpp_frame, MPP_FMT_YUV420SP);
    mpp_frame_set_pts(mpp_frame, frame.capture_ts_ns / 1000);
    mpp_frame_set_buffer(mpp_frame, impl_->buffer);
    uint64_t mpp_begin = monotonic_ns();
    MPP_RET put = impl_->api->encode_put_frame(impl_->context, mpp_frame);
    mpp_frame_deinit(&mpp_frame);
    check_mpp(put, "encode_put_frame");
    MppPacket packet = nullptr;
    check_mpp(impl_->api->encode_get_packet(impl_->context, &packet),
              "encode_get_packet");
    if (packet) {
      write_all(impl_->output_fd, mpp_packet_get_pos(packet),
                mpp_packet_get_length(packet));
      mpp_packet_deinit(&packet);
    }
    uint64_t done = monotonic_ns();
    rga_ms_ = milliseconds(rga_begin, rga_done);
    mpp_ms_ = milliseconds(mpp_begin, done);
    ++frames_;
  }
  } catch (const std::exception& error) {
    std::cerr << "PIP worker stopped: " << error.what() << '\n';
    running_ = false;
  }
}

}  // namespace dual
