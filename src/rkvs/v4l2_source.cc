#include "rkvs/v4l2_source.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>

namespace rkvs {
namespace {

int xioctl(int fd, unsigned long request, void* argument) {
  int result;
  do result = ioctl(fd, request, argument); while (result < 0 && errno == EINTR);
  return result;
}

uint64_t timeval_ns(const timeval& value) {
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(value.tv_usec) * 1000ULL;
}

}  // namespace

V4L2Source::V4L2Source(SourceConfig config) : config_(std::move(config)) {
  enabled_ = config_.enabled;
  status_.id = config_.id;
  status_.enabled = config_.enabled;
}

V4L2Source::~V4L2Source() { stop(); }

bool V4L2Source::start(FrameCallback callback) {
  if (running_.exchange(true)) return true;
  callback_ = std::move(callback);
  thread_ = std::thread(&V4L2Source::loop, this);
  return true;
}

void V4L2Source::stop() {
  running_ = false;
  if (thread_.joinable()) thread_.join();
  close_stream();
  callback_ = {};
}

void V4L2Source::set_enabled(bool enabled) {
  enabled_ = enabled;
  std::lock_guard<std::mutex> lock(status_mutex_);
  status_.enabled = enabled;
}

SourceStatus V4L2Source::status() const {
  std::lock_guard<std::mutex> lock(status_mutex_);
  SourceStatus copy = status_;
  if (first_frame_ts_ns_ && copy.frames > 1) {
    uint64_t elapsed = monotonic_ns() - first_frame_ts_ns_;
    if (elapsed) copy.fps = (copy.frames - 1) * 1e9 / elapsed;
  }
  return copy;
}

void V4L2Source::set_error(std::string message) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  ++status_.errors;
  status_.last_error = std::move(message);
}

bool V4L2Source::open_stream() {
  fd_ = open(config_.uri.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
  if (fd_ < 0) {
    set_error("open: " + std::string(strerror(errno)));
    return false;
  }
  v4l2_capability capability{};
  if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0 ||
      !(capability.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) ||
      !(capability.device_caps & V4L2_CAP_STREAMING)) {
    set_error("device does not support multiplanar streaming");
    return false;
  }
  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  format.fmt.pix_mp.width = config_.width;
  format.fmt.pix_mp.height = config_.height;
  format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  format.fmt.pix_mp.field = V4L2_FIELD_NONE;
  if (xioctl(fd_, VIDIOC_S_FMT, &format) < 0 ||
      format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 ||
      format.fmt.pix_mp.num_planes < 1 || format.fmt.pix_mp.num_planes > VIDEO_MAX_PLANES ||
      static_cast<int>(format.fmt.pix_mp.width) != config_.width ||
      static_cast<int>(format.fmt.pix_mp.height) != config_.height) {
    set_error("NV12 geometry negotiation failed");
    return false;
  }
  width_ = format.fmt.pix_mp.width;
  height_ = format.fmt.pix_mp.height;
  plane_count_ = format.fmt.pix_mp.num_planes;
  plane_strides_.clear();
  plane_sizes_.clear();
  for (uint32_t plane = 0; plane < plane_count_; ++plane) {
    plane_strides_.push_back(format.fmt.pix_mp.plane_fmt[plane].bytesperline);
    plane_sizes_.push_back(format.fmt.pix_mp.plane_fmt[plane].sizeimage);
  }

  v4l2_streamparm parameters{};
  parameters.type = format.type;
  parameters.parm.capture.timeperframe.numerator = 1;
  parameters.parm.capture.timeperframe.denominator = config_.fps;
  xioctl(fd_, VIDIOC_S_PARM, &parameters);

  v4l2_requestbuffers request{};
  request.type = format.type;
  request.memory = V4L2_MEMORY_MMAP;
  request.count = config_.buffers;
  if (xioctl(fd_, VIDIOC_REQBUFS, &request) < 0 || request.count < 3) {
    set_error("VIDIOC_REQBUFS failed");
    return false;
  }
  buffers_.assign(request.count, {});
  for (uint32_t index = 0; index < request.count; ++index) {
    v4l2_buffer buffer{};
    v4l2_plane planes[VIDEO_MAX_PLANES]{};
    buffer.type = format.type;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    buffer.length = plane_count_;
    buffer.m.planes = planes;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
      set_error("VIDIOC_QUERYBUF failed");
      return false;
    }
    for (uint32_t plane = 0; plane < buffer.length; ++plane) {
      v4l2_exportbuffer export_buffer{};
      export_buffer.type = format.type;
      export_buffer.index = index;
      export_buffer.plane = plane;
      export_buffer.flags = O_CLOEXEC;
      if (xioctl(fd_, VIDIOC_EXPBUF, &export_buffer) < 0) {
        set_error("VIDIOC_EXPBUF failed; CPU-copy fallback is disabled");
        return false;
      }
      buffers_[index].fds.push_back(export_buffer.fd);
    }
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
      set_error("initial VIDIOC_QBUF failed");
      return false;
    }
  }
  auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
    set_error("VIDIOC_STREAMON failed");
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(leases_->mutex);
    leases_->accepting = true;
  }
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_.online = true;
    status_.width = width_;
    status_.height = height_;
    status_.last_error.clear();
  }
  return true;
}

void V4L2Source::drain_requeues() {
  std::deque<uint32_t> indices;
  {
    std::lock_guard<std::mutex> lock(leases_->mutex);
    indices.swap(leases_->requeue);
  }
  for (uint32_t index : indices) {
    v4l2_buffer buffer{};
    v4l2_plane planes[VIDEO_MAX_PLANES]{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    buffer.length = buffers_[index].fds.size();
    buffer.m.planes = planes;
    if (fd_ >= 0 && xioctl(fd_, VIDIOC_QBUF, &buffer) < 0 && errno != ENODEV)
      set_error("VIDIOC_QBUF: " + std::string(strerror(errno)));
  }
}

void V4L2Source::close_stream() {
  {
    std::lock_guard<std::mutex> lock(leases_->mutex);
    leases_->accepting = false;
    leases_->requeue.clear();
  }
  if (fd_ >= 0) {
    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
  }
  for (auto& buffer : buffers_)
    for (int descriptor : buffer.fds)
      if (descriptor >= 0) close(descriptor);
  buffers_.clear();
  plane_count_ = 0;
  plane_strides_.clear();
  plane_sizes_.clear();
  if (fd_ >= 0) close(fd_);
  fd_ = -1;
  std::lock_guard<std::mutex> lock(status_mutex_);
  status_.online = false;
}

void V4L2Source::loop() {
  unsigned backoff_seconds = 1;
  while (running_) {
    if (!enabled_) {
      close_stream();
      usleep(100000);
      continue;
    }
    if (!open_stream()) {
      close_stream();
      {
        std::lock_guard<std::mutex> lock(status_mutex_);
        ++status_.reconnects;
      }
      for (unsigned tick = 0; running_ && tick < backoff_seconds * 10; ++tick)
        usleep(100000);
      backoff_seconds = std::min(5u, backoff_seconds + 1);
      continue;
    }
    backoff_seconds = 1;
    int consecutive_timeouts = 0;
    while (running_ && enabled_) {
      drain_requeues();
      pollfd descriptor{fd_, POLLIN, 0};
      int result = poll(&descriptor, 1, 500);
      if (result == 0) {
        std::lock_guard<std::mutex> lock(status_mutex_);
        ++status_.timeouts;
        if (++consecutive_timeouts >= 4) break;
        continue;
      }
      if (result < 0 && errno == EINTR) continue;
      if (result < 0 || descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) break;
      v4l2_buffer buffer{};
      v4l2_plane planes[VIDEO_MAX_PLANES]{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.length = plane_count_;
      buffer.m.planes = planes;
      if (xioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
        if (errno == EAGAIN) continue;
        set_error("VIDIOC_DQBUF: " + std::string(strerror(errno)));
        break;
      }
      consecutive_timeouts = 0;
      auto frame = std::make_shared<FrameRef>();
      frame->source_id = config_.id;
      frame->sequence = buffer.sequence;
      frame->capture_ts_ns = timeval_ns(buffer.timestamp);
      frame->receive_ts_ns = monotonic_ns();
      frame->format = PixelFormat::kNv12;
      frame->width = width_;
      frame->height = height_;
      std::vector<int> frame_fds;
      bool duplicate_failed = false;
      for (uint32_t plane = 0; plane < buffer.length; ++plane) {
        // A frame owns duplicate descriptors. This is still zero-copy (the
        // DMA allocation is shared), and prevents reconnect/STREAMOFF from
        // invalidating a descriptor while RGA or MPP is using the frame.
        int descriptor = fcntl(buffers_[buffer.index].fds[plane],
                               F_DUPFD_CLOEXEC, 0);
        if (descriptor < 0) {
          duplicate_failed = true;
          break;
        }
        frame_fds.push_back(descriptor);
        frame->planes.push_back({descriptor, planes[plane].data_offset,
                                 plane_strides_[plane], plane_sizes_[plane]});
      }
      if (duplicate_failed) {
        for (int descriptor : frame_fds) close(descriptor);
        set_error("dup DMA-BUF: " + std::string(strerror(errno)));
        v4l2_buffer returned{};
        v4l2_plane returned_planes[VIDEO_MAX_PLANES]{};
        returned.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        returned.memory = V4L2_MEMORY_MMAP;
        returned.index = buffer.index;
        returned.length = plane_count_;
        returned.m.planes = returned_planes;
        xioctl(fd_, VIDIOC_QBUF, &returned);
        continue;
      }
      auto state = leases_;
      uint32_t index = buffer.index;
      frame->lease = std::make_shared<BufferLease>([state, index, frame_fds] {
        for (int descriptor : frame_fds) close(descriptor);
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->accepting) state->requeue.push_back(index);
      });
      {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (!status_.frames) first_frame_ts_ns_ = frame->receive_ts_ns;
        ++status_.frames;
      }
      if (callback_) callback_(std::move(frame));
    }
    close_stream();
    if (running_ && enabled_) {
      std::lock_guard<std::mutex> lock(status_mutex_);
      ++status_.reconnects;
    }
  }
}

}  // namespace rkvs
