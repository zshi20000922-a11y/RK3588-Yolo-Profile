#include "v4l2_camera.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <iostream>

namespace dual {
namespace {
int xioctl(int fd, unsigned long request, void* arg) {
  int r;
  do { r = ioctl(fd, request, arg); } while (r < 0 && errno == EINTR);
  return r;
}
uint64_t timeval_ns(const timeval& tv) {
  return static_cast<uint64_t>(tv.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(tv.tv_usec) * 1000ULL;
}
}

FrameLease::FrameLease(V4L2Camera* owner, uint32_t index, FrameHandle frame)
    : owner_(owner), index_(index), frame_(std::move(frame)) {}
FrameLease::FrameLease(std::function<void(uint32_t)> release, uint32_t index, FrameHandle frame)
    : owner_(nullptr), index_(index), frame_(std::move(frame)), release_(std::move(release)) {}
FrameLease::~FrameLease() { if (release_) release_(index_); else if (owner_) owner_->requeue(index_); }

V4L2Camera::V4L2Camera(std::string id, std::string device, int width,
                       int height, int fps, unsigned buffers)
    : id_(std::move(id)), device_(std::move(device)),
      requested_width_(width), requested_height_(height), fps_(fps),
      buffer_count_(buffers) {}
V4L2Camera::~V4L2Camera() { stop(); }

bool V4L2Camera::start() {
  if (running_.exchange(true)) return true;
  thread_ = std::thread(&V4L2Camera::loop, this);
  return true;
}
void V4L2Camera::stop() {
  running_ = false;
  if (thread_.joinable()) thread_.join();
  std::shared_ptr<FrameLease> old;
  { std::lock_guard<std::mutex> lk(mutex_); old=std::move(latest_); }
  old.reset();
  close_stream();
}

bool V4L2Camera::restart(int width, int height, int fps) {
  stop();
  requested_width_ = width;
  requested_height_ = height;
  fps_ = fps;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    stats_.captured = 0;
    stats_.first_capture_ts_ns = 0;
    stats_.last_capture_ts_ns = 0;
    stats_.width = 0;
    stats_.height = 0;
    stats_.requested_fps = fps;
  }
  return start();
}

V4L2Camera::Stats V4L2Camera::stats() const {
  std::lock_guard<std::mutex> lk(mutex_); return stats_;
}

std::shared_ptr<FrameLease> V4L2Camera::take_latest() {
  std::lock_guard<std::mutex> lk(mutex_);
  auto out = std::move(latest_); latest_.reset(); return out;
}

void V4L2Camera::set_frame_observer(
    std::function<void(std::shared_ptr<FrameLease>)> observer) {
  std::lock_guard<std::mutex> lock(mutex_);
  observer_ = std::move(observer);
}

void V4L2Camera::requeue(uint32_t index) {
  std::lock_guard<std::mutex> lk(mutex_);
  requeue_.push_back(index);
}

void V4L2Camera::drain_requeues() {
  std::deque<uint32_t> local;
  { std::lock_guard<std::mutex> lk(mutex_); local.swap(requeue_); }
  for (uint32_t index : local) {
    v4l2_buffer b{}; v4l2_plane p{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    b.memory = V4L2_MEMORY_MMAP; b.index = index;
    b.length = 1; b.m.planes = &p;
    if (fd_ >= 0 && xioctl(fd_, VIDIOC_QBUF, &b) < 0 && errno != ENODEV)
      std::cerr << id_ << ": QBUF failed: " << strerror(errno) << '\n';
  }
}

bool V4L2Camera::open_stream() {
  fd_ = open(device_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
  if (fd_ < 0) return false;
  v4l2_capability cap{};
  if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0 ||
      !(cap.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) ||
      !(cap.device_caps & V4L2_CAP_STREAMING)) return false;

  v4l2_format fmt{}; fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.width = requested_width_;
  fmt.fmt.pix_mp.height = requested_height_;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
  fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
  if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0 ||
      fmt.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 ||
      fmt.fmt.pix_mp.num_planes != 1) return false;
  width_ = fmt.fmt.pix_mp.width; height_ = fmt.fmt.pix_mp.height;
  stride_ = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
  if (width_ != requested_width_ || height_ != requested_height_) return false;

  v4l2_streamparm parm{}; parm.type = fmt.type;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = fps_;
  xioctl(fd_, VIDIOC_S_PARM, &parm);

  v4l2_requestbuffers req{}; req.type = fmt.type;
  req.memory = V4L2_MEMORY_MMAP; req.count = buffer_count_;
  if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 3) return false;
  buffers_.assign(req.count, {});
  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer b{}; v4l2_plane p{}; b.type = fmt.type;
    b.memory = V4L2_MEMORY_MMAP; b.index = i; b.length = 1; b.m.planes = &p;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &b) < 0) return false;
    v4l2_exportbuffer exp{}; exp.type = fmt.type; exp.index = i;
    exp.plane = 0; exp.flags = O_CLOEXEC;
    if (xioctl(fd_, VIDIOC_EXPBUF, &exp) < 0) return false;
    buffers_[i].dmabuf_fd = exp.fd;
    if (xioctl(fd_, VIDIOC_QBUF, &b) < 0) return false;
  }
  auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) return false;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    stats_.captured = 0;
    stats_.first_capture_ts_ns = 0;
    stats_.last_capture_ts_ns = 0;
    stats_.width = width_;
    stats_.height = height_;
    stats_.requested_fps = fps_;
  }
  online_ = true;
  std::cerr << id_ << ": online " << width_ << 'x' << height_
            << " stride=" << stride_ << " buffers=" << buffers_.size() << '\n';
  return true;
}

void V4L2Camera::close_stream() {
  online_ = false;
  if (fd_ >= 0) {
    auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(fd_, VIDIOC_STREAMOFF, &type);
  }
  for (auto& b : buffers_) if (b.dmabuf_fd >= 0) close(b.dmabuf_fd);
  buffers_.clear();
  if (fd_ >= 0) close(fd_); fd_ = -1;
  requeue_.clear();
}

void V4L2Camera::loop() {
  unsigned backoff = 1;
  while (running_) {
    if (!open_stream()) {
      close_stream();
      { std::lock_guard<std::mutex> lk(mutex_); ++stats_.reconnects; }
      for (unsigned i = 0; running_ && i < backoff * 10; ++i) usleep(100000);
      backoff = std::min(5u, backoff + 1); continue;
    }
    backoff = 1;
    unsigned consecutive_timeouts = 0;
    while (running_ && online_) {
      drain_requeues();
      pollfd pfd{fd_, POLLIN, 0};
      int pr = poll(&pfd, 1, 500);
      if (pr == 0) {
        {
          std::lock_guard<std::mutex> lk(mutex_); ++stats_.timeouts;
        }
        // A sensor/CSI failure can leave STREAMON successful while no DMA
        // buffer ever completes. Rebuild the entire V4L2 stream instead of
        // remaining "online" forever with an increasing timeout counter.
        if(++consecutive_timeouts>=4) break;
        continue;
      }
      if (pr < 0 && errno == EINTR) continue;
      if (pr < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) break;
      v4l2_buffer b{}; v4l2_plane p{};
      b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
      b.memory = V4L2_MEMORY_MMAP; b.length = 1; b.m.planes = &p;
      if (xioctl(fd_, VIDIOC_DQBUF, &b) < 0) {
        if (errno == EAGAIN) continue; break;
      }
      consecutive_timeouts = 0;
      FrameHandle f{id_, b.sequence, timeval_ns(b.timestamp), monotonic_ns(),
                    buffers_[b.index].dmabuf_fd, width_, height_, stride_};
      {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!stats_.captured) stats_.first_capture_ts_ns = f.capture_ts_ns;
        stats_.last_capture_ts_ns = f.capture_ts_ns;
        ++stats_.captured;
      }
      auto lease = std::make_shared<FrameLease>(this, b.index, std::move(f));
      std::shared_ptr<FrameLease> old;
      std::function<void(std::shared_ptr<FrameLease>)> observer;
      { std::lock_guard<std::mutex> lk(mutex_); old = std::move(latest_);
        latest_ = lease; observer = observer_; if (old) ++stats_.dropped; }
      if (observer) observer(lease);
      old.reset();
    }
    std::shared_ptr<FrameLease> old;
    { std::lock_guard<std::mutex> lk(mutex_); old=std::move(latest_); }
    old.reset();
    close_stream();
    if(running_) {
      {
        std::lock_guard<std::mutex> lk(mutex_); ++stats_.reconnects;
      }
      for(unsigned i=0;running_ && i<10;++i) usleep(100000);
    }
  }
}

}  // namespace dual
