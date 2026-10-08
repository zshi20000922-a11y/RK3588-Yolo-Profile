#include "rkvs/wayland_display.hpp"

#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/gstvideometa.h>

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace rkvs {

class WaylandDisplay::Impl {
 public:
  Impl(int value_width, int value_height, int value_fps, bool value_fullscreen)
      : width(value_width), height(value_height), fps(value_fps),
        fullscreen(value_fullscreen) {}
  ~Impl() { stop(); }

  void start() {
    if (running.exchange(true)) return;
    static std::once_flag once;
    std::call_once(once, [] { gst_init(nullptr, nullptr); });
    GError* error = nullptr;
    const std::string description =
        "appsrc name=rkvs_source is-live=true format=time block=false "
        "do-timestamp=false ! queue max-size-buffers=1 leaky=downstream ! "
        "waylandsink name=rkvs_display sync=false";
    pipeline = gst_parse_launch(description.c_str(), &error);
    if (!pipeline) {
      std::string message = error ? error->message : "unknown error";
      if (error) g_error_free(error);
      running = false;
      throw std::runtime_error("create Wayland display: " + message);
    }
    appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "rkvs_source");
    GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "rkvs_display");
    g_object_set(sink, "fullscreen", fullscreen ? TRUE : FALSE, nullptr);
    gst_object_unref(sink);
    GstCaps* caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "NV12", "width", G_TYPE_INT, width,
        "height", G_TYPE_INT, height, "framerate", GST_TYPE_FRACTION, fps, 1,
        nullptr);
    GstCapsFeatures* features = gst_caps_features_new("memory:DMABuf", nullptr);
    gst_caps_set_features(caps, 0, features);
    gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
    gst_caps_unref(caps);
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
      stop();
      throw std::runtime_error("Wayland display failed to enter PLAYING");
    }
    thread = std::thread(&Impl::loop, this);
  }

  void submit(SharedFrame value) {
    if (!running || !value || value->planes.empty()) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (latest) ++dropped_count;
    latest = std::move(value);
    cv.notify_one();
  }

  void loop() {
    GstAllocator* allocator = gst_dmabuf_allocator_new();
    while (running) {
      SharedFrame frame;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return !running || latest; });
        if (!running) break;
        frame = std::move(latest);
      }
      if (frame->format != PixelFormat::kNv12 || frame->width != width ||
          frame->height != height) { ++dropped_count; continue; }
      const int duplicate = fcntl(frame->planes[0].dmabuf_fd, F_DUPFD_CLOEXEC, 0);
      if (duplicate < 0) { ++dropped_count; continue; }
      const gsize size = static_cast<gsize>(width) * height * 3 / 2;
      GstMemory* memory = gst_dmabuf_allocator_alloc(allocator, duplicate, size);
      GstBuffer* buffer = gst_buffer_new();
      gst_buffer_append_memory(buffer, memory);
      gsize offsets[GST_VIDEO_MAX_PLANES] = {
          0, static_cast<gsize>(width) * height, 0, 0};
      gint strides[GST_VIDEO_MAX_PLANES] = {width, width, 0, 0};
      gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE,
          GST_VIDEO_FORMAT_NV12, width, height, 2, offsets, strides);
      GST_BUFFER_PTS(buffer) = frame->sequence * GST_SECOND / fps;
      GST_BUFFER_DURATION(buffer) = GST_SECOND / fps;
      const GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
      if (flow == GST_FLOW_OK) ++displayed_count;
      else ++dropped_count;
    }
    gst_object_unref(allocator);
  }

  void stop() {
    if (!running.exchange(false)) return;
    cv.notify_all();
    if (thread.joinable()) thread.join();
    {
      std::lock_guard<std::mutex> lock(mutex);
      latest.reset();
    }
    if (appsrc) gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    if (pipeline) gst_element_set_state(pipeline, GST_STATE_NULL);
    if (appsrc) gst_object_unref(appsrc);
    if (pipeline) gst_object_unref(pipeline);
    appsrc = nullptr;
    pipeline = nullptr;
  }

  int width, height, fps;
  bool fullscreen;
  GstElement* pipeline = nullptr;
  GstElement* appsrc = nullptr;
  std::atomic<bool> running{false};
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cv;
  SharedFrame latest;
  std::atomic<uint64_t> displayed_count{0}, dropped_count{0};
};

WaylandDisplay::WaylandDisplay(int width, int height, int fps, bool fullscreen)
    : impl_(std::make_unique<Impl>(width, height, fps, fullscreen)) {}
WaylandDisplay::~WaylandDisplay() = default;
void WaylandDisplay::start() { impl_->start(); }
void WaylandDisplay::stop() { impl_->stop(); }
void WaylandDisplay::submit(SharedFrame frame) { impl_->submit(std::move(frame)); }
uint64_t WaylandDisplay::displayed() const { return impl_->displayed_count; }
uint64_t WaylandDisplay::dropped() const { return impl_->dropped_count; }

}  // namespace rkvs
