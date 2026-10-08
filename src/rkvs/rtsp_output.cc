#include "rkvs/rtsp_output.hpp"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace rkvs {
class RtspOutputServer::Impl {
 public:
  struct Mount {
    StreamOutputConfig config;
    Impl* owner = nullptr;
    GstElement* appsrc = nullptr;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<SharedPacket> queue;
    bool waiting_for_keyframe = true;
    bool running = false;
    std::thread worker;
    std::atomic<uint64_t> packets{0};
    std::atomic<uint64_t> dropped{0};
    uint64_t first_pts_ns = 0;
  };

  Impl(int value_port, std::vector<StreamOutputConfig> outputs,
       std::function<void()> idr)
      : port(value_port), request_idr(std::move(idr)) {
    for (auto& output : outputs) if (output.rtsp_enabled) {
      auto mount = std::make_unique<Mount>();
      mount->config = std::move(output);
      mount->owner = this;
      mounts.emplace(mount->config.id, std::move(mount));
    }
  }
  ~Impl() { stop(); }

  static void media_configure(GstRTSPMediaFactory*, GstRTSPMedia* media,
                              gpointer data) {
    auto* mount = static_cast<Mount*>(data);
    GstElement* element = gst_rtsp_media_get_element(media);
    GstElement* appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), "rkvs_source");
    gst_object_unref(element);
    if (!appsrc) return;
    g_signal_connect(media, "unprepared", G_CALLBACK(&Impl::media_unprepared), mount);
    {
      std::lock_guard<std::mutex> lock(mount->mutex);
      if (mount->appsrc) gst_object_unref(mount->appsrc);
      mount->appsrc = appsrc;
      mount->waiting_for_keyframe = true;
      mount->first_pts_ns = 0;
      mount->queue.clear();
    }
    if (mount->owner->request_idr) mount->owner->request_idr();
  }

  static void media_unprepared(GstRTSPMedia*, gpointer data) {
    auto* mount = static_cast<Mount*>(data);
    std::lock_guard<std::mutex> lock(mount->mutex);
    if (mount->appsrc) gst_object_unref(mount->appsrc);
    mount->appsrc = nullptr;
    mount->queue.clear();
    mount->waiting_for_keyframe = true;
  }

  void mount_loop(Mount* mount) {
    while (mount->running) {
      SharedPacket packet;
      GstElement* appsrc = nullptr;
      {
        std::unique_lock<std::mutex> lock(mount->mutex);
        mount->cv.wait(lock, [&] { return !mount->running || !mount->queue.empty(); });
        if (!mount->running) break;
        packet = std::move(mount->queue.front());
        mount->queue.pop_front();
        if (mount->appsrc) appsrc = GST_ELEMENT(gst_object_ref(mount->appsrc));
      }
      if (!appsrc) { ++mount->dropped; continue; }
      GstBuffer* buffer = gst_buffer_new_allocate(nullptr, packet->data.size(), nullptr);
      gst_buffer_fill(buffer, 0, packet->data.data(), packet->data.size());
      if (!mount->first_pts_ns) mount->first_pts_ns = packet->pts_ns;
      const uint64_t relative_pts = packet->pts_ns >= mount->first_pts_ns
          ? packet->pts_ns - mount->first_pts_ns : 0;
      GST_BUFFER_PTS(buffer) = relative_pts;
      GST_BUFFER_DTS(buffer) = relative_pts;
      GST_BUFFER_DURATION(buffer) =
          static_cast<GstClockTime>(1000000000ULL / std::max(1, mount->config.fps));
      if (packet->keyframe) GST_BUFFER_FLAG_UNSET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
      else GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
      const GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
      gst_object_unref(appsrc);
      if (flow == GST_FLOW_OK) ++mount->packets;
      else ++mount->dropped;
    }
  }

  void start() {
    if (running.exchange(true)) return;
    static std::once_flag once;
    std::call_once(once, [] { gst_init(nullptr, nullptr); });
    context = g_main_context_new();
    loop = g_main_loop_new(context, FALSE);
    server = gst_rtsp_server_new();
    const std::string service = std::to_string(port);
    gst_rtsp_server_set_service(server, service.c_str());
    GstRTSPMountPoints* points = gst_rtsp_server_get_mount_points(server);
    for (auto& item : mounts) {
      auto& mount = *item.second;
      const std::string media = mount.config.codec == "h265" ? "h265" : "h264";
      const std::string payloader = media == "h265" ? "rtph265pay" : "rtph264pay";
      const std::string launch =
          "( appsrc name=rkvs_source is-live=true format=time block=false "
          "max-bytes=2097152 ! video/x-" + media +
          ",stream-format=byte-stream,alignment=au,framerate=" +
          std::to_string(std::max(1, mount.config.fps)) + "/1 ! " + media +
          "parse config-interval=-1 ! " + payloader +
          " name=pay0 pt=96 config-interval=1 )";
      GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
      gst_rtsp_media_factory_set_launch(factory, launch.c_str());
      gst_rtsp_media_factory_set_shared(factory, TRUE);
      g_signal_connect(factory, "media-configure", G_CALLBACK(&Impl::media_configure),
                       &mount);
      gst_rtsp_mount_points_add_factory(points, mount.config.rtsp_path.c_str(), factory);
      mount.running = true;
      mount.worker = std::thread(&Impl::mount_loop, this, &mount);
    }
    g_object_unref(points);
    g_main_context_push_thread_default(context);
    if (!gst_rtsp_server_attach(server, context)) {
      g_main_context_pop_thread_default(context);
      stop();
      throw std::runtime_error("cannot attach RTSP server");
    }
    g_main_context_pop_thread_default(context);
    main_thread = std::thread([this] { g_main_loop_run(loop); });
  }

  void submit(SharedPacket packet) {
    if (!running || !packet) return;
    auto found = mounts.find(packet->output_id);
    if (found == mounts.end()) return;
    auto& mount = *found->second;
    std::lock_guard<std::mutex> lock(mount.mutex);
    if (!mount.appsrc) return;
    if (mount.waiting_for_keyframe && !packet->keyframe) { ++mount.dropped; return; }
    if (mount.queue.size() >= 64) {
      mount.dropped += mount.queue.size();
      mount.queue.clear();
      mount.waiting_for_keyframe = true;
      if (request_idr) request_idr();
      if (!packet->keyframe) { ++mount.dropped; return; }
    }
    if (packet->keyframe) mount.waiting_for_keyframe = false;
    mount.queue.push_back(std::move(packet));
    mount.cv.notify_one();
  }

  void stop() {
    if (!running.exchange(false)) return;
    for (auto& item : mounts) {
      auto& mount = *item.second;
      {
        std::lock_guard<std::mutex> lock(mount.mutex);
        mount.running = false;
        mount.queue.clear();
      }
      mount.cv.notify_all();
      if (mount.worker.joinable()) mount.worker.join();
      std::lock_guard<std::mutex> lock(mount.mutex);
      if (mount.appsrc) gst_object_unref(mount.appsrc);
      mount.appsrc = nullptr;
    }
    if (loop) g_main_loop_quit(loop);
    if (main_thread.joinable()) main_thread.join();
    if (server) g_object_unref(server);
    if (loop) g_main_loop_unref(loop);
    if (context) g_main_context_unref(context);
    server = nullptr;
    loop = nullptr;
    context = nullptr;
  }

  int port;
  std::function<void()> request_idr;
  std::map<std::string, std::unique_ptr<Mount>> mounts;
  GstRTSPServer* server = nullptr;
  GMainContext* context = nullptr;
  GMainLoop* loop = nullptr;
  std::atomic<bool> running{false};
  std::thread main_thread;
};

RtspOutputServer::RtspOutputServer(int port, std::vector<StreamOutputConfig> outputs,
    std::function<void()> request_idr)
    : impl_(std::make_unique<Impl>(port, std::move(outputs), std::move(request_idr))) {}
RtspOutputServer::~RtspOutputServer() = default;
void RtspOutputServer::start() { impl_->start(); }
void RtspOutputServer::stop() { impl_->stop(); }
void RtspOutputServer::submit(SharedPacket packet) { impl_->submit(std::move(packet)); }
uint64_t RtspOutputServer::packets() const {
  uint64_t result = 0;
  for (const auto& item : impl_->mounts) result += item.second->packets;
  return result;
}
uint64_t RtspOutputServer::dropped() const {
  uint64_t result = 0;
  for (const auto& item : impl_->mounts) result += item.second->dropped;
  return result;
}
}  // namespace rkvs
