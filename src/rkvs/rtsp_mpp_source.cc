#include "rkvs/rtsp_mpp_source.hpp"

#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/rtsp/gstrtsptransport.h>
#include <gst/video/video-info.h>
#include <gst/video/gstvideometa.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace rkvs {

class RtspMppSource::Impl {
 public:
  explicit Impl(SourceConfig value) : config(std::move(value)) {
    status.id = config.id;
    status.enabled = config.enabled;
    enabled = config.enabled;
  }
  ~Impl() { stop(); }

  static void pad_added(GstElement*, GstPad* pad, gpointer data) {
    static_cast<Impl*>(data)->connect_rtp_pad(pad);
  }
  static GstFlowReturn new_sample(GstAppSink* sink, gpointer data) {
    return static_cast<Impl*>(data)->handle_sample(sink);
  }

  void set_error(const std::string& message) {
    std::lock_guard<std::mutex> lock(status_mutex);
    ++status.errors;
    status.last_error = message;
  }

  void connect_rtp_pad(GstPad* pad) {
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);
    const GstStructure* structure = caps ? gst_caps_get_structure(caps, 0) : nullptr;
    const char* encoding = structure ? gst_structure_get_string(structure, "encoding-name") : nullptr;
    const bool h265 = encoding && (!g_ascii_strcasecmp(encoding, "H265") ||
                                   !g_ascii_strcasecmp(encoding, "HEVC"));
    const bool h264 = encoding && !g_ascii_strcasecmp(encoding, "H264");
    if (!h264 && !h265) {
      if (caps) gst_caps_unref(caps);
      return;
    }
    if (branch_created.exchange(true)) {
      if (caps) gst_caps_unref(caps);
      return;
    }
    if (caps) gst_caps_unref(caps);
    GstElement* depay = gst_element_factory_make(h265 ? "rtph265depay" : "rtph264depay", nullptr);
    GstElement* parser = gst_element_factory_make(h265 ? "h265parse" : "h264parse", nullptr);
    GstElement* decoder = gst_element_factory_make("mppvideodec", nullptr);
    GstElement* filter = gst_element_factory_make("capsfilter", nullptr);
    appsink = gst_element_factory_make("appsink", nullptr);
    if (!depay || !parser || !decoder || !filter || !appsink) {
      set_error("missing depay/parser/mppvideodec/appsink GStreamer element");
      return;
    }
    g_object_set(decoder, "dma-feature", TRUE, "fbc", FALSE, "format", 23, nullptr);
    GstCaps* raw_caps = gst_caps_from_string("video/x-raw(memory:DMABuf),format=NV12");
    g_object_set(filter, "caps", raw_caps, nullptr);
    gst_caps_unref(raw_caps);
    g_object_set(appsink, "max-buffers", 1u, "drop", TRUE, "sync", FALSE,
                 "enable-last-sample", FALSE, nullptr);
    GstAppSinkCallbacks callbacks{};
    callbacks.new_sample = &Impl::new_sample;
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink), &callbacks, this, nullptr);
    gst_bin_add_many(GST_BIN(pipeline), depay, parser, decoder, filter, appsink, nullptr);
    if (!gst_element_link_many(depay, parser, decoder, filter, appsink, nullptr)) {
      set_error("cannot link RTSP MPP decode branch");
      return;
    }
    GstPad* sink_pad = gst_element_get_static_pad(depay, "sink");
    const GstPadLinkReturn linked = gst_pad_link(pad, sink_pad);
    gst_object_unref(sink_pad);
    if (linked != GST_PAD_LINK_OK) {
      set_error("cannot link rtspsrc to depayloader");
      return;
    }
    gst_element_sync_state_with_parent(depay);
    gst_element_sync_state_with_parent(parser);
    gst_element_sync_state_with_parent(decoder);
    gst_element_sync_state_with_parent(filter);
    gst_element_sync_state_with_parent(appsink);
  }

  GstFlowReturn handle_sample(GstAppSink* sink) {
    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_EOS;
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    GstVideoInfo info{};
    if (!buffer || !caps || !gst_video_info_from_caps(&info, caps) ||
        GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12) {
      gst_sample_unref(sample);
      set_error("decoder did not output NV12");
      return GST_FLOW_ERROR;
    }
    const guint memory_count = gst_buffer_n_memory(buffer);
    if (!memory_count) {
      gst_sample_unref(sample);
      return GST_FLOW_ERROR;
    }
    auto frame = std::make_shared<FrameRef>();
    frame->source_id = config.id;
    frame->sequence = ++sequence;
    frame->receive_ts_ns = monotonic_ns();
    frame->format = PixelFormat::kNv12;
    frame->width = GST_VIDEO_INFO_WIDTH(&info);
    frame->height = GST_VIDEO_INFO_HEIGHT(&info);
    for (guint index = 0; index < memory_count; ++index) {
      GstMemory* memory = gst_buffer_peek_memory(buffer, index);
      if (!gst_is_dmabuf_memory(memory)) {
        gst_sample_unref(sample);
        set_error("mppvideodec output is not DMA-BUF; CPU fallback disabled");
        return GST_FLOW_ERROR;
      }
      const int plane = std::min<int>(index, GST_VIDEO_INFO_N_PLANES(&info) - 1);
      frame->planes.push_back({gst_dmabuf_memory_get_fd(memory),
          static_cast<uint32_t>(GST_VIDEO_INFO_PLANE_OFFSET(&info, plane)),
          static_cast<uint32_t>(GST_VIDEO_INFO_PLANE_STRIDE(&info, plane)),
          static_cast<uint32_t>(memory->maxsize)});
    }
    const GstClockTime pts = GST_BUFFER_PTS(buffer);
    if (GST_CLOCK_TIME_IS_VALID(pts)) {
      if (!GST_CLOCK_TIME_IS_VALID(base_pts) || pts < base_pts || pts - base_pts > 5000000000ULL) {
        base_pts = pts;
        base_monotonic_ns = frame->receive_ts_ns;
      }
      frame->capture_ts_ns = base_monotonic_ns + (pts - base_pts);
      if (frame->capture_ts_ns > frame->receive_ts_ns)
        frame->capture_ts_ns = frame->receive_ts_ns;
    } else frame->capture_ts_ns = frame->receive_ts_ns;
    frame->lease = std::make_shared<BufferLease>([sample] { gst_sample_unref(sample); });
    {
      std::lock_guard<std::mutex> lock(status_mutex);
      if (!status.frames) first_frame_ns = frame->receive_ts_ns;
      ++status.frames;
      status.online = true;
      status.width = frame->width;
      status.height = frame->height;
      status.last_error.clear();
    }
    if (callback) callback(std::move(frame));
    return GST_FLOW_OK;
  }

  bool create_pipeline() {
    branch_created = false;
    base_pts = GST_CLOCK_TIME_NONE;
    pipeline = gst_pipeline_new(nullptr);
    source = gst_element_factory_make("rtspsrc", nullptr);
    if (!pipeline || !source) {
      set_error("cannot create rtspsrc pipeline");
      return false;
    }
    const GstRTSPLowerTrans protocols = config.transport == RtspTransport::kTcp
        ? GST_RTSP_LOWER_TRANS_TCP : GST_RTSP_LOWER_TRANS_UDP;
    g_object_set(source, "location", config.uri.c_str(), "latency", 50u,
                 "drop-on-latency", TRUE, "protocols", protocols, nullptr);
    g_signal_connect(source, "pad-added", G_CALLBACK(&Impl::pad_added), this);
    gst_bin_add(GST_BIN(pipeline), source);
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
      set_error("RTSP pipeline failed to enter PLAYING");
      return false;
    }
    return true;
  }

  void destroy_pipeline() {
    if (pipeline) {
      gst_element_set_state(pipeline, GST_STATE_NULL);
      gst_object_unref(pipeline);
    }
    pipeline = nullptr;
    source = nullptr;
    appsink = nullptr;
    std::lock_guard<std::mutex> lock(status_mutex);
    status.online = false;
  }

  void loop() {
    unsigned backoff = 1;
    while (running) {
      if (!enabled) {
        destroy_pipeline();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      if (!create_pipeline()) {
        destroy_pipeline();
      } else {
        backoff = 1;
        const uint64_t connect_begin_ns = monotonic_ns();
        uint64_t last_seen_frames = 0;
        GstBus* bus = gst_element_get_bus(pipeline);
        while (running && enabled) {
          GstMessage* message = gst_bus_timed_pop_filtered(bus, 250 * GST_MSECOND,
              static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
          if (!message) {
            SourceStatus snapshot;
            {
              std::lock_guard<std::mutex> lock(status_mutex);
              snapshot = status;
            }
            if (!snapshot.frames && monotonic_ns() - connect_begin_ns > 5000000000ULL) {
              set_error("RTSP connected but no decoded frame within 5s");
              break;
            }
            if (snapshot.frames != last_seen_frames) {
              last_seen_frames = snapshot.frames;
              backoff = 1;
            }
            continue;
          }
          if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
            GError* error = nullptr;
            gchar* details = nullptr;
            gst_message_parse_error(message, &error, &details);
            set_error(error ? error->message : "GStreamer error");
            if (error) g_error_free(error);
            g_free(details);
          }
          gst_message_unref(message);
          break;
        }
        gst_object_unref(bus);
        destroy_pipeline();
      }
      if (running && enabled) {
        {
          std::lock_guard<std::mutex> lock(status_mutex);
          ++status.reconnects;
        }
        for (unsigned tick = 0; running && tick < backoff * 10; ++tick)
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        backoff = std::min(10u, backoff * 2);
      }
    }
    destroy_pipeline();
  }

  bool start(FrameCallback value) {
    if (running.exchange(true)) return true;
    callback = std::move(value);
    thread = std::thread(&Impl::loop, this);
    return true;
  }
  void stop() {
    running = false;
    if (thread.joinable()) thread.join();
    callback = {};
  }
  SourceStatus get_status() const {
    std::lock_guard<std::mutex> lock(status_mutex);
    auto copy = status;
    if (first_frame_ns && copy.frames > 1) {
      const uint64_t elapsed = monotonic_ns() - first_frame_ns;
      if (elapsed) copy.fps = (copy.frames - 1) * 1e9 / elapsed;
    }
    return copy;
  }

  SourceConfig config;
  FrameCallback callback;
  GstElement* pipeline = nullptr;
  GstElement* source = nullptr;
  GstElement* appsink = nullptr;
  std::atomic<bool> branch_created{false};
  std::atomic<bool> running{false};
  std::atomic<bool> enabled{true};
  std::thread thread;
  mutable std::mutex status_mutex;
  SourceStatus status;
  uint64_t sequence = 0;
  uint64_t first_frame_ns = 0;
  GstClockTime base_pts = GST_CLOCK_TIME_NONE;
  uint64_t base_monotonic_ns = 0;
};

RtspMppSource::RtspMppSource(SourceConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
  static std::once_flag once;
  std::call_once(once, [] { gst_init(nullptr, nullptr); });
}
RtspMppSource::~RtspMppSource() = default;
bool RtspMppSource::start(FrameCallback callback) { return impl_->start(std::move(callback)); }
void RtspMppSource::stop() { impl_->stop(); }
void RtspMppSource::set_enabled(bool value) {
  impl_->enabled = value;
  std::lock_guard<std::mutex> lock(impl_->status_mutex);
  impl_->status.enabled = value;
}
SourceStatus RtspMppSource::status() const { return impl_->get_status(); }

}  // namespace rkvs
