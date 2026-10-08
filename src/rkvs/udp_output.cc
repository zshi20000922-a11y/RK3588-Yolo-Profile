#include "rkvs/udp_output.hpp"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace rkvs {
class UdpMpegTsOutput::Impl {
 public:
  explicit Impl(StreamOutputConfig value) : config(std::move(value)) {}
  ~Impl() { stop(); }
  void start() {
    if (running.exchange(true)) return;
    static std::once_flag once;
    std::call_once(once, [] { gst_init(nullptr, nullptr); });
    const std::string media = config.codec == "h265" ? "h265" : "h264";
    const std::string pipeline_text =
        "appsrc name=source is-live=true format=time block=false ! video/x-" + media +
        ",stream-format=byte-stream,alignment=au ! " + media +
        "parse config-interval=-1 ! mpegtsmux alignment=7 ! udpsink host=" +
        config.udp_host + " port=" + std::to_string(config.udp_port) +
        " sync=false async=false";
    GError* error = nullptr;
    pipeline = gst_parse_launch(pipeline_text.c_str(), &error);
    if (!pipeline) {
      std::string message = error ? error->message : "unknown error";
      if (error) g_error_free(error);
      running = false;
      throw std::runtime_error("create UDP pipeline: " + message);
    }
    appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "source");
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
      stop();
      throw std::runtime_error("UDP pipeline failed to enter PLAYING");
    }
    thread = std::thread(&Impl::loop, this);
  }
  void submit(SharedPacket packet) {
    if (!running || !packet || packet->output_id != config.id) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (waiting_for_keyframe && !packet->keyframe) { ++dropped_count; return; }
    if (queue.size() >= 64) {
      dropped_count += queue.size();
      queue.clear();
      waiting_for_keyframe = true;
      if (!packet->keyframe) { ++dropped_count; return; }
    }
    if (packet->keyframe) waiting_for_keyframe = false;
    queue.push_back(std::move(packet));
    cv.notify_one();
  }
  void loop() {
    while (running) {
      SharedPacket packet;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return !running || !queue.empty(); });
        if (!running) break;
        packet = std::move(queue.front());
        queue.pop_front();
      }
      GstBuffer* buffer = gst_buffer_new_allocate(nullptr, packet->data.size(), nullptr);
      gst_buffer_fill(buffer, 0, packet->data.data(), packet->data.size());
      GST_BUFFER_PTS(buffer) = packet->pts_ns;
      GST_BUFFER_DTS(buffer) = packet->pts_ns;
      if (packet->keyframe) GST_BUFFER_FLAG_UNSET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
      else GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
      if (gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer) == GST_FLOW_OK)
        ++packet_count;
      else ++dropped_count;
    }
  }
  void stop() {
    if (!running.exchange(false)) return;
    cv.notify_all();
    if (thread.joinable()) thread.join();
    if (appsrc) gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
    if (pipeline) gst_element_set_state(pipeline, GST_STATE_NULL);
    if (appsrc) gst_object_unref(appsrc);
    if (pipeline) gst_object_unref(pipeline);
    appsrc = nullptr;
    pipeline = nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    queue.clear();
  }
  StreamOutputConfig config;
  GstElement* pipeline = nullptr;
  GstElement* appsrc = nullptr;
  std::atomic<bool> running{false};
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<SharedPacket> queue;
  bool waiting_for_keyframe = false;
  std::atomic<uint64_t> packet_count{0}, dropped_count{0};
};
UdpMpegTsOutput::UdpMpegTsOutput(StreamOutputConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
UdpMpegTsOutput::~UdpMpegTsOutput() = default;
void UdpMpegTsOutput::start() { impl_->start(); }
void UdpMpegTsOutput::stop() { impl_->stop(); }
void UdpMpegTsOutput::submit(SharedPacket packet) { impl_->submit(std::move(packet)); }
uint64_t UdpMpegTsOutput::packets() const { return impl_->packet_count; }
uint64_t UdpMpegTsOutput::dropped() const { return impl_->dropped_count; }
}  // namespace rkvs
