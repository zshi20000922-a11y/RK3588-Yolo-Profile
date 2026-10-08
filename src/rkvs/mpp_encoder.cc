#include "rkvs/mpp_encoder.hpp"

#include "rk_mpi.h"
#include "rk_venc_cfg.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace rkvs {
namespace {
void check(MPP_RET result, const char* operation) {
  if (result != MPP_OK)
    throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
}  // namespace

class MppEncoder::Impl {
 public:
  Impl(StreamOutputConfig value, EncodedPacketBus* value_bus)
      : config(std::move(value)), bus(value_bus) {
    if (!bus) throw std::invalid_argument("encoded packet bus required");
  }
  ~Impl() { stop(); cleanup(); }

  void initialize() {
    const MppCodingType coding = config.codec == "h265"
        ? MPP_VIDEO_CodingHEVC : MPP_VIDEO_CodingAVC;
    check(mpp_create(&context, &api), "mpp_create encoder");
    check(mpp_init(context, MPP_CTX_ENC, coding), "mpp_init encoder");
    check(mpp_enc_cfg_init(&encoder_config), "mpp_enc_cfg_init");
    check(api->control(context, MPP_ENC_GET_CFG, encoder_config), "MPP_ENC_GET_CFG");
    mpp_enc_cfg_set_s32(encoder_config, "prep:width", config.width);
    mpp_enc_cfg_set_s32(encoder_config, "prep:height", config.height);
    mpp_enc_cfg_set_s32(encoder_config, "prep:hor_stride", config.width);
    mpp_enc_cfg_set_s32(encoder_config, "prep:ver_stride", config.height);
    mpp_enc_cfg_set_s32(encoder_config, "prep:format", MPP_FMT_YUV420SP);
    mpp_enc_cfg_set_s32(encoder_config, "rc:mode", MPP_ENC_RC_MODE_CBR);
    mpp_enc_cfg_set_s32(encoder_config, "rc:bps_target", config.bitrate);
    mpp_enc_cfg_set_s32(encoder_config, "rc:bps_min", config.bitrate * 7 / 8);
    mpp_enc_cfg_set_s32(encoder_config, "rc:bps_max", config.bitrate * 9 / 8);
    mpp_enc_cfg_set_s32(encoder_config, "rc:fps_in_num", config.fps);
    mpp_enc_cfg_set_s32(encoder_config, "rc:fps_in_denom", 1);
    mpp_enc_cfg_set_s32(encoder_config, "rc:fps_out_num", config.fps);
    mpp_enc_cfg_set_s32(encoder_config, "rc:fps_out_denom", 1);
    mpp_enc_cfg_set_s32(encoder_config, "rc:gop", config.gop);
    if (coding == MPP_VIDEO_CodingAVC) {
      mpp_enc_cfg_set_s32(encoder_config, "h264:profile", 100);
      mpp_enc_cfg_set_s32(encoder_config, "h264:level", 42);
      mpp_enc_cfg_set_s32(encoder_config, "h264:cabac_en", 1);
    }
    check(api->control(context, MPP_ENC_SET_CFG, encoder_config), "MPP_ENC_SET_CFG");
    MppEncHeaderMode header = MPP_ENC_HEADER_MODE_EACH_IDR;
    check(api->control(context, MPP_ENC_SET_HEADER_MODE, &header),
          "MPP_ENC_SET_HEADER_MODE");
    MppPollType timeout = MPP_POLL_BLOCK;
    api->control(context, MPP_SET_OUTPUT_TIMEOUT, &timeout);
  }

  void cleanup() {
    if (encoder_config) mpp_enc_cfg_deinit(encoder_config);
    encoder_config = nullptr;
    if (context) mpp_destroy(context);
    context = nullptr;
    api = nullptr;
  }

  void loop() {
    try {
      while (running) {
        SharedFrame frame;
        {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait(lock, [&] { return !running || latest; });
          if (!running) break;
          frame = std::move(latest);
        }
        if (frame->format != PixelFormat::kNv12 || frame->planes.empty() ||
            frame->width != config.width || frame->height != config.height) {
          ++dropped_count;
          continue;
        }
        MppBuffer buffer = nullptr;
        MppBufferInfo info{};
        info.type = MPP_BUFFER_TYPE_DRM;
        info.size = static_cast<size_t>(config.width) * config.height * 3 / 2;
        info.fd = frame->planes[0].dmabuf_fd;
        check(mpp_buffer_import(&buffer, &info), "mpp_buffer_import");
        MppFrame mpp_frame = nullptr;
        check(mpp_frame_init(&mpp_frame), "mpp_frame_init");
        mpp_frame_set_width(mpp_frame, config.width);
        mpp_frame_set_height(mpp_frame, config.height);
        mpp_frame_set_hor_stride(mpp_frame, config.width);
        mpp_frame_set_ver_stride(mpp_frame, config.height);
        mpp_frame_set_fmt(mpp_frame, MPP_FMT_YUV420SP);
        mpp_frame_set_pts(mpp_frame, frame->capture_ts_ns / 1000);
        mpp_frame_set_buffer(mpp_frame, buffer);
        if (force_idr.exchange(false)) api->control(context, MPP_ENC_SET_IDR_FRAME, nullptr);
        const uint64_t begin = monotonic_ns();
        MPP_RET result = api->encode_put_frame(context, mpp_frame);
        mpp_frame_deinit(&mpp_frame);
        check(result, "encode_put_frame");
        MppPacket packet = nullptr;
        check(api->encode_get_packet(context, &packet), "encode_get_packet");
        if (packet) {
          auto output = std::make_shared<EncodedPacket>();
          output->output_id = config.id;
          output->codec = config.codec;
          output->sequence = ++sequence;
          output->pts_ns = frame->capture_ts_ns;
          output->keyframe = sequence == 1 || ((sequence - 1) % config.gop == 0);
          const auto* bytes = static_cast<const uint8_t*>(mpp_packet_get_pos(packet));
          output->data.assign(bytes, bytes + mpp_packet_get_length(packet));
          mpp_packet_deinit(&packet);
          bus->publish(std::move(output));
          ++encoded_count;
        }
        mpp_buffer_put(buffer);
        last_ms = static_cast<double>(monotonic_ns() - begin) / 1e6;
      }
    } catch (...) {
      ++errors;
      running = false;
    }
  }

  void start() {
    if (running.exchange(true)) return;
    try { initialize(); }
    catch (...) { running = false; cleanup(); throw; }
    thread = std::thread(&Impl::loop, this);
  }
  void stop() {
    if (!running.exchange(false)) return;
    cv.notify_all();
    if (thread.joinable()) thread.join();
    std::lock_guard<std::mutex> lock(mutex);
    latest.reset();
  }
  void submit(SharedFrame frame) {
    if (!running || !frame) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (latest) ++dropped_count;
    latest = std::move(frame);
    cv.notify_one();
  }

  StreamOutputConfig config;
  EncodedPacketBus* bus;
  MppCtx context = nullptr;
  MppApi* api = nullptr;
  MppEncCfg encoder_config = nullptr;
  std::atomic<bool> running{false}, force_idr{false};
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cv;
  SharedFrame latest;
  uint64_t sequence = 0;
  std::atomic<uint64_t> encoded_count{0}, dropped_count{0}, errors{0};
  std::atomic<double> last_ms{0};
};

MppEncoder::MppEncoder(StreamOutputConfig config, EncodedPacketBus* bus)
    : impl_(std::make_unique<Impl>(std::move(config), bus)) {}
MppEncoder::~MppEncoder() = default;
void MppEncoder::start() { impl_->start(); }
void MppEncoder::stop() { impl_->stop(); }
void MppEncoder::submit(SharedFrame frame) { impl_->submit(std::move(frame)); }
void MppEncoder::request_idr() { impl_->force_idr = true; }
uint64_t MppEncoder::encoded() const { return impl_->encoded_count; }
uint64_t MppEncoder::dropped() const { return impl_->dropped_count; }
double MppEncoder::last_encode_ms() const { return impl_->last_ms; }
}  // namespace rkvs
