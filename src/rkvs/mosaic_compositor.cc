#include "rkvs/mosaic_compositor.hpp"

#include "im2d.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace rkvs {
namespace {
struct DmaRequest { uint64_t len; uint32_t fd; uint32_t flags; uint64_t heap_flags; };
#define RKVS_COMPOSE_DMA_ALLOC _IOWR('H', 0x0, DmaRequest)
int allocate_dma(size_t size) {
  const char* heaps[] = {"/dev/dma_heap/cma", "/dev/dma_heap/system-uncached",
                         "/dev/dma_heap/system"};
  for (const char* name : heaps) {
    int heap = open(name, O_RDWR | O_CLOEXEC);
    if (heap < 0) continue;
    DmaRequest request{};
    request.len = size;
    request.flags = O_RDWR | O_CLOEXEC;
    const int result = ioctl(heap, RKVS_COMPOSE_DMA_ALLOC, &request);
    close(heap);
    if (!result) return request.fd;
  }
  return -1;
}

bool fill_rect(rga_buffer_t canvas, int canvas_width, int canvas_height,
               int x, int y, int width, int height, int color) {
  if (width <= 0 || height <= 0) return true;
  x = std::clamp(x, 0, canvas_width - 1);
  y = std::clamp(y, 0, canvas_height - 1);
  width = std::min(width, canvas_width - x);
  height = std::min(height, canvas_height - y);
  if (width <= 0 || height <= 0) return true;
  x &= ~1;
  y &= ~1;
  width &= ~1;
  height &= ~1;
  if (width <= 0 || height <= 0) return true;
  return imfill(canvas, {x, y, width, height}, color) == IM_STATUS_SUCCESS;
}

bool draw_outline(rga_buffer_t canvas, int canvas_width, int canvas_height,
                  const im_rect& rect, int color, int thickness) {
  const int x = rect.x & ~1;
  const int y = rect.y & ~1;
  const int w = rect.width & ~1;
  const int h = rect.height & ~1;
  const int t = std::max(2, thickness & ~1);
  bool ok = true;
  ok &= fill_rect(canvas, canvas_width, canvas_height, x, y, w, t, color);
  ok &= fill_rect(canvas, canvas_width, canvas_height, x, y + h - t, w, t, color);
  ok &= fill_rect(canvas, canvas_width, canvas_height, x, y, t, h, color);
  ok &= fill_rect(canvas, canvas_width, canvas_height, x + w - t, y, t, h, color);
  return ok;
}

int track_color(int64_t track_id) {
  static constexpr int colors[] = {
      0x00ff00, 0xff0000, 0x0000ff, 0xffff00,
      0xff00ff, 0x00ffff, 0xff8000, 0x80ff00};
  if (track_id < 0) return 0x00ff00;
  return colors[static_cast<size_t>(track_id) %
                (sizeof(colors) / sizeof(colors[0]))];
}

const uint8_t* glyph(char value) {
  static constexpr uint8_t blank[7] = {0, 0, 0, 0, 0, 0, 0};
  static constexpr uint8_t font[][7] = {
      {0x0e,0x11,0x13,0x15,0x19,0x11,0x0e}, // 0
      {0x04,0x0c,0x04,0x04,0x04,0x04,0x0e}, // 1
      {0x0e,0x11,0x01,0x02,0x04,0x08,0x1f}, // 2
      {0x1f,0x02,0x04,0x02,0x01,0x11,0x0e}, // 3
      {0x02,0x06,0x0a,0x12,0x1f,0x02,0x02}, // 4
      {0x1f,0x10,0x1e,0x01,0x01,0x11,0x0e}, // 5
      {0x06,0x08,0x10,0x1e,0x11,0x11,0x0e}, // 6
      {0x1f,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
      {0x0e,0x11,0x11,0x0e,0x11,0x11,0x0e}, // 8
      {0x0e,0x11,0x11,0x0f,0x01,0x02,0x0c}, // 9
      {0x0e,0x11,0x11,0x1f,0x11,0x11,0x11}, // A
      {0x1e,0x11,0x11,0x1e,0x11,0x11,0x1e}, // B
      {0x0e,0x11,0x10,0x10,0x10,0x11,0x0e}, // C
      {0x1e,0x11,0x11,0x11,0x11,0x11,0x1e}, // D
      {0x1f,0x10,0x10,0x1e,0x10,0x10,0x1f}, // E
      {0x1f,0x10,0x10,0x1e,0x10,0x10,0x10}, // F
      {0x0e,0x11,0x10,0x17,0x11,0x11,0x0e}, // G
      {0x11,0x11,0x11,0x1f,0x11,0x11,0x11}, // H
      {0x0e,0x04,0x04,0x04,0x04,0x04,0x0e}, // I
      {0x07,0x02,0x02,0x02,0x12,0x12,0x0c}, // J
      {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, // K
      {0x10,0x10,0x10,0x10,0x10,0x10,0x1f}, // L
      {0x11,0x1b,0x15,0x15,0x11,0x11,0x11}, // M
      {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, // N
      {0x0e,0x11,0x11,0x11,0x11,0x11,0x0e}, // O
      {0x1e,0x11,0x11,0x1e,0x10,0x10,0x10}, // P
      {0x0e,0x11,0x11,0x11,0x15,0x12,0x0d}, // Q
      {0x1e,0x11,0x11,0x1e,0x14,0x12,0x11}, // R
      {0x0f,0x10,0x10,0x0e,0x01,0x01,0x1e}, // S
      {0x1f,0x04,0x04,0x04,0x04,0x04,0x04}, // T
      {0x11,0x11,0x11,0x11,0x11,0x11,0x0e}, // U
      {0x11,0x11,0x11,0x11,0x11,0x0a,0x04}, // V
      {0x11,0x11,0x11,0x15,0x15,0x15,0x0a}, // W
      {0x11,0x11,0x0a,0x04,0x0a,0x11,0x11}, // X
      {0x11,0x11,0x0a,0x04,0x04,0x04,0x04}, // Y
      {0x1f,0x01,0x02,0x04,0x08,0x10,0x1f}, // Z
      {0x00,0x00,0x00,0x00,0x00,0x0c,0x0c}, // .
      {0x00,0x04,0x04,0x00,0x04,0x04,0x00}, // :
      {0x18,0x19,0x02,0x04,0x08,0x13,0x03}, // %
      {0x00,0x00,0x00,0x1f,0x00,0x00,0x00}, // -
      {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // space
  };
  if (value >= '0' && value <= '9') return font[value - '0'];
  value = static_cast<char>(std::toupper(static_cast<unsigned char>(value)));
  if (value >= 'A' && value <= 'Z') return font[10 + value - 'A'];
  if (value == '.') return font[36];
  if (value == ':') return font[37];
  if (value == '%') return font[38];
  if (value == '-') return font[39];
  if (value == ' ') return font[40];
  return blank;
}

void draw_text_y(uint8_t* nv12, int canvas_width, int canvas_height, int x, int y,
                 const std::string& text, uint8_t luma = 235) {
  if (!nv12) return;
  constexpr int scale = 2;
  constexpr int glyph_width = 5;
  constexpr int glyph_height = 7;
  int cursor = x;
  for (char ch : text) {
    const uint8_t* g = glyph(ch);
    for (int row = 0; row < glyph_height; ++row) {
      for (int col = 0; col < glyph_width; ++col) {
        if (!(g[row] & (1 << (glyph_width - 1 - col)))) continue;
        for (int yy = 0; yy < scale; ++yy) {
          const int py = y + row * scale + yy;
          if (py < 0 || py >= canvas_height) continue;
          for (int xx = 0; xx < scale; ++xx) {
            const int px = cursor + col * scale + xx;
            if (px < 0 || px >= canvas_width) continue;
            nv12[static_cast<size_t>(py) * canvas_width + px] = luma;
          }
        }
      }
    }
    cursor += (glyph_width + 1) * scale;
    if (cursor >= canvas_width) break;
  }
}

std::string detection_text(const Detection& detection) {
  std::ostringstream output;
  output << detection.label << ' ' << std::fixed << std::setprecision(2)
         << detection.score;
  if (detection.track_id >= 0) output << " ID:" << detection.track_id;
  return output.str();
}
}  // namespace

class RgaMosaicCompositor::Impl {
 public:
  struct Pool {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<int> fds;
    std::vector<uint8_t*> maps;
    size_t map_size = 0;
    std::deque<int> free;
    bool active = true;
    ~Pool() {
      for (auto* map : maps) if (map && map != MAP_FAILED) munmap(map, map_size);
      for (int fd : fds) if (fd >= 0) close(fd);
    }
  };

  Impl(DisplayConfig value, std::vector<std::string> ids, LatestFrameHub* value_hub)
      : config(std::move(value)), source_ids(std::move(ids)), hub(value_hub),
        consumer(hub->subscribe("mosaic")), pool(std::make_shared<Pool>()) {
    if (!hub || source_ids.empty() || source_ids.size() > 9)
      throw std::invalid_argument("mosaic requires hub and 1..9 sources");
    if ((config.width & 1) || (config.height & 1))
      throw std::invalid_argument("NV12 mosaic dimensions must be even");
    const size_t size = static_cast<size_t>(config.width) * config.height * 3 / 2;
    pool->map_size = size;
    for (int index = 0; index < 3; ++index) {
      int fd = allocate_dma(size);
      if (fd < 0) throw std::runtime_error("mosaic DMA allocation failed");
      auto* map = static_cast<uint8_t*>(
          mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
      if (map == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mosaic DMA mmap failed");
      }
      pool->fds.push_back(fd);
      pool->maps.push_back(map);
      pool->free.push_back(index);
    }
  }
  ~Impl() { stop(); }

  int acquire() {
    std::lock_guard<std::mutex> lock(pool->mutex);
    if (pool->free.empty()) return -1;
    int result = pool->free.front();
    pool->free.pop_front();
    return result;
  }

  void loop() {
    const uint64_t period = 1000000000ULL / std::max(1, config.fps);
    uint64_t next = monotonic_ns();
    bool dirty = false;
    while (running) {
      auto ready = consumer.take_ready();
      bool accepted = false;
      for (auto& frame : ready) {
        if (std::find(source_ids.begin(), source_ids.end(), frame->source_id) ==
            source_ids.end())
          continue;
        latest[frame->source_id] = frame;
        accepted = true;
      }
      if (accepted) {
        dirty = true;
        last_input_ns = monotonic_ns();
      }
      if (!dirty) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        next = monotonic_ns();
        continue;
      }
      const uint64_t now = monotonic_ns();
      if (now < next)
        std::this_thread::sleep_for(std::chrono::nanoseconds(next - now));
      next = monotonic_ns() + period;
      dirty = false;
      int slot = acquire();
      if (slot < 0) { ++dropped_count; continue; }
      const uint64_t begin = monotonic_ns();
      rga_buffer_t canvas = wrapbuffer_fd(pool->fds[slot], config.width,
          config.height, RK_FORMAT_YCbCr_420_SP, config.width, config.height);
      uint8_t* canvas_map = pool->maps[slot];
      if (imfill(canvas, {0, 0, config.width, config.height}, 0x000000) !=
          IM_STATUS_SUCCESS) {
        release(slot);
        ++dropped_count;
        continue;
      }
      const int columns = source_ids.size() == 1 ? 1 :
                          source_ids.size() == 2 ? 2 : 3;
      const int rows = static_cast<int>((source_ids.size() + columns - 1) / columns);
      const int cell_width = (config.width / columns) & ~1;
      const int cell_height = (config.height / rows) & ~1;
      std::map<std::string, ResultBatch> result_copy;
      {
        std::lock_guard<std::mutex> lock(result_mutex);
        result_copy = results;
      }
      for (size_t index = 0; index < source_ids.size(); ++index) {
        auto found = latest.find(source_ids[index]);
        if (found == latest.end() || !found->second || found->second->planes.empty()) continue;
        const auto& frame = *found->second;
        const int cell_x = static_cast<int>(index % columns) * cell_width;
        const int cell_y = static_cast<int>(index / columns) * cell_height;
        int crop_left = 0, crop_top = 0;
        int crop_width = frame.width, crop_height = frame.height;
        {
          std::lock_guard<std::mutex> lock(result_mutex);
          if (roi_mode == "auto" && roi_valid &&
              monotonic_ns() - roi_last_seen_ns <= roi_lost_ns) {
            crop_left = std::clamp(static_cast<int>(roi_left) & ~1, 0,
                                   std::max(0, frame.width - 2));
            crop_top = std::clamp(static_cast<int>(roi_top) & ~1, 0,
                                  std::max(0, frame.height - 2));
            crop_width = std::clamp(static_cast<int>(roi_width) & ~1, 2,
                                    frame.width - crop_left);
            crop_height = std::clamp(static_cast<int>(roi_height) & ~1, 2,
                                     frame.height - crop_top);
          }
        }
        const float scale = std::min(static_cast<float>(cell_width) / crop_width,
                                     static_cast<float>(cell_height) / crop_height);
        const int shown_width = std::max(2, static_cast<int>(crop_width * scale) & ~1);
        const int shown_height = std::max(2, static_cast<int>(crop_height * scale) & ~1);
        const int left = (cell_x + (cell_width - shown_width) / 2) & ~1;
        const int top = (cell_y + (cell_height - shown_height) / 2) & ~1;
        rga_buffer_t source = wrapbuffer_fd(frame.planes[0].dmabuf_fd,
            frame.width, frame.height, RK_FORMAT_YCbCr_420_SP,
            frame.planes[0].stride ? frame.planes[0].stride : frame.width,
            frame.height);
        rga_buffer_t pattern{};
        im_rect source_rect{crop_left, crop_top, crop_width, crop_height};
        im_rect destination_rect{left, top, shown_width, shown_height};
        im_rect pattern_rect{};
        if (improcess(source, canvas, pattern, source_rect, destination_rect,
                      pattern_rect, IM_SYNC) != IM_STATUS_SUCCESS) continue;
        auto result = result_copy.find(frame.source_id);
        if (result == result_copy.end()) continue;
        std::vector<im_rect> rectangles;
        std::vector<std::pair<im_rect, int>> tracked_rectangles;
        for (const auto& detection : result->second.detections) {
          int box_left = left + static_cast<int>((detection.box.left - crop_left) * scale);
          int box_top = top + static_cast<int>((detection.box.top - crop_top) * scale);
          int box_right = left + static_cast<int>((detection.box.right - crop_left) * scale);
          int box_bottom = top + static_cast<int>((detection.box.bottom - crop_top) * scale);
          box_left = std::clamp(box_left, left, left + shown_width - 2);
          box_top = std::clamp(box_top, top, top + shown_height - 2);
          box_right = std::clamp(box_right, box_left + 2, left + shown_width);
          box_bottom = std::clamp(box_bottom, box_top + 2, top + shown_height);
          im_rect rect{box_left, box_top, box_right - box_left,
                       box_bottom - box_top};
          rectangles.push_back(rect);
          tracked_rectangles.push_back({rect, track_color(detection.track_id)});
        }
        for (const auto& item : tracked_rectangles) {
          const auto& rect = item.first;
          const int color = item.second;
          if (!draw_outline(canvas, config.width, config.height, rect,
                            color, 4) && !reported_overlay_error.exchange(true))
            std::cerr << "mosaic: failed to draw detection overlay\n";
          const int badge_width = 36;
          const int badge_height = 18;
          if (!fill_rect(canvas, config.width, config.height, rect.x,
                         std::max(0, rect.y - badge_height), badge_width,
                         badge_height, color) &&
              !reported_overlay_error.exchange(true))
            std::cerr << "mosaic: failed to draw track badge overlay\n";
          const std::string text = detection_text(result->second.detections[
              static_cast<size_t>(&item - tracked_rectangles.data())]);
          const int text_y = std::max(top, rect.y - 36);
          draw_text_y(canvas_map, config.width, config.height, rect.x + 4,
                      text_y + 4, text);
        }
        std::vector<im_rect> motion_rectangles;
        for (const auto& region : result->second.motion.regions) {
          int box_left = left + static_cast<int>(region.box.left * scale);
          int box_top = top + static_cast<int>(region.box.top * scale);
          int box_right = left + static_cast<int>(region.box.right * scale);
          int box_bottom = top + static_cast<int>(region.box.bottom * scale);
          box_left = std::clamp(box_left, left, left + shown_width - 2);
          box_top = std::clamp(box_top, top, top + shown_height - 2);
          box_right = std::clamp(box_right, box_left + 2, left + shown_width);
          box_bottom = std::clamp(box_bottom, box_top + 2, top + shown_height);
          motion_rectangles.push_back({box_left, box_top, box_right - box_left,
                                       box_bottom - box_top});
        }
        for (const auto& rect : motion_rectangles) {
          if (!draw_outline(canvas, config.width, config.height, rect,
                            0xffff00, 2) && !reported_overlay_error.exchange(true))
            std::cerr << "mosaic: failed to draw motion overlay\n";
        }
      }
      last_rga = static_cast<double>(monotonic_ns() - begin) / 1e6;
      auto output = std::make_shared<FrameRef>();
      output->source_id = "mosaic";
      output->sequence = ++sequence;
      output->capture_ts_ns = begin;
      output->receive_ts_ns = monotonic_ns();
      output->format = PixelFormat::kNv12;
      output->width = config.width;
      output->height = config.height;
      output->planes.push_back({pool->fds[slot], 0,
          static_cast<uint32_t>(config.width),
          static_cast<uint32_t>(config.width * config.height * 3 / 2)});
      auto state = pool;
      output->lease = std::make_shared<BufferLease>([state, slot] {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->active) state->free.push_back(slot);
        state->cv.notify_one();
      });
      if (callback) callback(std::move(output));
      ++frame_count;
    }
  }

  void release(int slot) {
    std::lock_guard<std::mutex> lock(pool->mutex);
    pool->free.push_back(slot);
  }
  void stop() {
    running = false;
    if (thread.joinable()) thread.join();
    callback = {};
  }

  DisplayConfig config;
  std::vector<std::string> source_ids;
  LatestFrameHub* hub;
  LatestFrameHub::Consumer consumer;
  std::shared_ptr<Pool> pool;
  std::map<std::string, SharedFrame> latest;
  std::mutex result_mutex;
  std::map<std::string, ResultBatch> results;
  std::string roi_mode = "full";
  bool roi_valid = false;
  float roi_left = 0, roi_top = 0, roi_width = 0, roi_height = 0;
  uint64_t roi_last_seen_ns = 0;
  const uint64_t roi_lost_ns = 1500000000ULL;
  OutputCallback callback;
  std::atomic<bool> running{false};
  std::thread thread;
  std::atomic<uint64_t> frame_count{0}, dropped_count{0};
  std::atomic<double> last_rga{0};
  std::atomic<uint64_t> last_input_ns{0};
  std::atomic<bool> reported_overlay_error{false};
  uint64_t sequence = 0;
};

RgaMosaicCompositor::RgaMosaicCompositor(DisplayConfig config,
    std::vector<std::string> source_ids, LatestFrameHub* hub)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(source_ids), hub)) {}
RgaMosaicCompositor::~RgaMosaicCompositor() = default;
void RgaMosaicCompositor::start(OutputCallback callback) {
  if (impl_->running.exchange(true)) return;
  impl_->callback = std::move(callback);
  impl_->thread = std::thread(&Impl::loop, impl_.get());
}
void RgaMosaicCompositor::stop() { impl_->stop(); }
void RgaMosaicCompositor::update_result(const ResultBatch& result) {
  std::lock_guard<std::mutex> lock(impl_->result_mutex);
  impl_->results[result.source_id] = result;
  if (impl_->roi_mode != "auto" || impl_->source_ids.size() != 1 ||
      result.source_id != impl_->source_ids.front())
    return;
  const Detection* target = nullptr;
  for (const auto& detection : result.detections)
    if (detection.class_id == 0 && detection.score >= 0.30f &&
        (!target || detection.score > target->score))
      target = &detection;
  if (!target || result.width <= 0 || result.height <= 0) return;
  const float aspect = static_cast<float>(impl_->config.width) / impl_->config.height;
  float height = std::max(720.0f, target->box.height() * 1.8f);
  float width = height * aspect;
  if (width < target->box.width() * 1.6f) {
    width = target->box.width() * 1.6f;
    height = width / aspect;
  }
  if (width > result.width) { width = result.width; height = width / aspect; }
  if (height > result.height) { height = result.height; width = height * aspect; }
  const float center_x = (target->box.left + target->box.right) * 0.5f;
  const float center_y = (target->box.top + target->box.bottom) * 0.5f;
  const float left = std::clamp(center_x - width * 0.5f, 0.0f,
                                std::max(0.0f, result.width - width));
  const float top = std::clamp(center_y - height * 0.5f, 0.0f,
                               std::max(0.0f, result.height - height));
  const float alpha = impl_->roi_valid ? 0.25f : 1.0f;
  impl_->roi_left += alpha * (left - impl_->roi_left);
  impl_->roi_top += alpha * (top - impl_->roi_top);
  impl_->roi_width += alpha * (width - impl_->roi_width);
  impl_->roi_height += alpha * (height - impl_->roi_height);
  impl_->roi_valid = true;
  impl_->roi_last_seen_ns = monotonic_ns();
}
uint64_t RgaMosaicCompositor::frames() const { return impl_->frame_count; }
uint64_t RgaMosaicCompositor::dropped() const { return impl_->dropped_count; }
double RgaMosaicCompositor::last_rga_ms() const { return impl_->last_rga; }
bool RgaMosaicCompositor::idle() const {
  const uint64_t last = impl_->last_input_ns.load();
  return !last || monotonic_ns() - last > 250000000ULL;
}
bool RgaMosaicCompositor::set_roi_mode(const std::string& mode) {
  if (mode != "full" && mode != "auto") return false;
  std::lock_guard<std::mutex> lock(impl_->result_mutex);
  impl_->roi_mode = mode;
  if (mode == "full") impl_->roi_valid = false;
  return true;
}
std::string RgaMosaicCompositor::roi_status_json() const {
  std::lock_guard<std::mutex> lock(impl_->result_mutex);
  const bool tracking = impl_->roi_mode == "auto" && impl_->roi_valid &&
      monotonic_ns() - impl_->roi_last_seen_ns <= impl_->roi_lost_ns;
  std::ostringstream out;
  out << "{\"source_id\":\"" << impl_->source_ids.front()
      << "\",\"mode\":\"" << impl_->roi_mode
      << "\",\"tracking\":" << (tracking ? "true" : "false")
      << ",\"rect\":[" << static_cast<int>(impl_->roi_left) << ','
      << static_cast<int>(impl_->roi_top) << ','
      << static_cast<int>(impl_->roi_width) << ','
      << static_cast<int>(impl_->roi_height) << "]}";
  return out.str();
}

}  // namespace rkvs
