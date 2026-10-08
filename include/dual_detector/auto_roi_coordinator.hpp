#pragma once

#include "roi_controller.hpp"
#include "types.hpp"
#include <mutex>
#include <optional>
#include <string>

namespace dual {

enum class AutoRoiMode { FULL_SEARCH, SWITCHING_TO_ROI, ROI_TRACK,
                         TARGET_LOST, SWITCHING_TO_FULL, ERROR };
enum class AutoRoiActionKind { ENTER_ROI, MOVE_ROI, ENTER_FULL };
struct AutoRoiAction { AutoRoiActionKind kind; Point2f center; };

class AutoRoiCoordinator {
 public:
  AutoRoiCoordinator(std::string camera_id, int class_id = 0,
                     int lost_return_ms = 500);
  void process(DetectionBatch* batch);
  std::optional<AutoRoiAction> take_action();
  void action_complete(AutoRoiActionKind kind, bool ok, RoiRect roi,
                       const std::string& error);
  void force_full();
  void force_roi(Point2f center = {1920.0f, 1080.0f});
  void set_auto(bool enabled);
  bool auto_enabled() const;
  bool roi_mode() const;
  std::string status_json() const;

 private:
  void schedule_locked(AutoRoiAction action);
  static const char* mode_name(AutoRoiMode mode);
  std::string camera_id_;
  int class_id_ = 0;
  uint64_t lost_return_ns_ = 500000000ULL;
  mutable std::mutex mutex_;
  AutoRoiMode mode_ = AutoRoiMode::FULL_SEARCH;
  std::optional<AutoRoiAction> action_;
  RoiRect roi_{};
  Point2f filtered_center_{};
  uint64_t last_seen_ns_ = 0, last_move_ns_ = 0, switch_started_ns_ = 0;
  uint64_t switch_count_ = 0, switch_failures_ = 0;
  double last_switch_ms_ = 0, max_switch_ms_ = 0;
  float target_score_ = 0;
  bool target_visible_ = false, auto_enabled_ = true;
  std::string last_error_;
};

}  // namespace dual
