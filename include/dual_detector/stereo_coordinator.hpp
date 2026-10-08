#pragma once

#include "cross_camera_matcher.hpp"
#include "roi_controller.hpp"
#include <deque>
#include <mutex>
#include <optional>
#include <string>

namespace dual {

enum class StereoMode { GLOBAL_SEARCH,CROSS_CAMERA_MATCH,SWITCHING_TO_ROI,
                        ROI_TRACK,TARGET_LOST,GLOBAL_REACQUIRE };
enum class RoiActionKind { ENTER_ROI,MOVE_ROI,ENTER_FULL };
struct RoiAction { RoiActionKind kind; Point2f center; };

class StereoCoordinator {
 public:
  explicit StereoCoordinator(StereoCalibration calibration,int class_id=0,
                             bool allow_projected_fallback=false,
                             int lost_return_ms=300);
  void process(DetectionBatch* batch);
  std::optional<RoiAction> take_action();
  void action_complete(RoiActionKind kind,bool ok,RoiRect roi,
                       const std::string& error);
  std::string status_json() const;
  void unlock();
  void set_projected_fallback(bool enabled);
  StereoMode mode() const;
 private:
  void try_match_locked();
  void schedule_locked(RoiAction action);
  static const char* mode_name(StereoMode mode);
  mutable std::mutex mutex_;
  StereoCalibration calibration_;
  CrossCameraMatcher matcher_;
  std::deque<DetectionBatch> full_[2];
  std::optional<RoiAction> action_;
  StereoMode mode_=StereoMode::GLOBAL_SEARCH;
  int class_id_=0;
  uint64_t track_id_=0,last_seen_ns_=0,last_move_ns_=0;
  RoiRect roi_{};
  Point2f filtered_center_{};
  double depth_m_=0,mapping_error_px_=0,frame_delta_ms_=0,match_confidence_=0;
  std::string last_error_;
  bool target_visible_=false;
  bool allow_projected_fallback_=false;
  bool projected_fallback_used_=false;
  bool current_match_projected_=false;
  uint64_t lost_return_ns_=300000000ULL;
};

} // namespace dual
