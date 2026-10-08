#include "auto_roi_coordinator.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace dual {
namespace {
const Detection* best_person(const DetectionBatch& batch, int class_id,
                             float threshold) {
  const Detection* best = nullptr;
  for (const auto& d : batch.detections)
    if (d.class_id == class_id && d.score >= threshold &&
        (!best || d.score > best->score)) best = &d;
  return best;
}
Point2f center(const Detection& d) {
  return {(d.left + d.right) * 0.5f, (d.top + d.bottom) * 0.5f};
}
}  // namespace

AutoRoiCoordinator::AutoRoiCoordinator(std::string camera_id, int class_id,
                                       int lost_return_ms)
    : camera_id_(std::move(camera_id)), class_id_(class_id),
      lost_return_ns_(static_cast<uint64_t>(lost_return_ms) * 1000000ULL) {}

const char* AutoRoiCoordinator::mode_name(AutoRoiMode mode) {
  switch (mode) {
    case AutoRoiMode::FULL_SEARCH: return "FULL_SEARCH";
    case AutoRoiMode::SWITCHING_TO_ROI: return "SWITCHING_TO_ROI";
    case AutoRoiMode::ROI_TRACK: return "ROI_TRACK";
    case AutoRoiMode::TARGET_LOST: return "TARGET_LOST";
    case AutoRoiMode::SWITCHING_TO_FULL: return "SWITCHING_TO_FULL";
    case AutoRoiMode::ERROR: return "ERROR";
  }
  return "UNKNOWN";
}

void AutoRoiCoordinator::schedule_locked(AutoRoiAction action) {
  if (!action_) action_ = action;
}

void AutoRoiCoordinator::process(DetectionBatch* batch) {
  if (!batch || batch->camera_id != camera_id_) return;
  std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t now = batch->capture_ts_ns ? batch->capture_ts_ns : monotonic_ns();
  if (mode_ == AutoRoiMode::ROI_TRACK || mode_ == AutoRoiMode::TARGET_LOST) {
    const Detection* detection = best_person(*batch, class_id_, 0.30f);
    if (detection) {
      Point2f local = center(*detection);
      Point2f global{roi_.left + local.x, roi_.top + local.y};
      target_score_ = detection->score;
      target_visible_ = true;
      last_seen_ns_ = now;
      mode_ = AutoRoiMode::ROI_TRACK;
      if (filtered_center_.x == 0 && filtered_center_.y == 0)
        filtered_center_ = global;
      filtered_center_.x = 0.65f * filtered_center_.x + 0.35f * global.x;
      filtered_center_.y = 0.65f * filtered_center_.y + 0.35f * global.y;
      const float dx = local.x - 320.0f, dy = local.y - 320.0f;
      if (auto_enabled_ && (std::abs(dx) > 64 || std::abs(dy) > 64) &&
          now - last_move_ns_ >= 100000000ULL && !action_) {
        schedule_locked({AutoRoiActionKind::MOVE_ROI, filtered_center_});
        last_move_ns_ = now;
      }
    } else {
      target_visible_ = false;
      const uint64_t elapsed = now > last_seen_ns_ ? now - last_seen_ns_ : 0;
      if (elapsed >= 100000000ULL) mode_ = AutoRoiMode::TARGET_LOST;
      if (auto_enabled_ && elapsed >= lost_return_ns_ && !action_) {
        mode_ = AutoRoiMode::SWITCHING_TO_FULL;
        switch_started_ns_ = monotonic_ns();
        schedule_locked({AutoRoiActionKind::ENTER_FULL, {}});
      }
    }
    for (auto& d : batch->detections) {
      d.left += roi_.left; d.right += roi_.left;
      d.top += roi_.top; d.bottom += roi_.top;
    }
    batch->width = 3840; batch->height = 2160;
    return;
  }

  const Detection* detection = best_person(*batch, class_id_, 0.40f);
  target_visible_ = detection != nullptr;
  target_score_ = detection ? detection->score : 0;
  if (auto_enabled_ && mode_ == AutoRoiMode::FULL_SEARCH && detection && !action_) {
    Point2f local = center(*detection);
    Point2f global{local.x * 3840.0f / batch->width,
                   local.y * 2160.0f / batch->height};
    filtered_center_ = global;
    last_seen_ns_ = now;
    mode_ = AutoRoiMode::SWITCHING_TO_ROI;
    switch_started_ns_ = monotonic_ns();
    schedule_locked({AutoRoiActionKind::ENTER_ROI, global});
  }
}

std::optional<AutoRoiAction> AutoRoiCoordinator::take_action() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto result = action_; action_.reset(); return result;
}

void AutoRoiCoordinator::action_complete(AutoRoiActionKind kind, bool ok,
                                         RoiRect roi,
                                         const std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  last_error_ = error;
  if (kind != AutoRoiActionKind::MOVE_ROI && switch_started_ns_) {
    last_switch_ms_ = (monotonic_ns() - switch_started_ns_) / 1e6;
    max_switch_ms_ = std::max(max_switch_ms_, last_switch_ms_);
    ++switch_count_;
  }
  if (!ok) {
    ++switch_failures_;
    mode_ = kind == AutoRoiActionKind::ENTER_FULL ? AutoRoiMode::ERROR
                                                  : AutoRoiMode::SWITCHING_TO_FULL;
    if (kind != AutoRoiActionKind::ENTER_FULL)
      schedule_locked({AutoRoiActionKind::ENTER_FULL, {}});
    return;
  }
  roi_ = roi;
  if (kind == AutoRoiActionKind::ENTER_ROI) {
    mode_ = AutoRoiMode::ROI_TRACK;
    last_seen_ns_ = monotonic_ns();
  } else if (kind == AutoRoiActionKind::ENTER_FULL) {
    mode_ = AutoRoiMode::FULL_SEARCH;
    target_visible_ = false;
    filtered_center_ = {};
  }
}

void AutoRoiCoordinator::force_full() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto_enabled_ = false; mode_ = AutoRoiMode::SWITCHING_TO_FULL;
  switch_started_ns_ = monotonic_ns();
  schedule_locked({AutoRoiActionKind::ENTER_FULL, {}});
}
void AutoRoiCoordinator::force_roi(Point2f center) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto_enabled_ = false; mode_ = AutoRoiMode::SWITCHING_TO_ROI;
  switch_started_ns_ = monotonic_ns();
  schedule_locked({AutoRoiActionKind::ENTER_ROI, center});
}
void AutoRoiCoordinator::set_auto(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_); auto_enabled_ = enabled;
}
bool AutoRoiCoordinator::auto_enabled() const {
  std::lock_guard<std::mutex> lock(mutex_); return auto_enabled_;
}
bool AutoRoiCoordinator::roi_mode() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_ == AutoRoiMode::ROI_TRACK || mode_ == AutoRoiMode::TARGET_LOST;
}
std::string AutoRoiCoordinator::status_json() const {
  std::lock_guard<std::mutex> lock(mutex_); std::ostringstream out;
  out << "{\"camera\":\"" << camera_id_ << "\",\"mode\":\""
      << mode_name(mode_) << "\",\"auto\":" << (auto_enabled_ ? "true" : "false")
      << ",\"roi\":[" << roi_.left << ',' << roi_.top << ",640,640]"
      << ",\"target_visible\":" << (target_visible_ ? "true" : "false")
      << ",\"target_score\":" << target_score_
      << ",\"switch_count\":" << switch_count_
      << ",\"switch_failures\":" << switch_failures_
      << ",\"last_switch_ms\":" << last_switch_ms_
      << ",\"max_switch_ms\":" << max_switch_ms_
      << ",\"last_error\":\"" << json_escape(last_error_) << "\"}";
  return out.str();
}

}  // namespace dual
