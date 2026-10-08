#pragma once

#include "stereo_calibration.hpp"
#include "types.hpp"
#include <optional>

namespace dual {

struct CrossCameraMatch {
  Detection global_detection;
  Detection roi_camera_detection;
  Point2f roi_center;
  double depth_m=0;
  double epipolar_error_px=0;
  double frame_delta_ms=0;
  double confidence=0;
  bool projected_only=false;
};

class CrossCameraMatcher {
 public:
  explicit CrossCameraMatcher(const StereoCalibration* calibration)
      : calibration_(calibration) {}
  std::optional<CrossCameraMatch> match(const DetectionBatch& global,
                                        const DetectionBatch& roi_camera,
                                        int class_id=0,
                                        bool allow_projected_fallback=false) const;
  void set_limits(double max_frame_delta_ms,double max_epipolar_error_px) {
    max_frame_delta_ms_=max_frame_delta_ms;
    max_epipolar_error_px_=max_epipolar_error_px;
  }
 private:
  const StereoCalibration* calibration_;
  double max_frame_delta_ms_=12.0;
  double max_epipolar_error_px_=60.0;
};

} // namespace dual
