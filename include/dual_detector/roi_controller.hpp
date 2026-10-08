#pragma once

#include "stereo_calibration.hpp"
#include <string>

namespace dual {

struct RoiRect { int left=0,top=0,width=640,height=640; };

class RoiController {
 public:
  RoiController(std::string sensor_subdev,std::string cif_subdev,
                std::string isp_subdev,int roi_fps=166);
  bool enter_roi(Point2f center,std::string* error=nullptr);
  bool apply_roi_fps(std::string* error=nullptr);
  bool move(Point2f center,std::string* error=nullptr);
  bool enter_full(std::string* error=nullptr);
  RoiRect roi() const { return roi_; }
  bool roi_mode() const { return roi_mode_; }
  static RoiRect centered_roi(Point2f center);
 private:
  bool configure(bool roi_mode,RoiRect roi,std::string* error);
  std::string sensor_,cif_,isp_;
  RoiRect roi_{};
  bool roi_mode_=false;
  int roi_fps_=166;
};

} // namespace dual
