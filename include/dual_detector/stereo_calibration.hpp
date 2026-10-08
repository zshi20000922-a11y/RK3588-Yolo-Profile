#pragma once

#include <array>
#include <string>

namespace dual {

struct Point2f { double x=0, y=0; };
struct Point3f { double x=0, y=0, z=0; };

struct CameraCalibration {
  std::array<double,9> k{};
  std::array<double,5> d{}; // k1,k2,p1,p2,k3
};

class StereoCalibration {
 public:
  bool load(const std::string& path, std::string* error=nullptr);
  bool valid() const { return valid_; }
  bool homography_only() const { return homography_only_; }
  Point2f map_homography(Point2f camera0_pixel) const;
  Point2f undistort_normalized(int camera, Point2f pixel) const;
  Point2f project_to_camera1(Point2f camera0_pixel, double depth_m) const;
  double epipolar_distance(Point2f camera0_pixel, Point2f camera1_pixel) const;
  double triangulate_depth(Point2f camera0_pixel, Point2f camera1_pixel) const;
  int width() const { return width_; }
  int height() const { return height_; }
  double min_depth_m() const { return min_depth_m_; }
  double max_depth_m() const { return max_depth_m_; }
 private:
  Point2f distort_pixel(const CameraCalibration& c, Point2f normalized) const;
  CameraCalibration cameras_[2];
  std::array<double,9> r_{};
  std::array<double,3> t_{};
  std::array<double,9> f_{};
  std::array<double,9> h_{};
  int width_=0, height_=0;
  double min_depth_m_=1.0, max_depth_m_=10.0;
  bool valid_=false;
  bool homography_only_=false;
};

} // namespace dual
