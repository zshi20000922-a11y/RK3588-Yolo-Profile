#include "types.hpp"
#include "v4l2_camera.hpp"
#include "stereo_calibration.hpp"
#include "cross_camera_matcher.hpp"
#include "roi_controller.hpp"
#include "stereo_coordinator.hpp"
#include "auto_roi_coordinator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <fstream>
#include <cstdio>

using namespace dual;
int main() {
  auto lb=Letterbox::make(3840,2160);
  assert(lb.resized_width==640 && lb.resized_height==360);
  assert(lb.pad_left==0 && lb.pad_top==140);
  assert(std::abs(lb.source_x(320)-1920)<0.01);
  assert(std::abs(lb.source_y(320)-1080)<0.01);
  assert(lb.source_y(0)==0 && lb.source_y(640)==2159);

  int releases=0, index=-1;
  { FrameHandle f; auto lease=std::make_shared<FrameLease>(
      [&](uint32_t i){++releases;index=i;},3,std::move(f));
    auto copy=lease; lease.reset(); assert(releases==0); copy.reset(); }
  assert(releases==1 && index==3);
  try { FrameHandle f; FrameLease lease([&](uint32_t){++releases;},4,std::move(f)); throw 1; }
  catch(...) {} assert(releases==2);

  DetectionBatch b; b.camera_id="cam\"0"; b.width=3840;b.height=2160;
  b.detections.push_back({0,"person",.9f,1,2,3,4});
  auto json=to_json(b); assert(json.find("cam\\\"0")!=std::string::npos);
  assert(json.find("capture-to-result")==std::string::npos);

  const char* calibration_path="/tmp/rknn-stereo-test.yaml";
  { std::ofstream out(calibration_path);
    out<<"image_size: [3840, 2160]\n"
       <<"camera0_K: [2000,0,1920, 0,2000,1080, 0,0,1]\n"
       <<"camera0_D: [0,0,0,0,0]\n"
       <<"camera1_K: [2000,0,1920, 0,2000,1080, 0,0,1]\n"
       <<"camera1_D: [0,0,0,0,0]\n"
       <<"stereo_R: [1,0,0, 0,1,0, 0,0,1]\n"
       <<"stereo_T: [-0.1,0,0]\n"
       <<"stereo_F: [0,0,0, 0,0,0.00005, 0,-0.00005,0]\n"
       <<"working_distance_m: [1,10]\n"; }
  StereoCalibration calibration;std::string error;
  assert(calibration.load(calibration_path,&error));
  auto projected=calibration.project_to_camera1({1920,1080},2.0);
  assert(std::abs(projected.x-1820)<.01&&std::abs(projected.y-1080)<.01);
  assert(calibration.epipolar_distance({1920,1080},{1820,1080})<.01);
  assert(std::abs(calibration.triangulate_depth({1920,1080},{1820,1080})-2)<.01);
  DetectionBatch c0,c1;c0.camera_id="cam0";c1.camera_id="cam1";
  c0.capture_ts_ns=1000000000;c1.capture_ts_ns=1005000000;
  c0.detections.push_back({0,"person",.9f,1870,980,1970,1180});
  c1.detections.push_back({0,"person",.8f,1770,980,1870,1180});
  CrossCameraMatcher matcher(&calibration);auto match=matcher.match(c0,c1,0);
  assert(match&&std::abs(match->depth_m-2)<.01);
  auto roi=RoiController::centered_roi({1901,1001});
  assert(roi.left==1580&&roi.top==680);
  auto edge=RoiController::centered_roi({3839,2159});
  assert(edge.left==3200&&edge.top==1520);
  StereoCoordinator coordinator(calibration,0);
  coordinator.process(&c1);coordinator.process(&c0);
  auto enter=coordinator.take_action();
  assert(enter&&enter->kind==RoiActionKind::ENTER_ROI);
  auto tracked_roi=RoiController::centered_roi(enter->center);
  coordinator.action_complete(enter->kind,true,tracked_roi,"");
  assert(coordinator.mode()==StereoMode::ROI_TRACK);
  DetectionBatch local;local.camera_id="cam1";local.capture_ts_ns=2000000000;
  local.width=640;local.height=640;
  local.detections.push_back({0,"person",.9f,450,220,550,420});
  coordinator.process(&local);
  assert(local.width==3840&&local.height==2160);
  assert(local.detections[0].left==tracked_roi.left+450);
  auto move=coordinator.take_action();assert(move&&move->kind==RoiActionKind::MOVE_ROI);
  coordinator.action_complete(move->kind,true,RoiController::centered_roi(move->center),"");
  DetectionBatch lost;lost.camera_id="cam1";lost.capture_ts_ns=2400000000;
  lost.width=640;lost.height=640;coordinator.process(&lost);
  auto full=coordinator.take_action();assert(full&&full->kind==RoiActionKind::ENTER_FULL);
  AutoRoiCoordinator auto_roi("cam0",0,500);
  DetectionBatch search;search.camera_id="cam0";search.width=640;search.height=360;
  search.capture_ts_ns=1000000000;
  search.detections.push_back({0,"person",.9f,270,130,370,230});
  auto_roi.process(&search);
  auto auto_enter=auto_roi.take_action();
  assert(auto_enter&&auto_enter->kind==AutoRoiActionKind::ENTER_ROI);
  auto auto_rect=RoiController::centered_roi(auto_enter->center);
  assert(auto_rect.left==1600&&auto_rect.top==760);
  auto_roi.action_complete(auto_enter->kind,true,auto_rect,"");
  DetectionBatch auto_local;auto_local.camera_id="cam0";auto_local.width=640;
  auto_local.height=640;auto_local.capture_ts_ns=2000000000;
  auto_local.detections.push_back({0,"person",.8f,450,220,550,420});
  auto_roi.process(&auto_local);
  assert(auto_local.width==3840&&auto_local.height==2160);
  assert(auto_local.detections[0].left==auto_rect.left+450);
  auto auto_move=auto_roi.take_action();
  assert(auto_move&&auto_move->kind==AutoRoiActionKind::MOVE_ROI);
  auto_roi.action_complete(auto_move->kind,true,
      RoiController::centered_roi(auto_move->center),"");
  DetectionBatch auto_lost;auto_lost.camera_id="cam0";auto_lost.width=640;
  auto_lost.height=640;auto_lost.capture_ts_ns=2600000000;
  auto_roi.process(&auto_lost);
  auto auto_full=auto_roi.take_action();
  assert(auto_full&&auto_full->kind==AutoRoiActionKind::ENTER_FULL);
  std::remove(calibration_path);
  const char* homography_path="/tmp/rknn-homography-test.yaml";
  {std::ofstream out(homography_path);out<<"image_size: [3840,2160]\n"
    <<"homography_H: [1,0,100, 0,1,-20, 0,0,1]\n";}
  StereoCalibration homography;assert(homography.load(homography_path,&error));
  assert(homography.homography_only());auto mapped=homography.map_homography({1000,500});
  assert(mapped.x==1100&&mapped.y==480);
  DetectionBatch h0,h1;h0.camera_id="cam0";h1.camera_id="cam1";
  h0.capture_ts_ns=1000000000;h1.capture_ts_ns=1030000000;
  h0.detections.push_back({0,"person",.9f,900,400,1100,600});
  h1.detections.push_back({0,"person",.8f,1000,380,1200,580});
  CrossCameraMatcher hmatcher(&homography);auto hmatch=hmatcher.match(h0,h1,0);
  assert(hmatch&&hmatch->depth_m<0&&hmatch->frame_delta_ms==30);
  std::remove(homography_path);
  std::cout<<"dual_detector_unit_tests: PASS\n";
}
