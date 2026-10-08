#include "cross_camera_matcher.hpp"
#include <algorithm>
#include <cmath>

namespace dual {
namespace {
Point2f center(const Detection& d){return {(d.left+d.right)*.5,(d.top+d.bottom)*.5};}
double area(const Detection& d){return std::max(1.f,d.right-d.left)*std::max(1.f,d.bottom-d.top);}
std::array<double,64> histogram(const DetectionBatch& b,const Detection& d){
  std::array<double,64> h{};if(b.preview_rgb.size()!=640*360*3||b.width<=0||b.height<=0)return h;
  int l=std::clamp(static_cast<int>(d.left*640/b.width),0,639);
  int r=std::clamp(static_cast<int>(d.right*640/b.width),l+1,640);
  int t=std::clamp(static_cast<int>(d.top*360/b.height),0,359);
  int bot=std::clamp(static_cast<int>(d.bottom*360/b.height),t+1,360);
  double count=0;for(int y=t;y<bot;y+=2)for(int x=l;x<r;x+=2){
    const auto* p=b.preview_rgb.data()+(y*640+x)*3;
    int bin=(p[0]>>6)*16+(p[1]>>6)*4+(p[2]>>6);h[bin]++;count++;
  }
  if(count)for(auto& v:h)v/=count;return h;
}
double appearance_similarity(const DetectionBatch& a,const Detection& da,
                             const DetectionBatch& b,const Detection& db){
  auto ha=histogram(a,da),hb=histogram(b,db);double suma=0,sumb=0,intersection=0;
  for(size_t i=0;i<ha.size();++i){suma+=ha[i];sumb+=hb[i];intersection+=std::min(ha[i],hb[i]);}
  return suma>.5&&sumb>.5?intersection:-1;
}
}

std::optional<CrossCameraMatch> CrossCameraMatcher::match(
    const DetectionBatch& a,const DetectionBatch& b,int class_id,
    bool allow_projected_fallback) const {
  if(!calibration_||!calibration_->valid())return std::nullopt;
  double dt=std::abs(static_cast<double>(static_cast<int64_t>(a.capture_ts_ns)-
                                        static_cast<int64_t>(b.capture_ts_ns)))/1e6;
  const double time_limit=calibration_->homography_only()?40.0:max_frame_delta_ms_;
  if(dt>time_limit)return std::nullopt;
  const Detection* best0=nullptr;
  for(const auto& d:a.detections)
    if(d.class_id==class_id&&d.score>=.4f&&(!best0||d.score>best0->score))best0=&d;
  if(!best0)return std::nullopt;
  if(calibration_->homography_only()) {
    Point2f predicted=calibration_->map_homography(center(*best0));
    const Detection* selected=nullptr;double selected_cost=1e30,selected_error=0;
    for(const auto& d:b.detections){
      if(d.class_id!=class_id||d.score<.35f)continue;
      Point2f c=center(d);double error=std::hypot(c.x-predicted.x,c.y-predicted.y);
      if(error>600)continue;double ratio=area(d)/area(*best0);
      if(ratio<.15||ratio>6.0)continue;
      double appearance=appearance_similarity(a,*best0,b,d);
      if(appearance>=0&&appearance<.15)continue;
      double cost=error/600+.20*std::abs(std::log(ratio))-.15*d.score;
      if(appearance>=0)cost+=.50*(1-appearance);
      if(cost<selected_cost){selected=&d;selected_cost=cost;selected_error=error;}
    }
    if(!selected){
      if(!allow_projected_fallback)return std::nullopt;
      CrossCameraMatch out;out.global_detection=*best0;
      out.roi_camera_detection=*best0;out.roi_center=predicted;
      out.depth_m=-1;out.epipolar_error_px=-1;out.frame_delta_ms=dt;
      out.confidence=.10;out.projected_only=true;return out;
    }
    CrossCameraMatch out;out.global_detection=*best0;out.roi_camera_detection=*selected;
    out.roi_center=center(*selected);out.depth_m=-1;out.epipolar_error_px=selected_error;
    out.frame_delta_ms=dt;out.confidence=std::clamp(1-selected_cost/2,0.0,1.0);
    return out;
  }
  const Detection* best1=nullptr;double best_cost=1e30,best_epi=0,best_depth=0;
  for(const auto& d:b.detections){
    if(d.class_id!=class_id||d.score<.35f)continue;
    double epi=calibration_->epipolar_distance(center(*best0),center(d));
    if(epi>max_epipolar_error_px_)continue;
    double ratio=area(d)/area(*best0);
    if(ratio<.20||ratio>5.0)continue;
    double depth=calibration_->triangulate_depth(center(*best0),center(d));
    if(depth<calibration_->min_depth_m()*.75||depth>calibration_->max_depth_m()*1.25)
      continue;
    double appearance=appearance_similarity(a,*best0,b,d);
    if(appearance>=0&&appearance<.20)continue;
    double cost=epi/max_epipolar_error_px_+.25*std::abs(std::log(ratio))+
                .15*dt/max_frame_delta_ms_-.15*d.score;
    if(appearance>=0)cost+=.70*(1.0-appearance);
    if(cost<best_cost){best_cost=cost;best1=&d;best_epi=epi;best_depth=depth;}
  }
  if(!best1)return std::nullopt;
  CrossCameraMatch out;
  out.global_detection=*best0;out.roi_camera_detection=*best1;
  out.roi_center=center(*best1);out.depth_m=best_depth;
  out.epipolar_error_px=best_epi;out.frame_delta_ms=dt;
  out.confidence=std::clamp(1.0-best_cost/2.0,0.0,1.0);
  return out;
}
} // namespace dual
