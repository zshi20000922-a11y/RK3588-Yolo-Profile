#include "stereo_coordinator.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace dual {
namespace {
const Detection* best(const DetectionBatch& b,int cls,float threshold){
  const Detection* out=nullptr;for(const auto& d:b.detections)
    if(d.class_id==cls&&d.score>=threshold&&(!out||d.score>out->score))out=&d;
  return out;
}
Point2f centre(const Detection& d){return{(d.left+d.right)*.5,(d.top+d.bottom)*.5};}
}
StereoCoordinator::StereoCoordinator(StereoCalibration c,int cls,
                                     bool allow_projected_fallback,
                                     int lost_return_ms)
    :calibration_(std::move(c)),matcher_(&calibration_),class_id_(cls),
     allow_projected_fallback_(allow_projected_fallback),
     lost_return_ns_(static_cast<uint64_t>(lost_return_ms)*1000000ULL){}

const char* StereoCoordinator::mode_name(StereoMode m){
  switch(m){case StereoMode::GLOBAL_SEARCH:return "GLOBAL_SEARCH";
    case StereoMode::CROSS_CAMERA_MATCH:return "CROSS_CAMERA_MATCH";
    case StereoMode::SWITCHING_TO_ROI:return "SWITCHING_TO_ROI";
    case StereoMode::ROI_TRACK:return "ROI_TRACK";
    case StereoMode::TARGET_LOST:return "TARGET_LOST";
    case StereoMode::GLOBAL_REACQUIRE:return "GLOBAL_REACQUIRE";}
  return "UNKNOWN";
}
StereoMode StereoCoordinator::mode()const{std::lock_guard<std::mutex> l(mutex_);return mode_;}
void StereoCoordinator::schedule_locked(RoiAction a){if(!action_)action_=a;}

void StereoCoordinator::try_match_locked(){
  if(full_[0].empty()||full_[1].empty()||action_)return;
  const DetectionBatch& a=full_[0].back();
  const DetectionBatch* nearest=nullptr;uint64_t delta=~uint64_t{0};
  for(const auto& b:full_[1]){uint64_t d=a.capture_ts_ns>b.capture_ts_ns?
      a.capture_ts_ns-b.capture_ts_ns:b.capture_ts_ns-a.capture_ts_ns;
    if(d<delta){delta=d;nearest=&b;}}
  if(!nearest)return;
  auto match=matcher_.match(a,*nearest,class_id_,
      allow_projected_fallback_&&!projected_fallback_used_);if(!match)return;
  current_match_projected_=match->projected_only;
  if(match->projected_only)projected_fallback_used_=true;
  ++track_id_;depth_m_=match->depth_m;mapping_error_px_=match->epipolar_error_px;
  frame_delta_ms_=match->frame_delta_ms;match_confidence_=match->confidence;
  filtered_center_=match->roi_center;target_visible_=true;last_seen_ns_=a.capture_ts_ns;
  mode_=StereoMode::SWITCHING_TO_ROI;
  schedule_locked({RoiActionKind::ENTER_ROI,match->roi_center});
}

void StereoCoordinator::process(DetectionBatch* batch){
  if(!batch)return;std::lock_guard<std::mutex> lock(mutex_);
  const bool cam0=batch->camera_id=="cam0",cam1=batch->camera_id=="cam1";
  if(!cam0&&!cam1)return;
  if(cam1&&(mode_==StereoMode::ROI_TRACK||mode_==StereoMode::TARGET_LOST)){
    const Detection* d=best(*batch,class_id_,.30f);
    if(d){
      Point2f local=centre(*d),global{roi_.left+local.x,roi_.top+local.y};
      target_visible_=true;last_seen_ns_=batch->capture_ts_ns;
      if(mode_==StereoMode::TARGET_LOST)mode_=StereoMode::ROI_TRACK;
      if(filtered_center_.x==0&&filtered_center_.y==0)filtered_center_=global;
      filtered_center_.x=.65*filtered_center_.x+.35*global.x;
      filtered_center_.y=.65*filtered_center_.y+.35*global.y;
      double dx=local.x-320,dy=local.y-320;
      if((std::abs(dx)>64||std::abs(dy)>64)&&
         batch->capture_ts_ns-last_move_ns_>=100000000ULL&&!action_){
        Point2f limited{std::clamp(filtered_center_.x,roi_.left+160.0,roi_.left+480.0),
                        std::clamp(filtered_center_.y,roi_.top+160.0,roi_.top+480.0)};
        schedule_locked({RoiActionKind::MOVE_ROI,limited});last_move_ns_=batch->capture_ts_ns;
      }
    }else{
      uint64_t elapsed=batch->capture_ts_ns>last_seen_ns_?batch->capture_ts_ns-last_seen_ns_:0;
      target_visible_=false;
      if(elapsed>=100000000ULL)mode_=StereoMode::TARGET_LOST;
      if(elapsed>=lost_return_ns_&&!action_){mode_=StereoMode::GLOBAL_REACQUIRE;
        schedule_locked({RoiActionKind::ENTER_FULL,{}});}
    }
    for(auto& d:batch->detections){d.left+=roi_.left;d.right+=roi_.left;
      d.top+=roi_.top;d.bottom+=roi_.top;}
    batch->width=3840;batch->height=2160;return;
  }
  auto& q=full_[cam0?0:1];q.push_back(*batch);while(q.size()>8)q.pop_front();
  if(cam0&&best(*batch,class_id_,.40f)&&mode_==StereoMode::GLOBAL_SEARCH)
    mode_=StereoMode::CROSS_CAMERA_MATCH;
  if(mode_==StereoMode::CROSS_CAMERA_MATCH||mode_==StereoMode::GLOBAL_REACQUIRE)
    try_match_locked();
}

std::optional<RoiAction> StereoCoordinator::take_action(){std::lock_guard<std::mutex> l(mutex_);auto a=action_;action_.reset();return a;}
void StereoCoordinator::action_complete(RoiActionKind kind,bool ok,RoiRect roi,const std::string& error){
  std::lock_guard<std::mutex> l(mutex_);last_error_=error;
  if(!ok){mode_=StereoMode::GLOBAL_REACQUIRE;target_visible_=false;
    if(kind==RoiActionKind::MOVE_ROI)schedule_locked({RoiActionKind::ENTER_FULL,{}});
    return;}
  roi_=roi;
  if(kind==RoiActionKind::ENTER_ROI){mode_=StereoMode::ROI_TRACK;last_seen_ns_=monotonic_ns();}
  else if(kind==RoiActionKind::ENTER_FULL){mode_=StereoMode::GLOBAL_SEARCH;target_visible_=false;full_[1].clear();}
}
void StereoCoordinator::unlock(){std::lock_guard<std::mutex> l(mutex_);if(mode_!=StereoMode::GLOBAL_SEARCH){mode_=StereoMode::GLOBAL_REACQUIRE;schedule_locked({RoiActionKind::ENTER_FULL,{}});}}
void StereoCoordinator::set_projected_fallback(bool enabled){
  std::lock_guard<std::mutex> l(mutex_);
  allow_projected_fallback_=enabled;
  projected_fallback_used_=false;
}
std::string StereoCoordinator::status_json()const{
  std::lock_guard<std::mutex> l(mutex_);std::ostringstream o;
  o<<"{\"enabled\":true,\"mode\":\""<<mode_name(mode_)<<"\",\"global_camera\":\"cam0\",\"roi_camera\":\"cam1\",\"global_track_id\":"<<track_id_
   <<",\"roi\":["<<roi_.left<<','<<roi_.top<<",640,640],\"depth_m\":"<<depth_m_
   <<",\"mapping_error_px\":"<<mapping_error_px_<<",\"frame_delta_ms\":"<<frame_delta_ms_
   <<",\"match_confidence\":"<<match_confidence_<<",\"target_visible\":"<<(target_visible_?"true":"false")
   <<",\"projected_fallback_enabled\":"<<(allow_projected_fallback_?"true":"false")
   <<",\"projected_fallback_used\":"<<(projected_fallback_used_?"true":"false")
   <<",\"current_match_projected\":"<<(current_match_projected_?"true":"false")
   <<",\"last_error\":\""<<json_escape(last_error_)<<"\"}";return o.str();
}
} // namespace dual
