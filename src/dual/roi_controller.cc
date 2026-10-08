#include "roi_controller.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/media-bus-format.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>
#include <cmath>
#include <sys/ioctl.h>
#include <unistd.h>

namespace dual {
namespace {
int even_clamp(int value,int maximum){return std::max(0,std::min(maximum,value))&~1;}
bool set_fmt(const std::string& path,unsigned pad,unsigned width,unsigned height,
             unsigned code,std::string* error){
  int fd=open(path.c_str(),O_RDWR|O_CLOEXEC);if(fd<0){if(error)*error=path+": "+strerror(errno);return false;}
  v4l2_subdev_format f{};f.which=V4L2_SUBDEV_FORMAT_ACTIVE;f.pad=pad;
  f.format.width=width;f.format.height=height;f.format.code=code;f.format.field=V4L2_FIELD_NONE;
  bool ok=ioctl(fd,VIDIOC_SUBDEV_S_FMT,&f)==0&&f.format.width==width&&f.format.height==height;
  if(!ok&&error)*error="S_FMT "+path+": "+strerror(errno);close(fd);return ok;
}
bool set_crop(const std::string& path,unsigned pad,v4l2_rect r,std::string* error){
  int fd=open(path.c_str(),O_RDWR|O_CLOEXEC);if(fd<0){if(error)*error=path+": "+strerror(errno);return false;}
  v4l2_subdev_selection s{};s.which=V4L2_SUBDEV_FORMAT_ACTIVE;s.pad=pad;s.target=V4L2_SEL_TGT_CROP;s.r=r;
  bool ok=ioctl(fd,VIDIOC_SUBDEV_S_SELECTION,&s)==0;
  if(!ok&&error)*error="S_SELECTION "+path+": "+strerror(errno);close(fd);return ok;
}
bool set_roi_fps(const std::string& path,int fps,std::string* error){
  // 112.733 fps at the documented VMAX=1222. Lower VMAX raises the frame
  // rate while the board-validated 1485 Mbps/lane link remains unchanged.
  constexpr double timing_product=112.733*1222.0;
  int vmax=std::max(686,static_cast<int>(std::lround(timing_product/fps)));
  v4l2_control control{};control.id=V4L2_CID_VBLANK;control.value=vmax-640;
  int fd=open(path.c_str(),O_RDWR|O_CLOEXEC);
  if(fd<0){if(error)*error=path+": "+strerror(errno);return false;}
  bool ok=ioctl(fd,VIDIOC_S_CTRL,&control)==0;
  if(!ok&&error)*error="set ROI VBLANK on "+path+": "+strerror(errno);
  close(fd);return ok;
}
}

RoiController::RoiController(std::string sensor,std::string cif,std::string isp,
                             int roi_fps)
    :sensor_(std::move(sensor)),cif_(std::move(cif)),isp_(std::move(isp)),
     roi_fps_(roi_fps){}
RoiRect RoiController::centered_roi(Point2f c){return {even_clamp(static_cast<int>(c.x)-320,3200),even_clamp(static_cast<int>(c.y)-320,1520),640,640};}

bool RoiController::configure(bool roi_mode,RoiRect roi,std::string* error){
  constexpr unsigned raw=MEDIA_BUS_FMT_SGBRG10_1X10,yuv=MEDIA_BUS_FMT_YUYV8_2X8;
  if(roi_mode){
    if(!set_fmt(sensor_,0,648,640,raw,error))return false;
    if(!set_crop(sensor_,0,{roi.left+12,roi.top+16,640,640},error))return false;
    if(!set_roi_fps(sensor_,roi_fps_,error))return false;
    if(!set_fmt(cif_,0,640,640,raw,error))return false;
    if(!set_fmt(isp_,0,640,640,raw,error))return false;
    if(!set_crop(isp_,0,{0,0,640,640},error))return false;
    if(!set_fmt(isp_,2,640,640,yuv,error))return false;
    if(!set_crop(isp_,2,{0,0,640,640},error))return false;
  }else{
    if(!set_fmt(sensor_,0,3864,2192,raw,error))return false;
    if(!set_fmt(cif_,0,3840,2160,raw,error))return false;
    if(!set_fmt(isp_,0,3840,2160,raw,error))return false;
    if(!set_crop(isp_,0,{0,0,3840,2160},error))return false;
    if(!set_fmt(isp_,2,3840,2160,yuv,error))return false;
    if(!set_crop(isp_,2,{0,0,3840,2160},error))return false;
  }
  roi_=roi;roi_mode_=roi_mode;return true;
}
bool RoiController::enter_roi(Point2f c,std::string* e){return configure(true,centered_roi(c),e);}
bool RoiController::apply_roi_fps(std::string* e){
  if(!roi_mode_){if(e)*e="ROI mode is not active";return false;}
  return set_roi_fps(sensor_,roi_fps_,e);
}
bool RoiController::move(Point2f c,std::string* error){
  if(!roi_mode_){if(error)*error="ROI mode is not active";return false;}
  RoiRect next=centered_roi(c);
  if(!set_crop(sensor_,0,{next.left+12,next.top+16,640,640},error))return false;
  roi_=next;return true;
}
bool RoiController::enter_full(std::string* e){return configure(false,{0,0,640,640},e);}
} // namespace dual
