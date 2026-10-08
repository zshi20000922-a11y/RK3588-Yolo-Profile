#include "preview_sink.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace dual {
namespace {
constexpr int kSourceWidth = 640;
constexpr int kSourceHeight = 360;
// Weston is in mirror mode. Its common logical width is limited by the
// portrait DSI panel to 1080 pixels, even though HDMI itself is 1920x1080.
// Keeping the surface at 1080 pixels prevents the right camera being clipped.
constexpr int kPanelWidth = 540;
constexpr int kPanelHeight = 304;
constexpr int kCanvasWidth = 1080;
constexpr int kCanvasHeight = 608;
constexpr uint64_t kPreviewPeriodNs = 16666667ULL;  // 60 Hz composition
}
PreviewSink::PreviewSink():canvas_(kCanvasWidth*kCanvasHeight*3,0),cam0_(kPanelWidth*kPanelHeight*3,0),cam1_(kPanelWidth*kPanelHeight*3,0){}
PreviewSink::~PreviewSink(){running_=false;cv_.notify_all();task_cv_.notify_all();if(child_>0)kill(child_,SIGTERM);if(processor_.joinable())processor_.join();if(writer_.joinable())writer_.join();if(child_>0)waitpid(child_,nullptr,0);unlink("/run/rknn-dual-preview-0.rgb");unlink("/run/rknn-dual-preview-1.rgb");unlink("/run/rknn-dual-preview.rgb.tmp");}
bool PreviewSink::start(){
  for(int i=0;i<2;++i){std::string path="/run/rknn-dual-preview-"+std::to_string(i)+".rgb";int initial=open(path.c_str(),O_CREAT|O_TRUNC|O_WRONLY|O_CLOEXEC,0644);if(initial<0)return false;size_t off=0;while(off<canvas_.size()){ssize_t n=write(initial,canvas_.data()+off,canvas_.size()-off);if(n<=0){close(initial);return false;}off+=n;}close(initial);}
  child_=fork();if(child_==0){execl("/bin/sh","sh","-c","gst-launch-1.0 -q multifilesrc location=/run/rknn-dual-preview-%d.rgb start-index=0 stop-index=1 loop=true ! identity sleep-time=16667 ! video/x-raw,format=RGB,width=1080,height=608,framerate=60/1 ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! waylandsink sync=false fullscreen=true",(char*)nullptr);_exit(127);}
  running_=child_>0;if(running_){writer_=std::thread(&PreviewSink::writer_loop,this);processor_=std::thread(&PreviewSink::process_loop,this);}return running_;
}
void PreviewSink::submit(DetectionBatch& b){if(!running_||b.preview_rgb.size()!=kSourceWidth*kSourceHeight*3)return;++submitted_;std::lock_guard<std::mutex> l(mutex_);auto next=std::make_unique<Task>(Task{b.camera_id,b.sequence,b.capture_ts_ns,b.detections,std::move(b.preview_rgb)});auto& pending=b.camera_id=="cam0"?cam0_task_:cam1_task_;if(!pending||next->sequence>pending->sequence)pending=std::move(next);task_cv_.notify_one();}
void PreviewSink::process_loop(){while(running_){std::unique_ptr<Task> task;{std::unique_lock<std::mutex> l(mutex_);task_cv_.wait(l,[&]{return !running_||cam0_task_||cam1_task_;});if(!running_)break;if(cam0_task_&&cam1_task_){if(prefer_cam0_)task=std::move(cam0_task_);else task=std::move(cam1_task_);prefer_cam0_=!prefer_cam0_;}else if(cam0_task_)task=std::move(cam0_task_);else task=std::move(cam1_task_);}auto& source=task->image;
  auto& rendered_sequence=task->camera=="cam0"?cam0_sequence_:cam1_sequence_;
  if(task->sequence<=rendered_sequence)continue;
  rendered_sequence=task->sequence;
  std::vector<uint8_t> image(kPanelWidth*kPanelHeight*3);
  for(int y=0;y<kPanelHeight;++y){const int sy=y*kSourceHeight/kPanelHeight;for(int x=0;x<kPanelWidth;++x){const int sx=x*kSourceWidth/kPanelWidth;memcpy(image.data()+(y*kPanelWidth+x)*3,source.data()+(sy*kSourceWidth+sx)*3,3);}}
  for(const auto& d:task->detections){int l=std::clamp((int)(d.left*kPanelWidth/3840.0f),0,kPanelWidth-1),r=std::clamp((int)(d.right*kPanelWidth/3840.0f),0,kPanelWidth-1),t=std::clamp((int)(d.top*kPanelHeight/2160.0f),0,kPanelHeight-1),bot=std::clamp((int)(d.bottom*kPanelHeight/2160.0f),0,kPanelHeight-1);auto mark=[&](int x,int y){auto*p=image.data()+(y*kPanelWidth+x)*3;p[0]=255;p[1]=32;p[2]=32;};for(int x=l;x<=r;++x){mark(x,t);mark(x,bot);}for(int y=t;y<=bot;++y){mark(l,y);mark(r,y);}}std::lock_guard<std::mutex> l(mutex_);auto& camera=task->camera=="cam0"?cam0_:cam1_;camera.swap(image);
  uint64_t now=monotonic_ns();if(now-last_write_ns_<kPreviewPeriodNs)continue;last_write_ns_=now;
  std::fill(canvas_.begin(),canvas_.end(),0);for(int y=0;y<kPanelHeight;++y){
    memcpy(canvas_.data()+(y*kCanvasWidth)*3,cam0_.data()+y*kPanelWidth*3,kPanelWidth*3);
    memcpy(canvas_.data()+(y*kCanvasWidth+kPanelWidth)*3,cam1_.data()+y*kPanelWidth*3,kPanelWidth*3);}
  pending_=canvas_;++processed_;cv_.notify_one();
}}
void PreviewSink::writer_loop(){while(running_){std::vector<uint8_t> frame;{std::unique_lock<std::mutex> l(mutex_);cv_.wait(l,[&]{return !running_||!pending_.empty();});if(!running_)break;frame.swap(pending_);}int out=open("/run/rknn-dual-preview.rgb.tmp",O_CREAT|O_TRUNC|O_WRONLY|O_CLOEXEC,0644);if(out<0)continue;size_t off=0;while(off<frame.size()){ssize_t n=write(out,frame.data()+off,frame.size()-off);if(n>0)off+=n;else if(errno==EINTR)continue;else break;}close(out);if(off==frame.size()){std::string target="/run/rknn-dual-preview-"+std::to_string(write_index_)+".rgb";if(rename("/run/rknn-dual-preview.rgb.tmp",target.c_str())==0){++written_;write_index_^=1;}}}}
}
