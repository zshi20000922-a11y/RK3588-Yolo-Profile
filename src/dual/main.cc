#include "detector_service.hpp"
#include <csignal>
#include <atomic>
#include <thread>
#include <iostream>
#include <stdexcept>

namespace { volatile sig_atomic_t stop_requested=0; void signal_handler(int){stop_requested=1;}
void usage(const char* p){std::cerr<<"Usage: "<<p<<" --model FILE --labels FILE [--camera ID:DEVICE@WIDTHxHEIGHT] [--auto-roi ID:SENSOR:CIF:ISP] [--contexts 1..12] [--context-mode independent|duplicate] [--detect-fps N] [--max-inflight 1..12] [--buffers N] [--preprocess-threads 1..4] [--postprocess-threads 1..4] [--preview debug|off] [--pip-target IP] [--stereo-calibration FILE --roi-sensor-subdev DEV --roi-cif-subdev DEV --roi-isp-subdev DEV] [--stereo-project-fallback]\n";}
}
int main(int argc,char** argv){
  dual::ServiceConfig c;
  c.cameras={{"cam0","/dev/video45",640,360,3840,2160},
             {"cam1","/dev/video63",640,360,3840,2160}};
  bool camera_argument=false;
  try { for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::invalid_argument("missing value for "+a);return std::string(argv[i]);};
      if(a=="--model")c.model=value(); else if(a=="--labels")c.labels=value();
      else if(a=="--contexts")c.contexts=std::stoi(value()); else if(a=="--detect-fps")c.detect_fps=std::stoi(value());
      else if(a=="--context-mode"){
        auto mode=value();
        if(mode!="independent"&&mode!="duplicate")
          throw std::invalid_argument("--context-mode must be independent or duplicate");
        c.independent_contexts=mode=="independent";
      }
      else if(a=="--max-inflight")c.max_inflight_per_camera=std::stoi(value());
      else if(a=="--preprocess-threads")c.preprocess_threads=std::stoi(value());
      else if(a=="--postprocess-threads")c.postprocess_threads=std::stoi(value());
      else if(a=="--buffers")c.buffers=std::stoi(value()); else if(a=="--preview")c.preview=value()!="off";
      else if(a=="--pip-target")c.pip_target=value();
      else if(a=="--pip-udp-port")c.pip_udp_port=std::stoi(value());
      else if(a=="--pip-rtsp-feed-port")c.pip_rtsp_feed_port=std::stoi(value());
      else if(a=="--stereo-calibration")c.stereo_calibration=value();
      else if(a=="--roi-sensor-subdev")c.roi_sensor_subdev=value();
      else if(a=="--roi-cif-subdev")c.roi_cif_subdev=value();
      else if(a=="--roi-isp-subdev")c.roi_isp_subdev=value();
      else if(a=="--stereo-class-id")c.stereo_class_id=std::stoi(value());
      else if(a=="--stereo-project-fallback")c.stereo_project_fallback=true;
      else if(a=="--roi-fps")c.roi_capture_fps=std::stoi(value());
      else if(a=="--roi-lost-return-ms")c.roi_lost_return_ms=std::stoi(value());
      else if(a=="--auto-roi") {
        auto spec=value(); std::vector<std::string> parts; size_t begin=0;
        while(true){auto end=spec.find(':',begin);parts.push_back(spec.substr(begin,end-begin));if(end==std::string::npos)break;begin=end+1;}
        if(parts.size()!=4)throw std::invalid_argument("--auto-roi must be ID:SENSOR:CIF:ISP");
        c.auto_roi_bindings.push_back({parts[0],parts[1],parts[2],parts[3]});
      }
      else if(a=="--camera"){
        auto v=value(); auto p=v.find(':');
        if(p==std::string::npos) throw std::invalid_argument("camera must be ID:DEVICE@WIDTHxHEIGHT");
        std::string id=v.substr(0,p), device=v.substr(p+1);
        int width=640,height=360; auto at=device.find('@');
        if(at!=std::string::npos){
          std::string size=device.substr(at+1);device.resize(at);
          auto x=size.find('x');if(x==std::string::npos)throw std::invalid_argument("camera size must be WIDTHxHEIGHT");
          width=std::stoi(size.substr(0,x));height=std::stoi(size.substr(x+1));
        }
        if(width<16||height<16)throw std::invalid_argument("camera dimensions are too small");
        if(!camera_argument){c.cameras.clear();camera_argument=true;}
        c.cameras.push_back({id,device,width,height,3840,2160});
      }
      else if(a=="--help"){usage(argv[0]);return 0;} else throw std::invalid_argument("unknown option: "+a); }
    if(c.roi_capture_fps<60||c.roi_capture_fps>220)
      throw std::invalid_argument("--roi-fps must be 60..220");
    if(c.roi_lost_return_ms<100||c.roi_lost_return_ms>10000)
      throw std::invalid_argument("--roi-lost-return-ms must be 100..10000");
    if(c.model.empty()||c.labels.empty()){usage(argv[0]);return 2;}
    dual::DetectorService app(c);signal(SIGPIPE,SIG_IGN);signal(SIGINT,signal_handler);signal(SIGTERM,signal_handler);
    std::atomic<bool> done{false};std::thread watcher([&]{while(!done&&!stop_requested)std::this_thread::sleep_for(std::chrono::milliseconds(50));if(stop_requested)app.notify_stop();});
    app.run();done=true;watcher.join();
  }catch(const std::exception& e){std::cerr<<"fatal: "<<e.what()<<'\n';return 1;} return 0;
}
