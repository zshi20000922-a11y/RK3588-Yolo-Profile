#include "roi_controller.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc,char** argv){
  std::string sensor,cif,isp,mode;int left=0,top=0;
  try{
    for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("missing "+a);return std::string(argv[i]);};
      if(a=="--sensor")sensor=value();else if(a=="--cif")cif=value();
      else if(a=="--isp")isp=value();else if(a=="--mode")mode=value();
      else if(a=="--left")left=std::stoi(value());else if(a=="--top")top=std::stoi(value());
      else throw std::runtime_error("unknown option "+a);
    }
    if(sensor.empty()||cif.empty()||isp.empty()||(mode!="full"&&mode!="roi"))
      throw std::runtime_error("usage: stereo_roi_ctl --sensor DEV --cif DEV --isp DEV --mode full|roi [--left X --top Y]");
    dual::RoiController controller(sensor,cif,isp);std::string error;bool ok;
    if(mode=="roi")ok=controller.enter_roi({left+320.0,top+320.0},&error);
    else ok=controller.enter_full(&error);
    if(!ok)throw std::runtime_error(error);
    auto r=controller.roi();std::cout<<"mode="<<mode<<" roi=("<<r.left<<','<<r.top<<")/"<<r.width<<'x'<<r.height<<" drop_frames=1\n";
  }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}
  return 0;
}
