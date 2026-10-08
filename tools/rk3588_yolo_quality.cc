#include "rkvs/config.hpp"
#include "rkvs/rknn_detector.hpp"
#include "rkvs/types.hpp"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
struct DmaHeapAllocationData{uint64_t len;uint32_t fd;uint32_t fd_flags;uint64_t heap_flags;};
#define RK_QUALITY_DMA_HEAP_IOC_ALLOC _IOWR('H',0x0,DmaHeapAllocationData)
struct Options{std::string model,manifest,labels,family,input_list,detections,timings;int warmup=20;float confidence=0.001f;};
std::string next(int&i,int n,char**v){if(++i>=n)throw std::invalid_argument("missing value");return v[i];}
Options parse(int n,char**v){Options o;for(int i=1;i<n;++i){std::string a=v[i];if(a=="--model")o.model=next(i,n,v);else if(a=="--manifest")o.manifest=next(i,n,v);else if(a=="--labels")o.labels=next(i,n,v);else if(a=="--family")o.family=next(i,n,v);else if(a=="--input-list")o.input_list=next(i,n,v);else if(a=="--detections")o.detections=next(i,n,v);else if(a=="--timings")o.timings=next(i,n,v);else if(a=="--warmup")o.warmup=std::stoi(next(i,n,v));else if(a=="--confidence")o.confidence=std::stof(next(i,n,v));else throw std::invalid_argument("unknown option "+a);}if(o.model.empty()||o.manifest.empty()||o.labels.empty()||o.family.empty()||o.input_list.empty()||o.detections.empty()||o.timings.empty())throw std::invalid_argument("required options missing");return o;}
rkvs::ModelFamily family(const std::string&s){if(s=="yolov5")return rkvs::ModelFamily::kYolov5;if(s=="yolov8")return rkvs::ModelFamily::kYolov8;if(s=="yolo11")return rkvs::ModelFamily::kYolo11;if(s=="yolo26")return rkvs::ModelFamily::kYolo26;throw std::invalid_argument("family");}
int alloc_dma(size_t bytes){const char*hs[]={"/dev/dma_heap/cma","/dev/dma_heap/system-uncached","/dev/dma_heap/system"};for(auto p:hs){int h=open(p,O_RDWR|O_CLOEXEC);if(h<0)continue;DmaHeapAllocationData d{};d.len=bytes;d.fd_flags=O_RDWR|O_CLOEXEC;int rc=ioctl(h,RK_QUALITY_DMA_HEAP_IOC_ALLOC,&d);close(h);if(!rc)return d.fd;}throw std::runtime_error("DMA allocation");}
struct Input{std::string path;int width=0,height=0,image_id=0;};
std::vector<Input>read_list(const std::string&p){std::ifstream f(p);std::vector<Input>r;Input x;while(f>>x.path>>x.width>>x.height>>x.image_id)r.push_back(x);if(r.empty())throw std::runtime_error("empty list");return r;}
struct Dma{int fd=-1;void*ptr=nullptr;size_t bytes=0;~Dma(){if(ptr)munmap(ptr,bytes);if(fd>=0)close(fd);}};
std::shared_ptr<Dma>load(const Input&i){auto d=std::make_shared<Dma>();d->bytes=(size_t)i.width*i.height*3/2;d->fd=alloc_dma(d->bytes);d->ptr=mmap(nullptr,d->bytes,PROT_READ|PROT_WRITE,MAP_SHARED,d->fd,0);if(d->ptr==MAP_FAILED){d->ptr=nullptr;throw std::runtime_error("mmap");}std::ifstream f(i.path,std::ios::binary);if(!f.read((char*)d->ptr,d->bytes))throw std::runtime_error("short NV12 "+i.path);return d;}
rkvs::SharedFrame make_frame(const Input&i,const std::shared_ptr<Dma>&d,uint64_t seq){auto f=std::make_shared<rkvs::FrameRef>();f->source_id=std::to_string(i.image_id);f->sequence=seq;f->capture_ts_ns=f->receive_ts_ns=rkvs::monotonic_ns();f->format=rkvs::PixelFormat::kNv12;f->width=i.width;f->height=i.height;f->planes.push_back({d->fd,0,(uint32_t)i.width,(uint32_t)d->bytes});return f;}
}
int main(int argc,char**argv)try{auto o=parse(argc,argv);auto inputs=read_list(o.input_list);rkvs::ModelConfig c;c.path=o.model;c.manifest=o.manifest;c.labels=o.labels;c.family=family(o.family);c.contexts=1;c.confidence=o.confidence;rkvs::RknnDetectorPool detector(c);auto first=load(inputs.front());for(int i=0;i<o.warmup;++i)detector.process(make_frame(inputs.front(),first,i),0);std::ofstream det(o.detections),tim(o.timings);tim<<"image_id,width,height,rga_ms,inference_ms,output_sync_ms,postprocess_ms,e2e_ms,detections\n";uint64_t seq=0;for(const auto&i:inputs){auto dma=load(i);auto r=detector.process(make_frame(i,dma,seq++),0);auto&t=r.timings;tim<<i.image_id<<','<<i.width<<','<<i.height<<','<<t.rga_ms<<','<<t.inference_ms<<','<<t.output_sync_ms<<','<<t.postprocess_ms<<','<<t.capture_to_result_ms<<','<<r.detections.size()<<'\n';for(const auto&d:r.detections)det<<"{\"image_id\":"<<i.image_id<<",\"class_id\":"<<d.class_id<<",\"score\":"<<d.score<<",\"bbox\":["<<d.box.left<<','<<d.box.top<<','<<d.box.right-d.box.left<<','<<d.box.bottom-d.box.top<<"]}\n";}std::cout<<"images="<<inputs.size()<<'\n';return 0;}catch(const std::exception&e){std::cerr<<"quality failed: "<<e.what()<<'\n';return 2;}
