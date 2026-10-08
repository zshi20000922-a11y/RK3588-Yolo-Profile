#include "rknn_worker.hpp"
#include "im2d.h"
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace dual {
namespace {
double ms(uint64_t a, uint64_t b) { return static_cast<double>(b-a)/1e6; }
std::vector<unsigned char> read_file(const std::string& path) {
  std::ifstream f(path,std::ios::binary);
  if(!f) throw std::runtime_error("cannot open model: "+path);
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
struct DmaHeapAllocationData {
  uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags;
};
#define DUAL_DMA_HEAP_IOC_ALLOC _IOWR('H', 0x0, DmaHeapAllocationData)
int alloc_dmabuf(size_t size) {
  const char* heaps[]={"/dev/dma_heap/cma","/dev/dma_heap/system-uncached",
                       "/dev/dma_heap/system"};
  for(const char* path:heaps) {
    int heap=open(path,O_RDWR|O_CLOEXEC); if(heap<0) continue;
    DmaHeapAllocationData d{}; d.len=size; d.fd_flags=O_RDWR|O_CLOEXEC;
    int ret=ioctl(heap,DUAL_DMA_HEAP_IOC_ALLOC,&d); close(heap);
    if(ret==0) return static_cast<int>(d.fd);
  }
  return -1;
}
}

void RknnWorker::check(int ret, const char* op) {
  if (ret < 0)
    throw std::runtime_error(std::string(op)+" failed: "+std::to_string(ret));
}

RknnWorker::TensorSlot& RknnWorker::checked_slot(int slot) {
  if(slot<0 || slot>=kTensorSlots) throw std::out_of_range("invalid tensor slot");
  return slots_[slot];
}

RknnWorker::RknnWorker(rknn_context context, int id, int core,
                       const std::string& labels)
    : id_(id), ctx_(context), post_(labels) {
  rknn_core_mask mask = core==0 ? RKNN_NPU_CORE_0 :
                        core==1 ? RKNN_NPU_CORE_1 : RKNN_NPU_CORE_2;
  check(rknn_set_core_mask(ctx_,mask),"rknn_set_core_mask");
  check(rknn_query(ctx_,RKNN_QUERY_IN_OUT_NUM,&io_,sizeof(io_)),"query io count");
  if (io_.n_input!=1 || io_.n_output!=3)
    throw std::runtime_error("expected YOLOv5 1 input / 3 outputs");

  input_attr_.index=0;
  check(rknn_query(ctx_,RKNN_QUERY_NATIVE_INPUT_ATTR,&input_attr_,
                   sizeof(input_attr_)),"query native input");
  // Bind the NPU-native layout.  Binding the logical tensor with
  // pass_through=0 makes the runtime normalize/repack every frame inside
  // rknn_run and hides that cost in the inference timing.
  if(input_attr_.fmt!=RKNN_TENSOR_NHWC)
    throw std::runtime_error("native YOLO input must be NHWC");
  input_attr_.type=RKNN_TENSOR_UINT8;
  if(input_attr_.n_dims!=4 || input_attr_.dims[3]!=3)
    throw std::runtime_error("native YOLO input must be NHWC RGB");
  input_height_=input_attr_.dims[1]; input_width_=input_attr_.dims[2];
  if(input_width_<320 || input_height_<320 || input_width_%32 || input_height_%32)
    throw std::runtime_error("YOLO input dimensions must be >=320 and divisible by 32");
  if(!input_attr_.size_with_stride)
    input_attr_.size_with_stride=input_attr_.size;
  if(id_==0)
    std::cerr<<"native input fmt="<<input_attr_.fmt
             <<" dims="<<input_attr_.dims[0]<<'x'<<input_attr_.dims[1]
             <<'x'<<input_attr_.dims[2]<<'x'<<input_attr_.dims[3]
             <<" w_stride="<<input_attr_.w_stride
             <<" size="<<input_attr_.size
             <<" stride_size="<<input_attr_.size_with_stride<<'\n';

  output_attrs_.resize(3);
  for (int i=0;i<3;++i) {
    rknn_tensor_attr logical{}; logical.index=i;
    check(rknn_query(ctx_,RKNN_QUERY_OUTPUT_ATTR,&logical,sizeof(logical)),
          "query logical output");
    auto& a=output_attrs_[i]; a.index=i;
    check(rknn_query(ctx_,RKNN_QUERY_NATIVE_OUTPUT_ATTR,&a,sizeof(a)),
          "query native output");
    if(a.type!=RKNN_TENSOR_INT8)
      throw std::runtime_error("only INT8 YOLO outputs are supported");
    if(a.fmt!=RKNN_TENSOR_NC1HWC2)
      throw std::runtime_error("native YOLO outputs must be NC1HWC2");
    zps_.push_back(a.zp); scales_.push_back(a.scale);
    if(id_==0) {
      std::cerr<<"output["<<i<<"] logical fmt="<<logical.fmt
               <<" type="<<logical.type
               <<" dims=";
      for(uint32_t d=0;d<logical.n_dims;++d)
        std::cerr<<(d?"x":"")<<logical.dims[d];
      std::cerr<<" size="<<logical.size<<" stride="<<logical.size_with_stride
               <<"; native fmt="<<a.fmt<<" type="<<a.type
               <<" dims=";
      for(uint32_t d=0;d<a.n_dims;++d)
        std::cerr<<(d?"x":"")<<a.dims[d];
      std::cerr<<" size="<<a.size<<" stride="<<a.size_with_stride<<'\n';
    }
  }

  for(auto& slot:slots_) {
    slot.input_mapping_size=input_attr_.size_with_stride;
    slot.input_dmabuf_fd=alloc_dmabuf(slot.input_mapping_size);
    if(slot.input_dmabuf_fd<0)
      throw std::runtime_error("DMA heap allocation for RKNN input failed");
    slot.input_mapping=mmap(nullptr,slot.input_mapping_size,
                            PROT_READ|PROT_WRITE,MAP_SHARED,
                            slot.input_dmabuf_fd,0);
    if(slot.input_mapping==MAP_FAILED) {
      slot.input_mapping=nullptr;
      throw std::runtime_error("mmap RKNN input failed");
    }
    memset(slot.input_mapping,114,slot.input_mapping_size);
    slot.input_mem=rknn_create_mem_from_fd(ctx_,slot.input_dmabuf_fd,
                                           slot.input_mapping,
                                           slot.input_mapping_size,0);
    if(!slot.input_mem)
      throw std::runtime_error("rknn_create_mem_from_fd input failed");
    slot.output_mems.resize(3,nullptr);
    for(int i=0;i<3;++i) {
      const auto& a=output_attrs_[i];
      slot.output_mems[i]=rknn_create_mem(ctx_,
          a.size_with_stride ? a.size_with_stride : a.size);
      if(!slot.output_mems[i])
        throw std::runtime_error("rknn_create_mem output failed");
    }
  }
}

RknnWorker::~RknnWorker() {
  for(auto& slot:slots_) {
    for(auto* m:slot.output_mems) if(m) rknn_destroy_mem(ctx_,m);
    if(slot.input_mem) rknn_destroy_mem(ctx_,slot.input_mem);
    if(slot.input_mapping) munmap(slot.input_mapping,slot.input_mapping_size);
    if(slot.input_dmabuf_fd>=0) close(slot.input_dmabuf_fd);
  }
  if(ctx_) rknn_destroy(ctx_);
}

int RknnWorker::try_acquire_slot() {
  for(int i=0;i<kTensorSlots;++i) {
    bool expected=false;
    if(slots_[i].in_use.compare_exchange_strong(expected,true,
                                                std::memory_order_acq_rel))
      return i;
  }
  return -1;
}

void RknnWorker::release_slot(int slot) {
  checked_slot(slot).in_use.store(false,std::memory_order_release);
}

std::unique_ptr<PipelineJob> RknnWorker::preprocess(
    const FrameHandle& f, size_t camera_index, int slot_index,
    bool make_preview) {
  auto& slot=checked_slot(slot_index);
  if(!slot.in_use.load(std::memory_order_acquire))
    throw std::runtime_error("preprocess called with an unreserved slot");
  auto job=std::make_unique<PipelineJob>();
  job->worker_id=id_; job->slot=slot_index; job->camera_index=camera_index;
  job->make_preview=make_preview;
  job->batch.camera_id=f.camera_id; job->batch.sequence=f.sequence;
  job->batch.worker_id=id_; job->batch.npu_core=id_%3;
  job->batch.capture_ts_ns=f.capture_ts_ns;
  job->batch.width=f.width; job->batch.height=f.height;
  job->batch.timings.capture_dequeue_ms=ms(f.capture_ts_ns,f.dequeue_ts_ns);
  uint64_t begin=monotonic_ns();
  job->batch.timings.queue_wait_ms=ms(f.dequeue_ts_ns,begin);
  job->letterbox=Letterbox::make(f.width,f.height,input_width_,input_height_);
  const auto& lb=job->letterbox;
  rga_buffer_t src=wrapbuffer_fd(f.dmabuf_fd,f.width,f.height,
                                 RK_FORMAT_YCbCr_420_SP,f.stride,f.height);
  rga_buffer_t dst=wrapbuffer_fd(slot.input_mem->fd,input_width_,input_height_,
                                 RK_FORMAT_RGB_888,
                                 static_cast<int>(input_attr_.w_stride ?
                                     input_attr_.w_stride : input_width_),
                                 static_cast<int>(input_attr_.h_stride ?
                                     input_attr_.h_stride : input_height_));
  rga_buffer_t pat{}; im_rect sr{0,0,f.width,f.height};
  im_rect dr{lb.pad_left,lb.pad_top,lb.resized_width,lb.resized_height};
  im_rect pr{};
  IM_STATUS status=improcess(src,dst,pat,sr,dr,pr,IM_SYNC);
  if(status!=IM_STATUS_SUCCESS)
    throw std::runtime_error(std::string("RGA letterbox: ")+imStrError(status));
  uint64_t done=monotonic_ns();
  job->batch.timings.rga_ms=ms(begin,done);
  job->inference_queued_ns=done;
  return job;
}

void RknnWorker::run(PipelineJob& job) {
  auto& slot=checked_slot(job.slot);
  check(rknn_set_io_mem(ctx_,slot.input_mem,&input_attr_),"bind input memory");
  for(int i=0;i<3;++i)
    check(rknn_set_io_mem(ctx_,slot.output_mems[i],&output_attrs_[i]),
          "bind output memory");
  uint64_t begin=monotonic_ns();
  job.batch.timings.inference_queue_ms=
      ms(job.inference_queued_ns,begin);
  check(rknn_run(ctx_,nullptr),"rknn_run");
  uint64_t done=monotonic_ns();
  job.batch.timings.inference_ms=ms(begin,done);
  job.postprocess_queued_ns=done;
}

void RknnWorker::postprocess(PipelineJob& job) {
  auto& slot=checked_slot(job.slot);
  uint64_t begin=monotonic_ns();
  job.batch.timings.postprocess_queue_ms=
      ms(job.postprocess_queued_ns,begin);
  job.batch.detections=post_.decode_native(
      static_cast<int8_t*>(slot.output_mems[0]->virt_addr),
      static_cast<int8_t*>(slot.output_mems[1]->virt_addr),
      static_cast<int8_t*>(slot.output_mems[2]->virt_addr),
      zps_,scales_,job.letterbox);
  uint64_t done=monotonic_ns();
  job.batch.timings.postprocess_ms=ms(begin,done);
  job.batch.timings.capture_to_result_ms=
      ms(job.batch.capture_ts_ns,done);
  if(job.make_preview) {
    // Debug-only preview: copy and nearest-neighbour scale the small RKNN
    // tensor image, never the original 4K frame. The production path keeps
    // preview disabled and remains zero-copy.
    constexpr int preview_width=640, preview_height=360;
    job.batch.preview_rgb.resize(preview_width*preview_height*3);
    const auto* source=static_cast<const uint8_t*>(slot.input_mapping);
    const int source_stride=static_cast<int>(input_attr_.w_stride ?
                            input_attr_.w_stride : input_width_)*3;
    for(int y=0;y<preview_height;++y) {
      int sy=job.letterbox.pad_top+
             y*job.letterbox.resized_height/preview_height;
      auto* destination=job.batch.preview_rgb.data()+y*preview_width*3;
      for(int x=0;x<preview_width;++x) {
        int sx=job.letterbox.pad_left+
               x*job.letterbox.resized_width/preview_width;
        const auto* pixel=source+sy*source_stride+sx*3;
        destination[x*3]=pixel[0];
        destination[x*3+1]=pixel[1];
        destination[x*3+2]=pixel[2];
      }
    }
  }
}

RknnContextPool::RknnContextPool(const std::string& model,
                                 const std::string& labels,int count,
                                 bool independent_contexts) {
  if(count<1||count>12) throw std::invalid_argument("--contexts must be 1..12");
  auto data=read_file(model);
  std::vector<rknn_context> contexts;
  try {
    if(independent_contexts) {
      for(int i=0;i<count;++i) {
        rknn_context context=0;
        int ret=rknn_init(&context,data.data(),data.size(),0,nullptr);
        if(ret<0)
          throw std::runtime_error("rknn_init failed: "+std::to_string(ret));
        contexts.push_back(context);
      }
    } else {
      rknn_context base=0;
      int ret=rknn_init(&base,data.data(),data.size(),0,nullptr);
      if(ret<0)
        throw std::runtime_error("rknn_init failed: "+std::to_string(ret));
      contexts.push_back(base);
      for(int i=1;i<count;++i) {
        rknn_context context=0;
        ret=rknn_dup_context(&base,&context);
        if(ret<0)
          throw std::runtime_error("rknn_dup_context failed: "+std::to_string(ret));
        contexts.push_back(context);
      }
    }
    for(int i=0;i<count;++i) {
      auto ctx=contexts[i]; contexts[i]=0;
      workers_.push_back(std::make_unique<RknnWorker>(ctx,i,i%3,labels));
    }
  } catch(...) {
    for(auto ctx:contexts) if(ctx) rknn_destroy(ctx);
    throw;
  }
}

std::vector<std::unique_ptr<RknnWorker>> RknnContextPool::take_workers() {
  return std::move(workers_);
}
} // namespace dual
