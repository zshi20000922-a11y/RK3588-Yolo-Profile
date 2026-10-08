#include "yolo_postprocess.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace dual {
namespace {
constexpr int kClasses = 80;
constexpr int kProp = 5 + kClasses;
constexpr int kAnchors[3][6] = {
  {10,13,16,30,33,23}, {30,61,62,45,59,119}, {116,90,156,198,373,326}};

struct Candidate { float x, y, w, h, score; int cls; };
float dequant(int8_t v, int32_t zp, float scale) { return (v - zp) * scale; }
int8_t quant(float v, int32_t zp, float scale) {
  return static_cast<int8_t>(std::clamp(static_cast<int>(std::round(v / scale + zp)), -128, 127));
}
float iou(const Candidate& a, const Candidate& b) {
  float l = std::max(a.x, b.x), t = std::max(a.y, b.y);
  float r = std::min(a.x + a.w, b.x + b.w);
  float bot = std::min(a.y + a.h, b.y + b.h);
  float inter = std::max(0.0f, r-l) * std::max(0.0f, bot-t);
  float uni = a.w*a.h + b.w*b.h - inter;
  return uni > 0 ? inter / uni : 0;
}
void decode_head(const int8_t* input, int head, int32_t zp, float scale,
                 int input_width, int input_height, float threshold,
                 bool nc1hwc2, std::vector<Candidate>& out) {
  int stride = 8 << head, gw = input_width / stride, gh = input_height / stride;
  int grid = gw * gh; int8_t qth = quant(threshold, zp, scale);
  for (int a = 0; a < 3; ++a) for (int y = 0; y < gh; ++y)
    for (int x = 0; x < gw; ++x) {
      int cell = y * gw + x;
      const int channel_base=kProp*a;
      auto value=[&](int channel) -> int8_t {
        channel+=channel_base;
        if(!nc1hwc2) return input[channel*grid+cell];
        return input[(channel/16)*grid*16+cell*16+(channel%16)];
      };
      int8_t objectness = value(4);
      if (objectness < qth) continue;
      int cls = 0; int8_t best = value(5);
      for (int c = 1; c < kClasses; ++c) {
        int8_t p = value(5+c); if (p > best) { best=p; cls=c; }
      }
      float score = dequant(objectness,zp,scale) * dequant(best,zp,scale);
      if (score < threshold) continue;
      float bx=(dequant(value(0),zp,scale)*2-.5f+x)*stride;
      float by=(dequant(value(1),zp,scale)*2-.5f+y)*stride;
      float bw=dequant(value(2),zp,scale)*2;
      float bh=dequant(value(3),zp,scale)*2;
      bw=bw*bw*kAnchors[head][a*2]; bh=bh*bh*kAnchors[head][a*2+1];
      out.push_back({bx-bw/2, by-bh/2, bw, bh, score, cls});
    }
}
}

YoloPostprocessor::YoloPostprocessor(const std::string& path) {
  std::ifstream f(path); if (!f) throw std::runtime_error("cannot open labels: " + path);
  std::string line; while (std::getline(f,line)) { if (!line.empty() && line.back()=='\r') line.pop_back(); labels_.push_back(line); }
  if (labels_.size() != kClasses) throw std::runtime_error("labels file must contain 80 lines");
}

std::vector<Detection> YoloPostprocessor::decode(const int8_t* a, const int8_t* b,
    const int8_t* c, const std::vector<int32_t>& zps, const std::vector<float>& scales,
    const Letterbox& lb, float conf, float nms_threshold) const {
  return decode_layout(a,b,c,zps,scales,lb,conf,nms_threshold,false);
}

std::vector<Detection> YoloPostprocessor::decode_native(const int8_t* a,
    const int8_t* b, const int8_t* c, const std::vector<int32_t>& zps,
    const std::vector<float>& scales, const Letterbox& lb, float conf,
    float nms_threshold) const {
  return decode_layout(a,b,c,zps,scales,lb,conf,nms_threshold,true);
}

std::vector<Detection> YoloPostprocessor::decode_layout(const int8_t* a,
    const int8_t* b, const int8_t* c, const std::vector<int32_t>& zps,
    const std::vector<float>& scales, const Letterbox& lb, float conf,
    float nms_threshold, bool nc1hwc2) const {
  if (zps.size()<3 || scales.size()<3) throw std::runtime_error("model must have 3 quantized outputs");
  std::vector<Candidate> candidates;
  decode_head(a,0,zps[0],scales[0],lb.target_width,lb.target_height,conf,nc1hwc2,candidates);
  decode_head(b,1,zps[1],scales[1],lb.target_width,lb.target_height,conf,nc1hwc2,candidates);
  decode_head(c,2,zps[2],scales[2],lb.target_width,lb.target_height,conf,nc1hwc2,candidates);
  std::sort(candidates.begin(),candidates.end(),[](auto& x,auto& y){return x.score>y.score;});
  std::vector<Candidate> kept;
  for (const auto& x : candidates) {
    bool suppress=false; for (const auto& y : kept)
      if (x.cls==y.cls && iou(x,y)>nms_threshold) { suppress=true; break; }
    if (!suppress) { kept.push_back(x); if (kept.size()==64) break; }
  }
  std::vector<Detection> result; result.reserve(kept.size());
  for (const auto& x : kept) result.push_back({x.cls, labels_[x.cls], x.score,
    lb.source_x(x.x), lb.source_y(x.y), lb.source_x(x.x+x.w), lb.source_y(x.y+x.h)});
  return result;
}

std::string to_json(const DetectionBatch& b) {
  std::ostringstream o; o << "{\"camera\":\"" << json_escape(b.camera_id)
    << "\",\"sequence\":" << b.sequence << ",\"capture_ts_ns\":" << b.capture_ts_ns
    << ",\"worker_id\":" << b.worker_id << ",\"npu_core\":" << b.npu_core
    << ",\"width\":" << b.width << ",\"height\":" << b.height << ",\"detections\":[";
  for (size_t i=0;i<b.detections.size();++i) { const auto& d=b.detections[i]; if(i)o<<',';
    o << "{\"class_id\":"<<d.class_id<<",\"label\":\""<<json_escape(d.label)
      <<"\",\"score\":"<<d.score<<",\"box\":["<<d.left<<','<<d.top<<','<<d.right<<','<<d.bottom<<"]}"; }
  const auto& t=b.timings;
  o << "],\"timings_ms\":{\"capture_dequeue\":"<<t.capture_dequeue_ms
    <<",\"queue_wait\":"<<t.queue_wait_ms<<",\"rga\":"<<t.rga_ms
    <<",\"inference_queue\":"<<t.inference_queue_ms
    <<",\"inference\":"<<t.inference_ms<<",\"postprocess\":"<<t.postprocess_ms
    <<",\"postprocess_queue\":"<<t.postprocess_queue_ms
    <<",\"capture_to_result\":"<<t.capture_to_result_ms<<"},\"stats\":{\"dropped_frames\":"
    <<b.dropped_frames<<",\"v4l2_timeouts\":"<<b.v4l2_timeouts<<",\"reconnects\":"<<b.reconnects<<"}}\n";
  return o.str();
}
} // namespace dual
