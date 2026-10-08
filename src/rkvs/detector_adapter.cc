#include "rkvs/detector_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace rkvs {
namespace {

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  value = value.substr(first, last - first + 1);
  if (value.size() >= 2 && ((value.front() == '\"' && value.back() == '\"') ||
                            (value.front() == '\'' && value.back() == '\'')))
    value = value.substr(1, value.size() - 2);
  return value;
}

float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float overlap_with_areas(const Box& a, const Box& b, float area_a,
                        float area_b) {
  const float left = std::max(a.left, b.left);
  const float top = std::max(a.top, b.top);
  const float right = std::min(a.right, b.right);
  const float bottom = std::min(a.bottom, b.bottom);
  const float intersection = std::max(0.0f, right - left) *
                            std::max(0.0f, bottom - top);
  const float total = area_a + area_b - intersection;
  return total > 0 ? intersection / total : 0;
}

bool allowed(int class_id, const DetectorParams& params) {
  return params.class_filter.empty() ||
         std::find(params.class_filter.begin(), params.class_filter.end(),
                   class_id) != params.class_filter.end();
}

std::pair<int, float> max_channel_range_at(const TensorView& view, int y, int x,
                                           int first_channel, int channels) {
  const int h = view.schema.dims.at(2), w = view.schema.dims.at(3);
  const size_t spatial = static_cast<size_t>(h) * w;
  const size_t position = static_cast<size_t>(y) * w + x;
  auto offset = [&](int channel) -> size_t {
    if (view.schema.layout == TensorLayout::kNchw)
      return static_cast<size_t>(channel) * spatial + position;
    const int c2 = view.schema.dims.at(4);
    return (static_cast<size_t>(channel / c2) * spatial + position) * c2 +
           channel % c2;
  };
  int best_class = 0;
  if (view.schema.type == TensorDataType::kInt8) {
    const auto* data = static_cast<const int8_t*>(view.data);
    int8_t best = data[offset(first_channel)];
    for (int c = 1; c < channels; ++c) {
      const int8_t value = data[offset(first_channel + c)];
      if (value > best) best = value, best_class = c;
    }
    return {best_class, (best - view.schema.zero_point) * view.schema.scale};
  }
  if (view.schema.type == TensorDataType::kUint8) {
    const auto* data = static_cast<const uint8_t*>(view.data);
    uint8_t best = data[offset(first_channel)];
    for (int c = 1; c < channels; ++c) {
      const uint8_t value = data[offset(first_channel + c)];
      if (value > best) best = value, best_class = c;
    }
    return {best_class, (best - view.schema.zero_point) * view.schema.scale};
  }
  const auto* data = static_cast<const float*>(view.data);
  float best = data[offset(first_channel)];
  for (int c = 1; c < channels; ++c) {
    const float value = data[offset(first_channel + c)];
    if (value > best) best = value, best_class = c;
  }
  return {best_class, best};
}

int quantized_min_for_score(const TensorSchema& schema, float threshold) {
  int minimum = 0, maximum = 0;
  if (schema.type == TensorDataType::kInt8) {
    minimum = -128;
    maximum = 127;
  } else if (schema.type == TensorDataType::kUint8) {
    minimum = 0;
    maximum = 255;
  } else {
    return std::numeric_limits<int>::min();
  }
  if (!(schema.scale > 0)) return minimum;
  const double boundary = std::ceil(
      static_cast<double>(threshold) / schema.scale + schema.zero_point);
  int q = boundary <= minimum ? minimum
                               : boundary > maximum ? maximum + 1
                                                    : static_cast<int>(boundary);
  const auto dequant = [&](int value) {
    return (value - schema.zero_point) * schema.scale;
  };
  // Match the original float comparison exactly at quantization boundaries.
  while (q <= maximum && dequant(q) < threshold) ++q;
  while (q > minimum && dequant(q - 1) >= threshold) --q;
  return q;
}

int quantized_value_at(const TensorView& view, int channel, int y, int x) {
  size_t index = 0;
  if (view.schema.layout == TensorLayout::kNchw) {
    const int h = view.schema.dims.at(2), w = view.schema.dims.at(3);
    index = (static_cast<size_t>(channel) * h + y) * w + x;
  } else if (view.schema.layout == TensorLayout::kNc1hwc2) {
    const int h = view.schema.dims.at(2), w = view.schema.dims.at(3);
    const int c2 = view.schema.dims.at(4);
    index = ((static_cast<size_t>(channel / c2) * h + y) * w + x) * c2 +
            channel % c2;
  } else {
    throw std::invalid_argument("quantized threshold requires NCHW/NC1HWC2");
  }
  if (view.schema.type == TensorDataType::kInt8)
    return static_cast<const int8_t*>(view.data)[index];
  if (view.schema.type == TensorDataType::kUint8)
    return static_cast<const uint8_t*>(view.data)[index];
  throw std::invalid_argument("quantized threshold requires integer tensor");
}

std::pair<int, float> max_class_at(const TensorView& view, int y, int x,
                                   int classes, float minimum_score) {
  // RKNN's native NC1HWC2 class tensor stores the C2 lanes contiguously for
  // each spatial position. Walk those small blocks directly instead of
  // recomputing channel/C2 division and modulo for each of the 80 classes.
  // This is the hottest score scan in the YOLO26 raw-head path.
  if (view.schema.layout == TensorLayout::kNc1hwc2 &&
      view.schema.dims.size() == 5 && classes > 0) {
    const int c1 = view.schema.dims[1];
    const int h = view.schema.dims[2];
    const int w = view.schema.dims[3];
    const int c2 = view.schema.dims[4];
    if (c1 * c2 >= classes && c2 > 0) {
      const size_t spatial = static_cast<size_t>(h) * w;
      const size_t position = static_cast<size_t>(y) * w + x;
#if defined(__aarch64__) || defined(__ARM_NEON)
      // RK3588 is AArch64/NEON. In the common RKNN INT8 layout C2=16, so a
      // single cell's 80 class scores are five contiguous 128-bit vectors.
      // Reduce each vector in parallel, then scan only the winning 16 lanes
      // to recover the first argmax (same tie behaviour as the scalar path).
      // This keeps quantization and candidate/NMS semantics exactly intact.
      if (c2 == 16 && classes % c2 == 0 &&
          view.schema.type == TensorDataType::kInt8) {
        const auto* data = static_cast<const int8_t*>(view.data);
        int8_t best = std::numeric_limits<int8_t>::min();
        int best_group = 0;
        for (int group = 0; group < classes / c2; ++group) {
          const int8x16_t lanes = vld1q_s8(data +
              (static_cast<size_t>(group) * spatial + position) * c2);
          const int8_t group_max = vmaxvq_s8(lanes);
          if (group_max > best) best = group_max, best_group = group;
        }
        const float score = (best - view.schema.zero_point) * view.schema.scale;
        if (score < minimum_score) return {-1, score};
        const int8_t* lanes = data +
            (static_cast<size_t>(best_group) * spatial + position) * c2;
        int lane = 0;
        while (lanes[lane] != best) ++lane;
        return {best_group * c2 + lane, score};
      }
      if (c2 == 16 && classes % c2 == 0 &&
          view.schema.type == TensorDataType::kUint8) {
        const auto* data = static_cast<const uint8_t*>(view.data);
        uint8_t best = std::numeric_limits<uint8_t>::min();
        int best_group = 0;
        for (int group = 0; group < classes / c2; ++group) {
          const uint8x16_t lanes = vld1q_u8(data +
              (static_cast<size_t>(group) * spatial + position) * c2);
          const uint8_t group_max = vmaxvq_u8(lanes);
          if (group_max > best) best = group_max, best_group = group;
        }
        const float score = (best - view.schema.zero_point) * view.schema.scale;
        if (score < minimum_score) return {-1, score};
        const uint8_t* lanes = data +
            (static_cast<size_t>(best_group) * spatial + position) * c2;
        int lane = 0;
        while (lanes[lane] != best) ++lane;
        return {best_group * c2 + lane, score};
      }
#endif
      int best_class = 0;
      if (view.schema.type == TensorDataType::kInt8) {
        const auto* data = static_cast<const int8_t*>(view.data);
        int8_t best = data[position * c2];
        for (int group = 0; group < c1; ++group) {
          const int8_t* lanes = data +
              (static_cast<size_t>(group) * spatial + position) * c2;
          const int count = std::min(c2, classes - group * c2);
          for (int lane = group == 0 ? 1 : 0; lane < count; ++lane) {
            if (lanes[lane] > best) {
              best = lanes[lane];
              best_class = group * c2 + lane;
            }
          }
        }
        return {best_class,
                (best - view.schema.zero_point) * view.schema.scale};
      }
      if (view.schema.type == TensorDataType::kUint8) {
        const auto* data = static_cast<const uint8_t*>(view.data);
        uint8_t best = data[position * c2];
        for (int group = 0; group < c1; ++group) {
          const uint8_t* lanes = data +
              (static_cast<size_t>(group) * spatial + position) * c2;
          const int count = std::min(c2, classes - group * c2);
          for (int lane = group == 0 ? 1 : 0; lane < count; ++lane) {
            if (lanes[lane] > best) {
              best = lanes[lane];
              best_class = group * c2 + lane;
            }
          }
        }
        return {best_class,
                (best - view.schema.zero_point) * view.schema.scale};
      }
      const auto* data = static_cast<const float*>(view.data);
      float best = data[position * c2];
      for (int group = 0; group < c1; ++group) {
        const float* lanes = data +
            (static_cast<size_t>(group) * spatial + position) * c2;
        const int count = std::min(c2, classes - group * c2);
        for (int lane = group == 0 ? 1 : 0; lane < count; ++lane) {
          if (lanes[lane] > best) {
            best = lanes[lane];
            best_class = group * c2 + lane;
          }
        }
      }
      if (best < minimum_score) return {-1, best};
      return {best_class, best};
    }
  }
  auto result = max_channel_range_at(view, y, x, 0, classes);
  return result.second < minimum_score ? std::pair<int, float>{-1, result.second}
                                       : result;
}

std::vector<Detection> suppress(std::vector<Detection> candidates,
                                const DetectorParams& params, bool use_nms) {
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });
  std::vector<Detection> result;
  result.reserve(std::min(candidates.size(), static_cast<size_t>(
      std::max(params.max_detections, 0))));
  int max_class_id = -1;
  for (const auto& candidate : candidates)
    max_class_id = std::max(max_class_id, candidate.class_id);
  std::vector<std::vector<size_t>> kept_by_class;
  std::vector<size_t> kept_global;
  std::vector<float> areas;
  if (use_nms) {
    if (!params.class_agnostic_nms)
      kept_by_class.resize(static_cast<size_t>(max_class_id + 1));
    else
      kept_global.reserve(static_cast<size_t>(std::max(params.max_detections, 0)));
    areas.reserve(candidates.size());
    for (const auto& candidate : candidates) areas.push_back(candidate.box.area());
  }
  std::vector<size_t> selected;
  selected.reserve(result.capacity());
  for (size_t index = 0; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    if (!allowed(candidate.class_id, params)) continue;
    bool rejected = false;
    if (use_nms) {
      const auto reject_from = [&](const std::vector<size_t>& kept) {
        for (const size_t kept_index : kept)
          if (overlap_with_areas(candidates[kept_index].box, candidate.box,
                                 areas[kept_index], areas[index]) >
              params.nms_iou) return true;
        return false;
      };
      if (params.class_agnostic_nms) {
        rejected = reject_from(kept_global);
      } else if (candidate.class_id >= 0 &&
                 candidate.class_id < static_cast<int>(kept_by_class.size())) {
        rejected = reject_from(kept_by_class[candidate.class_id]);
      }
    }
    if (!rejected) {
      selected.push_back(index);
      if (use_nms) {
        if (params.class_agnostic_nms) {
          kept_global.push_back(index);
        } else if (candidate.class_id >= 0 &&
                   candidate.class_id < static_cast<int>(kept_by_class.size())) {
          kept_by_class[candidate.class_id].push_back(index);
        }
      }
    }
    if (static_cast<int>(selected.size()) >= params.max_detections) break;
  }
  for (const size_t index : selected)
    result.push_back(std::move(candidates[index]));
  return result;
}

class AdapterBase : public IDetectorAdapter {
 public:
  explicit AdapterBase(std::vector<std::string> labels)
      : labels_(std::move(labels)) {
    if (labels_.empty()) throw std::invalid_argument("labels must not be empty");
  }

 protected:
  Detection detection(int class_id, float score, Box model_box,
                      const LetterboxTransform& transform) const {
    Detection result;
    result.class_id = class_id;
    result.score = score;
    result.box = transform.to_source(model_box);
    return result;
  }
  std::vector<Detection> finalize(std::vector<Detection> candidates,
                                  const DetectorParams& params,
                                  bool use_nms) const {
    auto result = suppress(std::move(candidates), params, use_nms);
    for (auto& detection : result)
      detection.label = detection.class_id >= 0 &&
                                detection.class_id < static_cast<int>(labels_.size())
                            ? labels_[detection.class_id]
                            : std::to_string(detection.class_id);
    return result;
  }
  std::vector<std::string> labels_;
};

class Yolov5Adapter final : public AdapterBase {
 public:
  using AdapterBase::AdapterBase;
  void validate(const ModelSchema& schema,
                const DetectorManifest& manifest) const override {
    validate_input(schema, manifest);
    if (schema.outputs.size() != 3)
      throw std::invalid_argument("YOLOv5 requires exactly 3 output heads");
    for (size_t i = 0; i < schema.outputs.size(); ++i) {
      const auto& output = schema.outputs[i];
      if (output.layout != TensorLayout::kNchw &&
          output.layout != TensorLayout::kNc1hwc2)
        throw std::invalid_argument("YOLOv5 outputs must be NCHW or NC1HWC2");
      const int channels = output.layout == TensorLayout::kNchw
                               ? output.dims.at(1)
                               : output.dims.at(1) * output.dims.at(4);
      if (channels < 3 * (5 + static_cast<int>(labels_.size())))
        throw std::invalid_argument("YOLOv5 output channel count is too small");
    }
  }

  std::vector<Detection> decode(const std::vector<TensorView>& outputs,
      const LetterboxTransform& t, const DetectorParams& p) const override {
    if (outputs.size() != 3) throw std::invalid_argument("YOLOv5 output count");
    static constexpr int anchors[3][6] = {
        {10, 13, 16, 30, 33, 23}, {30, 61, 62, 45, 59, 119},
        {116, 90, 156, 198, 373, 326}};
    std::vector<Detection> candidates;
    const int properties = 5 + static_cast<int>(labels_.size());
    for (int head = 0; head < 3; ++head) {
      const auto& view = outputs[head];
      const int h = view.schema.layout == TensorLayout::kNchw
                        ? view.schema.dims.at(2) : view.schema.dims.at(2);
      const int w = view.schema.layout == TensorLayout::kNchw
                        ? view.schema.dims.at(3) : view.schema.dims.at(3);
      const int stride = t.target_width / w;
      for (int anchor = 0; anchor < 3; ++anchor)
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x) {
            const int base = anchor * properties;
            const float objectness = view.at(base + 4, y, x);
            if (objectness < p.confidence) continue;
            auto [class_id, class_score] = max_channel_range_at(
                view, y, x, base + 5, static_cast<int>(labels_.size()));
            const float score = objectness * class_score;
            if (score < p.confidence) continue;
            const float cx = (view.at(base, y, x) * 2 - 0.5f + x) * stride;
            const float cy = (view.at(base + 1, y, x) * 2 - 0.5f + y) * stride;
            float width = view.at(base + 2, y, x) * 2;
            float height = view.at(base + 3, y, x) * 2;
            width = width * width * anchors[head][anchor * 2];
            height = height * height * anchors[head][anchor * 2 + 1];
            candidates.push_back(detection(class_id, score,
                {cx - width / 2, cy - height / 2,
                 cx + width / 2, cy + height / 2}, t));
          }
    }
    return finalize(std::move(candidates), p, true);
  }

 private:
  static void validate_input(const ModelSchema& schema,
                             const DetectorManifest& manifest) {
    if (schema.input_layout != TensorLayout::kNhwc || schema.input_channels != 3)
      throw std::invalid_argument("detector input must be NHWC RGB");
    if (manifest.input_width && schema.input_width != manifest.input_width)
      throw std::invalid_argument("manifest/model input width mismatch");
    if (manifest.input_height && schema.input_height != manifest.input_height)
      throw std::invalid_argument("manifest/model input height mismatch");
  }
  friend class DflAdapter;
  friend class Yolo26Adapter;
};

// Rockchip-optimized YOLOv8/11/26 exports three branches. Each branch is either
// [box_dfl, class] or [box_dfl, class, score_sum], hence 6 or 9 tensors.
class DflAdapter : public AdapterBase {
 public:
  DflAdapter(std::vector<std::string> labels, ModelFamily family)
      : AdapterBase(std::move(labels)), family_(family) {}
  void validate(const ModelSchema& schema,
                const DetectorManifest& manifest) const override {
    Yolov5Adapter::validate_input(schema, manifest);
    if (schema.outputs.size() != 6 && schema.outputs.size() != 9)
      throw std::invalid_argument("DFL detector requires 6 or 9 outputs");
    for (const auto& output : schema.outputs)
      if (output.layout != TensorLayout::kNchw &&
          output.layout != TensorLayout::kNc1hwc2)
        throw std::invalid_argument("DFL detector output layout unsupported");
    for (int branch = 0; branch < 3; ++branch) {
      const auto& box = schema.outputs[branch * (schema.outputs.size() / 3)];
      const int channels = box.layout == TensorLayout::kNchw
                               ? box.dims.at(1) : box.dims.at(1) * box.dims.at(4);
      // RKNN native NC1HWC2 pads YOLO26's logical 4-channel regression head
      // to 32 channels (2x16). The exported YOLO26 reg_max is still 1.
      const bool direct_ltrb = family_ == ModelFamily::kYolo26;
      if (!direct_ltrb && (channels < 8 || channels % 4))
        throw std::invalid_argument("DFL box channel count must be 4*reg_max");
    }
  }
  std::vector<Detection> decode(const std::vector<TensorView>& outputs,
      const LetterboxTransform& t, const DetectorParams& p) const override {
    if (outputs.size() != 6 && outputs.size() != 9)
      throw std::invalid_argument("DFL detector output count");
    const int per_branch = static_cast<int>(outputs.size() / 3);
    std::vector<Detection> candidates;
    // Match the v8/11 path: reserve a modest likely candidate count rather
    // than every YOLO26 grid cell (8,400 Detection objects at 640x640).
    // Most cells are rejected by score_sum/confidence; vector growth remains
    // correct for crowded frames, while avoiding a large per-frame allocation.
    candidates.reserve(256);
    for (int branch = 0; branch < 3; ++branch) {
      const auto& boxes = outputs[branch * per_branch];
      const auto& scores = outputs[branch * per_branch + 1];
      const TensorView* score_sum = per_branch == 3
                                        ? &outputs[branch * per_branch + 2]
                                        : nullptr;
      const int score_sum_min_q = score_sum
          ? quantized_min_for_score(score_sum->schema, p.confidence)
          : std::numeric_limits<int>::min();
      const int h = boxes.schema.dims.at(2), w = boxes.schema.dims.at(3);
      const int box_channels = boxes.schema.layout == TensorLayout::kNchw
                                   ? boxes.schema.dims.at(1)
                                   : boxes.schema.dims.at(1) * boxes.schema.dims.at(4);
      const int reg_max = family_ == ModelFamily::kYolo26 ? 1 : box_channels / 4;
      const float stride_x = static_cast<float>(t.target_width) / w;
      const float stride_y = static_cast<float>(t.target_height) / h;
      for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        // Rockchip's optimized 9-output graph supplies sum(sigmoid(classes)),
        // clamped to [0,1]. max(class) <= sum(class), so this is a lossless
        // early rejection at the same confidence threshold. It avoids 80
        // dequantized class reads for nearly every background grid cell.
        if (score_sum) {
          const bool quantized = score_sum->schema.type == TensorDataType::kInt8 ||
                                 score_sum->schema.type == TensorDataType::kUint8;
          if (quantized) {
            if (quantized_value_at(*score_sum, 0, y, x) < score_sum_min_q)
              continue;
          } else if (score_sum->at(0, y, x) < p.confidence) {
            continue;
          }
        }
        const float early_confidence = per_branch == 3
                                           ? p.confidence
                                           : std::numeric_limits<float>::lowest();
        auto [class_id, class_score] = max_class_at(
            scores, y, x, static_cast<int>(labels_.size()), early_confidence);
        // Six-output document exports expose raw cv3 logits. Nine-output Model
        // Zoo graphs already contain sigmoid plus a score-sum tensor.
        if (per_branch == 2) class_score = sigmoid(class_score);
        if (class_id < 0 || class_score < p.confidence) continue;
        float distance[4]{};
        for (int side = 0; side < 4; ++side) {
          if (family_ == ModelFamily::kYolo26) {
            distance[side] = boxes.at(side, y, x);
            continue;
          }
          float max_logit = -1e30f;
          for (int bin = 0; bin < reg_max; ++bin)
            max_logit = std::max(max_logit, boxes.at(side * reg_max + bin, y, x));
          float denominator = 0;
          for (int bin = 0; bin < reg_max; ++bin) {
            const float probability = std::exp(
                boxes.at(side * reg_max + bin, y, x) - max_logit);
            denominator += probability;
            distance[side] += bin * probability;
          }
          distance[side] /= denominator;
        }
        const float cx = x + 0.5f, cy = y + 0.5f;
        candidates.push_back(detection(class_id, class_score,
            {(cx - distance[0]) * stride_x, (cy - distance[1]) * stride_y,
             (cx + distance[2]) * stride_x, (cy + distance[3]) * stride_y}, t));
      }
    }
    // The current RKNN raw-head export disables end2end and uses the
    // one-to-many branch, so class-wise NMS is required. Only a separately
    // verified one-to-one export may bypass NMS.
    return finalize(std::move(candidates), p, true);
  }
 private:
  ModelFamily family_;
};

class Yolo26Adapter final : public AdapterBase {
 public:
  using AdapterBase::AdapterBase;
  void validate(const ModelSchema& schema,
                const DetectorManifest& manifest) const override {
    Yolov5Adapter::validate_input(schema, manifest);
    if (schema.outputs.size() != 1 ||
        schema.outputs[0].layout != TensorLayout::kFlat ||
        schema.outputs[0].dims.size() < 2 || schema.outputs[0].dims.back() != 6)
      throw std::invalid_argument(
          "YOLO26 requires one validated end-to-end [N,6] output");
    if (schema.outputs[0].type != TensorDataType::kFloat32 &&
        schema.outputs[0].type != TensorDataType::kInt8 &&
        schema.outputs[0].type != TensorDataType::kUint8)
      throw std::invalid_argument("YOLO26 output type unsupported");
  }
  std::vector<Detection> decode(const std::vector<TensorView>& outputs,
      const LetterboxTransform& t, const DetectorParams& p) const override {
    if (outputs.size() != 1 || !outputs[0].data)
      throw std::invalid_argument("YOLO26 output missing");
    const size_t rows = outputs[0].elements() / 6;
    std::vector<Detection> candidates;
    for (size_t row = 0; row < rows; ++row) {
      const size_t base = row * 6;
      const float score = outputs[0].scalar(base + 4);
      const int class_id = static_cast<int>(std::round(outputs[0].scalar(base + 5)));
      if (score < p.confidence || class_id < 0 ||
          class_id >= static_cast<int>(labels_.size())) continue;
      candidates.push_back(detection(class_id, score,
          {outputs[0].scalar(base), outputs[0].scalar(base + 1),
           outputs[0].scalar(base + 2), outputs[0].scalar(base + 3)}, t));
    }
    // YOLO26 end-to-end output is one-to-one and intentionally NMS-free.
    return finalize(std::move(candidates), p, false);
  }
};

// YOLO26 has two useful deployment protocols.  The official export is a
// one-to-one, NMS-free [1,300,6] tensor, while Rockchip's optimized export
// exposes nine raw heads.  Select from the validated tensor schema instead of
// silently treating every YOLO26 file as the raw-head variant.
class Yolo26DispatchAdapter final : public IDetectorAdapter {
 public:
  explicit Yolo26DispatchAdapter(const std::vector<std::string>& labels)
      : end_to_end_(labels), raw_(labels, ModelFamily::kYolo26) {}
  void validate(const ModelSchema& schema,
                const DetectorManifest& manifest) const override {
    if (schema.outputs.size() == 1) end_to_end_.validate(schema, manifest);
    else raw_.validate(schema, manifest);
  }
  std::vector<Detection> decode(const std::vector<TensorView>& outputs,
      const LetterboxTransform& transform,
      const DetectorParams& params) const override {
    return outputs.size() == 1 ? end_to_end_.decode(outputs, transform, params)
                               : raw_.decode(outputs, transform, params);
  }
 private:
  Yolo26Adapter end_to_end_;
  DflAdapter raw_;
};

}  // namespace

size_t TensorView::elements() const {
  size_t count = 1;
  for (int dimension : schema.dims) count *= static_cast<size_t>(dimension);
  return count;
}

float TensorView::at(int channel, int y, int x) const {
  if (!data) throw std::invalid_argument("tensor data is null");
  size_t index = 0;
  if (schema.layout == TensorLayout::kNchw) {
    if (schema.dims.size() != 4) throw std::invalid_argument("NCHW rank");
    const int h = schema.dims[2], w = schema.dims[3];
    index = (static_cast<size_t>(channel) * h + y) * w + x;
  } else if (schema.layout == TensorLayout::kNc1hwc2) {
    if (schema.dims.size() != 5) throw std::invalid_argument("NC1HWC2 rank");
    const int c2 = schema.dims[4], h = schema.dims[2], w = schema.dims[3];
    index = ((static_cast<size_t>(channel / c2) * h + y) * w + x) * c2 +
            channel % c2;
  } else {
    throw std::invalid_argument("tensor at() requires NCHW/NC1HWC2");
  }
  // at() is the hot path of detection post-processing. The caller iterates
  // validated tensor dimensions, so recomputing elements() and checking the
  // same bound for every class/bin (millions of times per frame) is wasteful.
  if (schema.type == TensorDataType::kInt8)
    return (static_cast<const int8_t*>(data)[index] - schema.zero_point) * schema.scale;
  if (schema.type == TensorDataType::kUint8)
    return (static_cast<const uint8_t*>(data)[index] - schema.zero_point) * schema.scale;
  return static_cast<const float*>(data)[index];
}

float TensorView::scalar(size_t index) const {
  if (!data || index >= elements()) throw std::out_of_range("tensor scalar index");
  if (schema.type == TensorDataType::kInt8)
    return (static_cast<const int8_t*>(data)[index] - schema.zero_point) * schema.scale;
  if (schema.type == TensorDataType::kUint8)
    return (static_cast<const uint8_t*>(data)[index] - schema.zero_point) * schema.scale;
  return static_cast<const float*>(data)[index];
}

DetectorManifest DetectorManifest::load(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open model manifest: " + path);
  DetectorManifest result;
  std::string section;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.resize(comment);
    const int indent = static_cast<int>(line.find_first_not_of(' '));
    line = trim(line);
    if (line.empty()) continue;
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("invalid model manifest line: " + line);
    const std::string key = trim(line.substr(0, colon));
    const std::string value = trim(line.substr(colon + 1));
    if (value.empty()) { section = key; continue; }
    const std::string full = indent > 0 ? section + "." + key : key;
    if (full == "family") {
      if (value == "yolov5") result.family = ModelFamily::kYolov5;
      else if (value == "yolov8") result.family = ModelFamily::kYolov8;
      else if (value == "yolo11") result.family = ModelFamily::kYolo11;
      else if (value == "yolo26") result.family = ModelFamily::kYolo26;
      else throw std::runtime_error("unsupported manifest family: " + value);
    } else if (full == "task") result.task = value;
    else if (full == "input.width") result.input_width = std::stoi(value);
    else if (full == "input.height") result.input_height = std::stoi(value);
    else if (full == "input.format") result.input_format = value;
    else if (full == "input.layout") {
      if (value != "nhwc") throw std::runtime_error("only NHWC input supported");
      result.input_layout = TensorLayout::kNhwc;
    } else if (full == "quantization") result.quantization = value;
    else if (full == "labels") result.labels_path = value;
    else if (full == "postprocess.confidence") result.postprocess.confidence = std::stof(value);
    else if (full == "postprocess.nms_iou") result.postprocess.nms_iou = std::stof(value);
    else if (full == "postprocess.max_detections") result.postprocess.max_detections = std::stoi(value);
    else if (full == "postprocess.class_agnostic_nms")
      result.postprocess.class_agnostic_nms = value == "true";
    else throw std::runtime_error("unknown model manifest key: " + full);
  }
  if (result.task != "detect") throw std::runtime_error("only detect task supported");
  return result;
}

std::vector<std::string> load_labels(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open labels: " + path);
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) labels.push_back(line);
  }
  if (labels.empty()) throw std::runtime_error("labels file is empty: " + path);
  return labels;
}

std::unique_ptr<IDetectorAdapter> make_detector_adapter(
    ModelFamily family, const std::vector<std::string>& labels) {
  switch (family) {
    case ModelFamily::kYolov5:
      return std::make_unique<Yolov5Adapter>(labels);
    case ModelFamily::kYolov8:
    case ModelFamily::kYolo11:
      return std::make_unique<DflAdapter>(labels, family);
    case ModelFamily::kYolo26:
      return std::make_unique<Yolo26DispatchAdapter>(labels);
  }
  throw std::invalid_argument("unknown detector family");
}

}  // namespace rkvs
