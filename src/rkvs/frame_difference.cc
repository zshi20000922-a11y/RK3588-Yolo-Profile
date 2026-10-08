#include "rkvs/frame_difference.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rkvs {

ThreeFrameDifference::ThreeFrameDifference(MotionConfig config)
    : config_(std::move(config)) {}

void ThreeFrameDifference::update_config(const MotionConfig& config) {
  if (config.width != config_.width || config.height != config_.height) reset();
  config_ = config;
}

MotionResult ThreeFrameDifference::update(const uint8_t* gray, int width,
                                           int height, int stride) {
  if (!config_.enabled) return {};
  if (!gray || width != config_.width || height != config_.height || stride < width)
    throw std::invalid_argument("frame difference input geometry mismatch");
  const size_t pixels = static_cast<size_t>(width) * height;
  current_.resize(pixels);
  for (int row = 0; row < height; ++row)
    std::copy_n(gray + static_cast<size_t>(row) * stride, width,
                current_.data() + static_cast<size_t>(row) * width);
  if (previous2_.empty() || previous1_.empty()) {
    previous2_ = std::move(previous1_);
    previous1_ = current_;
    return {};
  }
  delta1_.resize(pixels); delta2_.resize(pixels); mask_.resize(pixels); opened_.resize(pixels);
  cv::Mat current(height, width, CV_8UC1, current_.data());
  cv::Mat previous1(height, width, CV_8UC1, previous1_.data());
  cv::Mat previous2(height, width, CV_8UC1, previous2_.data());
  cv::Mat delta1(height, width, CV_8UC1, delta1_.data());
  cv::Mat delta2(height, width, CV_8UC1, delta2_.data());
  cv::Mat mask(height, width, CV_8UC1, mask_.data());
  cv::Mat opened(height, width, CV_8UC1, opened_.data());
  cv::absdiff(current, previous1, delta1);
  cv::absdiff(previous1, previous2, delta2);
  cv::threshold(delta1, delta1, config_.pixel_threshold - 1, 255, cv::THRESH_BINARY);
  cv::threshold(delta2, delta2, config_.pixel_threshold - 1, 255, cv::THRESH_BINARY);
  cv::bitwise_and(delta1, delta2, mask);
  const uint64_t changed = cv::countNonZero(mask);
  previous2_.swap(previous1_);
  previous1_.swap(current_);
  const float raw_ratio = static_cast<float>(changed) / pixels;
  if (raw_ratio >= config_.global_change_ratio)
    return {false, true, raw_ratio, {}};
  cv::morphologyEx(mask, opened, cv::MORPH_OPEN,
                   cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
  return regions_from_mask(opened_, width, height);
}

MotionResult ThreeFrameDifference::regions_from_mask(
    const std::vector<uint8_t>& mask, int width, int height) const {
  MotionResult result;
  cv::Mat binary(height, width, CV_8UC1, const_cast<uint8_t*>(mask.data()));
  cv::Mat labels, stats, centroids;
  const int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 4, CV_32S);
  uint64_t active_pixels = cv::countNonZero(binary);
  for (int label = 1; label < count; ++label) {
    const int pixels = stats.at<int>(label, cv::CC_STAT_AREA);
    if (pixels < config_.min_region_pixels) continue;
    const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
    const int top = stats.at<int>(label, cv::CC_STAT_TOP);
    const int box_width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    const int box_height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    result.regions.push_back({{static_cast<float>(left), static_cast<float>(top),
                               static_cast<float>(left + box_width),
                               static_cast<float>(top + box_height)},
                              static_cast<uint32_t>(pixels)});
  }
  result.ratio = static_cast<float>(active_pixels) / mask.size();
  result.active = result.ratio >= config_.active_ratio && !result.regions.empty();
  return result;
}

void ThreeFrameDifference::reset() {
  previous2_.clear();
  previous1_.clear();
  current_.clear(); delta1_.clear(); delta2_.clear(); mask_.clear(); opened_.clear();
}

}  // namespace rkvs
