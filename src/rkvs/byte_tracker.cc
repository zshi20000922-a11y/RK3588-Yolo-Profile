#include "rkvs/byte_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace rkvs {

ByteTracker::ByteTracker(TrackerConfig config) : config_(std::move(config)) {}

void ByteTracker::update_config(const TrackerConfig& config) {
  config_ = config;
}

float ByteTracker::iou(const Box& a, const Box& b) {
  const float left = std::max(a.left, b.left);
  const float top = std::max(a.top, b.top);
  const float right = std::min(a.right, b.right);
  const float bottom = std::min(a.bottom, b.bottom);
  const float intersection = std::max(0.0f, right - left) *
                             std::max(0.0f, bottom - top);
  const float total = a.area() + b.area() - intersection;
  return total > 0 ? intersection / total : 0;
}

Box ByteTracker::predict(const Track& track, uint64_t timestamp_ns) {
  if (track.timestamp_ns == 0 || timestamp_ns <= track.timestamp_ns)
    return track.box;
  const float steps = std::clamp(
      static_cast<float>(timestamp_ns - track.timestamp_ns) / 33333333.0f,
      0.0f, 3.0f);
  Box result;
  result.left = track.box.left + (track.box.left - track.previous_box.left) * steps;
  result.top = track.box.top + (track.box.top - track.previous_box.top) * steps;
  result.right = track.box.right + (track.box.right - track.previous_box.right) * steps;
  result.bottom = track.box.bottom + (track.box.bottom - track.previous_box.bottom) * steps;
  return result;
}

void ByteTracker::associate(const std::vector<Detection>& detections,
                            const std::vector<size_t>& candidates,
                            std::vector<bool>* detection_used,
                            std::vector<bool>* track_used,
                            uint64_t timestamp_ns) {
  struct Match { float overlap; size_t track; size_t detection; };
  std::vector<Match> matches;
  for (size_t track = 0; track < tracks_.size(); ++track) {
    if ((*track_used)[track]) continue;
    Box predicted = predict(tracks_[track], timestamp_ns);
    for (size_t detection : candidates) {
      if ((*detection_used)[detection] ||
          detections[detection].class_id != tracks_[track].class_id)
        continue;
      float overlap = iou(predicted, detections[detection].box);
      if (overlap >= config_.match_iou)
        matches.push_back({overlap, track, detection});
    }
  }
  std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
    return a.overlap > b.overlap;
  });
  for (const auto& match : matches) {
    if ((*track_used)[match.track] || (*detection_used)[match.detection]) continue;
    Track& track = tracks_[match.track];
    track.previous_box = track.box;
    track.box = detections[match.detection].box;
    track.score = detections[match.detection].score;
    track.timestamp_ns = timestamp_ns;
    ++track.age;
    ++track.hits;
    track.missed = 0;
    (*track_used)[match.track] = true;
    (*detection_used)[match.detection] = true;
  }
}

std::vector<Detection> ByteTracker::update(
    const std::vector<Detection>& detections, uint64_t timestamp_ns) {
  if (!config_.enabled) return detections;
  std::vector<size_t> high, low;
  for (size_t i = 0; i < detections.size(); ++i) {
    if (detections[i].score >= config_.high_threshold) high.push_back(i);
    else if (detections[i].score >= config_.low_threshold) low.push_back(i);
  }
  std::vector<bool> detection_used(detections.size(), false);
  std::vector<bool> track_used(tracks_.size(), false);
  associate(detections, high, &detection_used, &track_used, timestamp_ns);
  associate(detections, low, &detection_used, &track_used, timestamp_ns);

  for (size_t i = 0; i < tracks_.size(); ++i) {
    if (!track_used[i]) {
      ++tracks_[i].age;
      ++tracks_[i].missed;
      tracks_[i].previous_box = tracks_[i].box;
      tracks_[i].box = predict(tracks_[i], timestamp_ns);
      tracks_[i].timestamp_ns = timestamp_ns;
    }
  }
  for (size_t index : high) {
    if (detection_used[index]) continue;
    Track track;
    track.id = next_id_++;
    track.class_id = detections[index].class_id;
    track.label = detections[index].label;
    track.box = track.previous_box = detections[index].box;
    track.score = detections[index].score;
    track.age = track.hits = 1;
    track.timestamp_ns = timestamp_ns;
    tracks_.push_back(track);
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& track) {
                  return track.missed > config_.track_buffer;
                }),
                tracks_.end());

  std::vector<Detection> result;
  result.reserve(tracks_.size());
  for (const Track& track : tracks_) {
    Detection detection;
    detection.class_id = track.class_id;
    detection.label = track.label;
    detection.score = track.score;
    detection.box = track.box;
    detection.track_id = track.id;
    detection.predicted = track.missed > 0;
    detection.track_state = track.missed > 0 ? "predicted" :
                            track.hits >= 2 ? "confirmed" : "tentative";
    result.push_back(std::move(detection));
  }
  return result;
}

void ByteTracker::reset() {
  tracks_.clear();
  next_id_ = 1;
}

}  // namespace rkvs
