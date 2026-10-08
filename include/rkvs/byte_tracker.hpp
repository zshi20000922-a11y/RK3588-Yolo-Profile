#pragma once

#include "rkvs/config.hpp"
#include "rkvs/types.hpp"

#include <cstdint>
#include <vector>

namespace rkvs {

class ByteTracker {
 public:
  explicit ByteTracker(TrackerConfig config = {});
  void update_config(const TrackerConfig& config);
  std::vector<Detection> update(const std::vector<Detection>& detections,
                                uint64_t timestamp_ns);
  void reset();

 private:
  struct Track {
    int64_t id = -1;
    int class_id = -1;
    std::string label;
    Box box;
    Box previous_box;
    float score = 0;
    int age = 0;
    int hits = 0;
    int missed = 0;
    uint64_t timestamp_ns = 0;
  };

  static float iou(const Box& first, const Box& second);
  static Box predict(const Track& track, uint64_t timestamp_ns);
  void associate(const std::vector<Detection>& detections,
                 const std::vector<size_t>& candidates,
                 std::vector<bool>* detection_used,
                 std::vector<bool>* track_used, uint64_t timestamp_ns);

  TrackerConfig config_;
  std::vector<Track> tracks_;
  int64_t next_id_ = 1;
};

}  // namespace rkvs
