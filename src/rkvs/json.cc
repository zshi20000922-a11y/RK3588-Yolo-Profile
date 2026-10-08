#include "rkvs/json.hpp"
#include <iomanip>
#include <sstream>

namespace rkvs {
std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (unsigned char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20)
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(character) << std::dec;
        else output << character;
    }
  }
  return output.str();
}

std::string to_json(const ResultBatch& result) {
  std::ostringstream out;
  out << "{\"source_id\":\"" << json_escape(result.source_id)
      << "\",\"sequence\":" << result.sequence
      << ",\"capture_ts_ns\":" << result.capture_ts_ns
      << ",\"width\":" << result.width << ",\"height\":" << result.height
      << ",\"detections\":[";
  for (size_t index = 0; index < result.detections.size(); ++index) {
    const auto& d = result.detections[index];
    if (index) out << ',';
    out << "{\"class_id\":" << d.class_id << ",\"label\":\""
        << json_escape(d.label) << "\",\"score\":" << d.score
        << ",\"box\":[" << d.box.left << ',' << d.box.top << ','
        << d.box.right << ',' << d.box.bottom << ']'
        << ",\"track_id\":" << d.track_id << ",\"track_state\":\""
        << json_escape(d.track_state) << "\",\"predicted\":"
        << (d.predicted ? "true" : "false") << '}';
  }
  out << "],\"motion\":{\"active\":" << (result.motion.active ? "true" : "false")
      << ",\"global_change\":" << (result.motion.global_change ? "true" : "false")
      << ",\"ratio\":" << result.motion.ratio << ",\"regions\":[";
  for (size_t index = 0; index < result.motion.regions.size(); ++index) {
    if (index) out << ',';
    const auto& region = result.motion.regions[index];
    out << '[' << region.box.left << ',' << region.box.top << ','
        << region.box.right << ',' << region.box.bottom << ']';
  }
  const auto& t = result.timings;
  out << "]},\"timings_ms\":{\"decode\":" << t.decode_ms
      << ",\"queue\":" << t.queue_ms << ",\"rga\":" << t.rga_ms
      << ",\"inference\":" << t.inference_ms
      << ",\"output_sync\":" << t.output_sync_ms
      << ",\"postprocess\":" << t.postprocess_ms
      << ",\"tracking\":" << t.tracking_ms
      << ",\"capture_to_result\":" << t.capture_to_result_ms << "}}\n";
  return out.str();
}
}  // namespace rkvs
