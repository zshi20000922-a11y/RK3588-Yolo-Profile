#include "rkvs/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace rkvs {
namespace {

struct Line {
  int number = 0;
  int indent = 0;
  bool list_item = false;
  std::string key;
  std::string value;
};

std::string trim(std::string value) {
  auto space = [](unsigned char c) { return std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), [&](char c) {
                return !space(static_cast<unsigned char>(c));
              }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [&](char c) {
                return !space(static_cast<unsigned char>(c));
              }).base(),
              value.end());
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\'')))
    value = value.substr(1, value.size() - 2);
  return value;
}

std::vector<Line> read_lines(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open config: " + path);
  std::vector<Line> result;
  std::string text;
  int number = 0;
  while (std::getline(input, text)) {
    ++number;
    bool quote = false;
    char quote_char = 0;
    for (size_t i = 0; i < text.size(); ++i) {
      if ((text[i] == '\'' || text[i] == '"') &&
          (i == 0 || text[i - 1] != '\\')) {
        if (!quote) {
          quote = true;
          quote_char = text[i];
        } else if (quote_char == text[i]) {
          quote = false;
        }
      }
      if (text[i] == '#' && !quote) {
        text.resize(i);
        break;
      }
    }
    if (trim(text).empty()) continue;
    if (text.find('\t') != std::string::npos)
      throw std::runtime_error("tabs are not allowed at config line " +
                               std::to_string(number));
    int indent = 0;
    while (indent < static_cast<int>(text.size()) && text[indent] == ' ')
      ++indent;
    if (indent % 2)
      throw std::runtime_error("indentation must use two spaces at line " +
                               std::to_string(number));
    std::string body = trim(text.substr(indent));
    bool list_item = body.rfind("- ", 0) == 0;
    if (list_item) body = trim(body.substr(2));
    auto colon = body.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("expected key:value at config line " +
                               std::to_string(number));
    result.push_back({number, indent, list_item, trim(body.substr(0, colon)),
                      trim(body.substr(colon + 1))});
  }
  return result;
}

bool boolean(const Line& line) {
  if (line.value == "true" || line.value == "yes" || line.value == "on")
    return true;
  if (line.value == "false" || line.value == "no" || line.value == "off")
    return false;
  throw std::runtime_error("invalid boolean at config line " +
                           std::to_string(line.number));
}

int integer(const Line& line) {
  size_t used = 0;
  int result = 0;
  try {
    result = std::stoi(line.value, &used);
  } catch (...) {
    throw std::runtime_error("invalid integer at config line " +
                             std::to_string(line.number));
  }
  if (used != line.value.size())
    throw std::runtime_error("invalid integer at config line " +
                             std::to_string(line.number));
  return result;
}

float number(const Line& line) {
  size_t used = 0;
  float result = 0;
  try {
    result = std::stof(line.value, &used);
  } catch (...) {
    throw std::runtime_error("invalid number at config line " +
                             std::to_string(line.number));
  }
  if (used != line.value.size())
    throw std::runtime_error("invalid number at config line " +
                             std::to_string(line.number));
  return result;
}

std::vector<std::string> string_list(const Line& line) {
  if (line.value.size() < 2 || line.value.front() != '[' ||
      line.value.back() != ']')
    throw std::runtime_error("expected [a,b] at config line " +
                             std::to_string(line.number));
  std::vector<std::string> result;
  std::stringstream stream(line.value.substr(1, line.value.size() - 2));
  std::string item;
  while (std::getline(stream, item, ',')) {
    item = trim(item);
    if (!item.empty()) result.push_back(item);
  }
  return result;
}

std::vector<int> int_list(const Line& line) {
  std::vector<int> result;
  for (const auto& value : string_list(line)) {
    Line item = line;
    item.value = value;
    result.push_back(integer(item));
  }
  return result;
}

[[noreturn]] void unknown(const std::string& section, const Line& line) {
  throw std::runtime_error("unknown " + section + " key '" + line.key +
                           "' at config line " +
                           std::to_string(line.number));
}

template <typename T>
void range(const char* name, T value, T minimum, T maximum) {
  if (value < minimum || value > maximum)
    throw std::invalid_argument(std::string(name) + " outside [" +
                                std::to_string(minimum) + "," +
                                std::to_string(maximum) + "]");
}

}  // namespace

const char* to_string(SourceType value) {
  return value == SourceType::kV4l2 ? "v4l2" : "rtsp";
}

const char* to_string(ModelFamily value) {
  switch (value) {
    case ModelFamily::kYolov5: return "yolov5";
    case ModelFamily::kYolov8: return "yolov8";
    case ModelFamily::kYolo11: return "yolo11";
    case ModelFamily::kYolo26: return "yolo26";
  }
  return "unknown";
}

void ServiceConfig::validate() const {
  if (version != 1) throw std::invalid_argument("unsupported config version");
  range("rtsp_port", rtsp_port, 1, 65535);
  if (sources.empty() || sources.size() > 10)
    throw std::invalid_argument("sources must contain 1..10 entries");
  std::set<std::string> ids;
  int local = 0, network = 0;
  for (const auto& source : sources) {
    if (source.id.empty() || source.uri.empty())
      throw std::invalid_argument("source id and uri are required");
    if (!ids.insert(source.id).second)
      throw std::invalid_argument("duplicate source id: " + source.id);
    source.type == SourceType::kV4l2 ? ++local : ++network;
    range("source fps", source.fps, 1, 240);
    range("source detect_fps", source.detect_fps, 1, 240);
    range("source buffers", source.buffers, 3, 32);
    if (source.type == SourceType::kV4l2 &&
        (source.width < 16 || source.height < 16))
      throw std::invalid_argument("V4L2 source requires width and height");
  }
  if (local > 2 || network > 8)
    throw std::invalid_argument("maximum is 2 V4L2 plus 8 RTSP sources");
  if (model.path.empty() || model.manifest.empty() || model.labels.empty())
    throw std::invalid_argument("model path, manifest and labels are required");
  range("contexts", model.contexts, 1, 3);
  range("confidence", model.confidence, 0.001f, 1.0f);
  range("nms_iou", model.nms_iou, 0.001f, 1.0f);
  range("max_detections", model.max_detections, 1, 1000);
  range("tracker high_threshold", tracker.high_threshold, 0.0f, 1.0f);
  range("tracker low_threshold", tracker.low_threshold, 0.0f,
        tracker.high_threshold);
  range("tracker match_iou", tracker.match_iou, 0.0f, 1.0f);
  range("tracker track_buffer", tracker.track_buffer, 1, 1000);
  range("motion width", motion.width, 32, 8192);
  range("motion height", motion.height, 32, 8192);
  range("motion fps", motion.fps, 1, 120);
  range("motion pixel_threshold", motion.pixel_threshold, 1, 255);
  range("motion active_ratio", motion.active_ratio, 0.0f, 1.0f);
  range("motion global_change_ratio", motion.global_change_ratio, 0.0f, 1.0f);
  if (motion.algorithm != "three_frame" && motion.algorithm != "mog2")
    throw std::invalid_argument("motion algorithm must be three_frame|mog2");
  range("motion mog2_history", motion.mog2_history, 2, 10000);
  range("motion mog2_var_threshold", motion.mog2_var_threshold, 0.1f, 1000.0f);
  if (display.sources.size() > 9)
    throw std::invalid_argument("display accepts at most 9 source ids");
  for (const auto& id : display.sources)
    if (!ids.count(id)) throw std::invalid_argument("unknown display source: " + id);
  if (outputs.size() > 3)
    throw std::invalid_argument("maximum is mosaic plus two focus outputs");
  std::set<std::string> output_ids;
  for (const auto& output : outputs) {
    if (!output_ids.insert(output.id).second)
      throw std::invalid_argument("duplicate output id: " + output.id);
    if (output.kind != "mosaic" && output.kind != "focus")
      throw std::invalid_argument("output kind must be mosaic or focus");
    if (output.kind == "focus" && !ids.count(output.source))
      throw std::invalid_argument("focus output references unknown source");
    if (output.codec != "h264" && output.codec != "h265")
      throw std::invalid_argument("output codec must be h264 or h265");
    if (output.rtsp_enabled && (output.rtsp_path.empty() || output.rtsp_path[0] != '/'))
      throw std::invalid_argument("RTSP path must start with /");
    if (output.udp_enabled && (output.udp_host.empty() || output.udp_port < 1))
      throw std::invalid_argument("UDP output requires host and port");
  }
}

std::string ServiceConfig::topology_fingerprint() const {
  std::ostringstream out;
  out << version << '|' << rtsp_port << '|'
      << model.path << '|' << model.manifest << '|' << model.labels << '|'
      << to_string(model.family) << '|' << model.contexts << '|'
      << display.enabled << ':' << display.backend << ':' << display.width
      << 'x' << display.height << '@' << display.fps << ':';
  for (const auto& id : display.sources) out << id << ',';
  out << '|';
  for (const auto& source : sources)
    out << source.id << ':' << to_string(source.type) << ':' << source.uri << ':'
        << source.enabled << ':' << source.width << 'x' << source.height << '@'
        << source.fps << ':' << source.buffers << '|';
  for (const auto& output : outputs)
    out << output.id << ':' << output.kind << ':' << output.source << ':'
        << output.codec << ':' << output.width << 'x' << output.height << '@'
        << output.fps << ':' << output.udp_enabled << ':' << output.udp_host
        << ':' << output.udp_port << ':' << output.rtsp_enabled << ':'
        << output.rtsp_path << '|';
  return out.str();
}

ServiceConfig ConfigLoader::load(const std::string& path) {
  ServiceConfig config;
  std::string section;
  SourceConfig* source = nullptr;
  StreamOutputConfig* output = nullptr;
  for (const Line& line : read_lines(path)) {
    if (line.indent == 0) {
      source = nullptr;
      output = nullptr;
      if (line.key == "version") {
        config.version = integer(line);
        section.clear();
      } else if (line.value.empty() &&
                 (line.key == "service" || line.key == "model" ||
                  line.key == "tracker" || line.key == "motion" ||
                  line.key == "display" || line.key == "sources" ||
                  line.key == "outputs")) {
        section = line.key;
      } else {
        unknown("top-level", line);
      }
      continue;
    }
    if (section == "sources" || section == "outputs") {
      if (line.indent == 2 && line.list_item) {
        if (line.key != "id") unknown(section, line);
        if (section == "sources") {
          config.sources.emplace_back();
          source = &config.sources.back();
          source->id = line.value;
        } else {
          config.outputs.emplace_back();
          output = &config.outputs.back();
          output->id = line.value;
        }
        continue;
      }
      if (line.indent != 4 || line.list_item || (!source && !output))
        throw std::runtime_error("invalid list structure at config line " +
                                 std::to_string(line.number));
    } else if (line.indent != 2 || line.list_item) {
      throw std::runtime_error("invalid section indentation at config line " +
                               std::to_string(line.number));
    }

    if (section == "service") {
      if (line.key == "results_socket") config.results_socket = line.value;
      else if (line.key == "control_socket") config.control_socket = line.value;
      else if (line.key == "log_dir") config.log_dir = line.value;
      else if (line.key == "stale_result_ms") config.stale_result_ms = integer(line);
      else if (line.key == "rtsp_port") config.rtsp_port = integer(line);
      else unknown(section, line);
    } else if (section == "model") {
      if (line.key == "path") config.model.path = line.value;
      else if (line.key == "manifest") config.model.manifest = line.value;
      else if (line.key == "labels") config.model.labels = line.value;
      else if (line.key == "family") {
        if (line.value == "yolov5") config.model.family = ModelFamily::kYolov5;
        else if (line.value == "yolov8") config.model.family = ModelFamily::kYolov8;
        else if (line.value == "yolo11") config.model.family = ModelFamily::kYolo11;
        else if (line.value == "yolo26") config.model.family = ModelFamily::kYolo26;
        else unknown(section, line);
      } else if (line.key == "contexts") config.model.contexts = integer(line);
      else if (line.key == "auto_select_cores") config.model.auto_select_cores = boolean(line);
      else if (line.key == "confidence") config.model.confidence = number(line);
      else if (line.key == "nms_iou") config.model.nms_iou = number(line);
      else if (line.key == "max_detections") config.model.max_detections = integer(line);
      else if (line.key == "class_agnostic_nms") config.model.class_agnostic_nms = boolean(line);
      else if (line.key == "class_filter") config.model.class_filter = int_list(line);
      else unknown(section, line);
    } else if (section == "tracker") {
      if (line.key == "enabled") config.tracker.enabled = boolean(line);
      else if (line.key == "high_threshold") config.tracker.high_threshold = number(line);
      else if (line.key == "low_threshold") config.tracker.low_threshold = number(line);
      else if (line.key == "match_iou") config.tracker.match_iou = number(line);
      else if (line.key == "track_buffer") config.tracker.track_buffer = integer(line);
      else unknown(section, line);
    } else if (section == "motion") {
      if (line.key == "enabled") config.motion.enabled = boolean(line);
      else if (line.key == "algorithm") config.motion.algorithm = line.value;
      else if (line.key == "direct_input") config.motion.direct_input = boolean(line);
      else if (line.key == "width") config.motion.width = integer(line);
      else if (line.key == "height") config.motion.height = integer(line);
      else if (line.key == "fps") config.motion.fps = integer(line);
      else if (line.key == "pixel_threshold") config.motion.pixel_threshold = integer(line);
      else if (line.key == "min_region_pixels") config.motion.min_region_pixels = integer(line);
      else if (line.key == "active_ratio") config.motion.active_ratio = number(line);
      else if (line.key == "global_change_ratio") config.motion.global_change_ratio = number(line);
      else if (line.key == "gate_detection") config.motion.gate_detection = boolean(line);
      else if (line.key == "idle_detect_fps") config.motion.idle_detect_fps = integer(line);
      else if (line.key == "mog2_history") config.motion.mog2_history = integer(line);
      else if (line.key == "mog2_var_threshold") config.motion.mog2_var_threshold = number(line);
      else if (line.key == "mog2_detect_shadows") config.motion.mog2_detect_shadows = boolean(line);
      else unknown(section, line);
    } else if (section == "display") {
      if (line.key == "enabled") config.display.enabled = boolean(line);
      else if (line.key == "backend") config.display.backend = line.value;
      else if (line.key == "width") config.display.width = integer(line);
      else if (line.key == "height") config.display.height = integer(line);
      else if (line.key == "fps") config.display.fps = integer(line);
      else if (line.key == "sources") config.display.sources = string_list(line);
      else unknown(section, line);
    } else if (section == "sources") {
      if (line.key == "type") {
        if (line.value == "v4l2") source->type = SourceType::kV4l2;
        else if (line.value == "rtsp") source->type = SourceType::kRtsp;
        else unknown(section, line);
      } else if (line.key == "uri") source->uri = line.value;
      else if (line.key == "enabled") source->enabled = boolean(line);
      else if (line.key == "width") source->width = integer(line);
      else if (line.key == "height") source->height = integer(line);
      else if (line.key == "fps") source->fps = integer(line);
      else if (line.key == "buffers") source->buffers = integer(line);
      else if (line.key == "detect_fps") source->detect_fps = integer(line);
      else if (line.key == "transport") {
        if (line.value == "udp") source->transport = RtspTransport::kUdp;
        else if (line.value == "tcp") source->transport = RtspTransport::kTcp;
        else unknown(section, line);
      } else unknown(section, line);
    } else if (section == "outputs") {
      if (line.key == "kind") output->kind = line.value;
      else if (line.key == "source") output->source = line.value;
      else if (line.key == "codec") output->codec = line.value;
      else if (line.key == "width") output->width = integer(line);
      else if (line.key == "height") output->height = integer(line);
      else if (line.key == "fps") output->fps = integer(line);
      else if (line.key == "bitrate") output->bitrate = integer(line);
      else if (line.key == "gop") output->gop = integer(line);
      else if (line.key == "udp_enabled") output->udp_enabled = boolean(line);
      else if (line.key == "udp_host") output->udp_host = line.value;
      else if (line.key == "udp_port") output->udp_port = integer(line);
      else if (line.key == "rtsp_enabled") output->rtsp_enabled = boolean(line);
      else if (line.key == "rtsp_path") output->rtsp_path = line.value;
      else unknown(section, line);
    } else {
      throw std::runtime_error("value outside section at config line " +
                               std::to_string(line.number));
    }
  }
  config.validate();
  return config;
}

}  // namespace rkvs
