#include "rkvs/byte_tracker.hpp"
#include "rkvs/config.hpp"
#include "rkvs/detector_adapter.hpp"
#include "rkvs/fair_scheduler.hpp"
#include "rkvs/frame_difference.hpp"
#include "rkvs/frame_hub.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <stdexcept>
#include <vector>

#ifndef RKVS_TEST_CONFIG
#define RKVS_TEST_CONFIG "config/rk_vision_service.example.yaml"
#endif

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

rkvs::Detection person(float left, float score = 0.9f) {
  rkvs::Detection detection;
  detection.class_id = 0;
  detection.label = "person";
  detection.score = score;
  detection.box = {left, 10, left + 20, 50};
  return detection;
}

void test_config() {
  auto config = rkvs::ConfigLoader::load(RKVS_TEST_CONFIG);
  require(config.sources.size() == 2, "config source count");
  require(config.sources[0].type == rkvs::SourceType::kV4l2,
          "config V4L2 type");
  require(config.model.family == rkvs::ModelFamily::kYolov5,
          "config model family");
  require(config.outputs.size() == 1, "config output count");
  require(!config.topology_fingerprint().empty(), "topology fingerprint");
  auto changed_fps = config;
  changed_fps.sources[0].detect_fps = std::max(1, config.sources[0].detect_fps / 2);
  require(changed_fps.topology_fingerprint() == config.topology_fingerprint(),
          "detect_fps must be hot-reloadable");
}

void test_frame_hub() {
  rkvs::LatestFrameHub hub;
  auto first_consumer = hub.subscribe("detector");
  auto second_consumer = hub.subscribe("display");
  std::atomic<int> releases{0};
  auto make_frame = [&](uint64_t sequence) {
    auto frame = std::make_shared<rkvs::FrameRef>();
    frame->source_id = "cam0";
    frame->sequence = sequence;
    frame->lease = std::make_shared<rkvs::BufferLease>([&] { ++releases; });
    return frame;
  };
  hub.publish(make_frame(1));
  hub.publish(make_frame(2));
  auto a = first_consumer.take("cam0");
  auto b = second_consumer.take("cam0");
  require(a && b && a->sequence == 2 && b->sequence == 2,
          "consumers must receive newest frame");
  require(hub.overwritten("cam0") == 1, "hub overwrite metric");
  a.reset();
  b.reset();
  hub.close();
  require(releases == 2, "frame leases must release exactly once");
}

void test_tracker() {
  rkvs::TrackerConfig config;
  config.high_threshold = 0.5f;
  config.low_threshold = 0.1f;
  config.match_iou = 0.2f;
  config.track_buffer = 2;
  rkvs::ByteTracker tracker(config);
  auto first = tracker.update({person(10)}, 1000000000ULL);
  auto second = tracker.update({person(12)}, 1033333333ULL);
  require(first.size() == 1 && second.size() == 1, "tracker output count");
  require(first[0].track_id == second[0].track_id, "tracker stable id");
  require(second[0].track_state == "confirmed", "tracker confirmation");
  auto predicted = tracker.update({}, 1066666666ULL);
  require(predicted.size() == 1 && predicted[0].predicted,
          "tracker prediction during short loss");
  tracker.update({}, 1099999999ULL);
  require(tracker.update({}, 1133333332ULL).empty(), "tracker expiry");
}

void fill_block(std::vector<uint8_t>* image, int width, int x0) {
  for (int y = 12; y < 28; ++y)
    for (int x = x0; x < x0 + 12; ++x)
      (*image)[static_cast<size_t>(y) * width + x] = 220;
}

void test_motion() {
  rkvs::MotionConfig config;
  config.width = 64;
  config.height = 48;
  config.pixel_threshold = 20;
  config.min_region_pixels = 20;
  config.active_ratio = 0.001f;
  config.global_change_ratio = 0.8f;
  rkvs::ThreeFrameDifference difference(config);
  std::vector<uint8_t> first(64 * 48), second(64 * 48), third(64 * 48);
  fill_block(&second, 64, 10);
  fill_block(&third, 64, 20);
  require(!difference.update(first.data(), 64, 48, 64).active,
          "motion warmup frame one");
  require(!difference.update(second.data(), 64, 48, 64).active,
          "motion warmup frame two");
  auto motion = difference.update(third.data(), 64, 48, 64);
  require(motion.active && !motion.regions.empty(), "three-frame motion");
}

void test_adapter_validation_and_yolo26() {
  std::vector<std::string> labels{"person", "car"};
  rkvs::DetectorManifest manifest;
  manifest.family = rkvs::ModelFamily::kYolo26;
  manifest.input_width = 640;
  manifest.input_height = 640;
  rkvs::ModelSchema schema;
  schema.input_width = 640;
  schema.input_height = 640;
  schema.input_channels = 3;
  for (int branch = 0; branch < 3; ++branch) {
    schema.outputs.push_back({"box", rkvs::TensorDataType::kFloat32,
                              rkvs::TensorLayout::kNchw, {1, 4, 1, 1}, 0, 1});
    schema.outputs.push_back({"class", rkvs::TensorDataType::kFloat32,
                              rkvs::TensorLayout::kNchw, {1, 2, 1, 1}, 0, 1});
  }
  auto adapter = rkvs::make_detector_adapter(manifest.family, labels);
  adapter->validate(schema, manifest);
  const float box[] = {0.5f, 0.5f, 0.5f, 0.5f};
  const float score[] = {0.9f, 0.1f};
  std::vector<rkvs::TensorView> views;
  for (int branch = 0; branch < 3; ++branch) {
    views.push_back({schema.outputs[branch * 2], box, sizeof(box)});
    views.push_back({schema.outputs[branch * 2 + 1], score, sizeof(score)});
  }
  rkvs::DetectorParams params;
  params.confidence = 0.25f;
  auto transform = rkvs::LetterboxTransform::make(640, 640, 640, 640);
  auto detections = adapter->decode(views, transform, params);
  require(detections.size() == 1 && detections[0].class_id == 0,
          "YOLO26 one-to-many raw heads must apply class-wise NMS");

  auto v5 = rkvs::make_detector_adapter(rkvs::ModelFamily::kYolov5, labels);
  bool rejected = false;
  try { v5->validate(schema, manifest); } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "model family must reject incompatible tensor protocol");
}

void test_yolo26_nc1hwc2_argmax() {
  std::vector<std::string> labels;
  for (int i = 0; i < 16; ++i) labels.push_back("class" + std::to_string(i));
  rkvs::ModelSchema schema;
  schema.input_width = 640;
  schema.input_height = 640;
  schema.input_channels = 3;
  std::vector<rkvs::TensorView> outputs;
  std::vector<int8_t> boxes(16, 0);
  for (int i = 0; i < 4; ++i) boxes[i] = 1;
  std::vector<int8_t> scores(16, -10);
  scores[15] = 10;
  for (int branch = 0; branch < 3; ++branch) {
    rkvs::TensorSchema box_schema{"box", rkvs::TensorDataType::kInt8,
        rkvs::TensorLayout::kNc1hwc2, {1, 1, 1, 1, 16}, 0, 1};
    rkvs::TensorSchema score_schema{"class", rkvs::TensorDataType::kInt8,
        rkvs::TensorLayout::kNc1hwc2, {1, 1, 1, 1, 16}, 0, 1};
    schema.outputs.push_back(box_schema);
    schema.outputs.push_back(score_schema);
    outputs.push_back({box_schema, boxes.data(), boxes.size()});
    outputs.push_back({score_schema, scores.data(), scores.size()});
  }
  auto adapter = rkvs::make_detector_adapter(rkvs::ModelFamily::kYolo26, labels);
  rkvs::DetectorManifest manifest;
  manifest.family = rkvs::ModelFamily::kYolo26;
  manifest.input_width = manifest.input_height = 640;
  adapter->validate(schema, manifest);
  rkvs::DetectorParams params;
  params.confidence = 0.5f;
  const auto detections = adapter->decode(
      outputs, rkvs::LetterboxTransform::make(640, 640, 640, 640), params);
  require(detections.size() == 1 && detections.front().class_id == 15,
          "YOLO26 NC1HWC2 INT8 argmax must return the correct class");
}

void test_yolo26_quantized_threshold_gates() {
  std::vector<std::string> labels;
  for (int i = 0; i < 16; ++i) labels.push_back("class" + std::to_string(i));
  rkvs::ModelSchema schema;
  schema.input_width = schema.input_height = 640;
  schema.input_channels = 3;
  std::vector<std::vector<int8_t>> boxes(3, std::vector<int8_t>(16, 0));
  std::vector<std::vector<int8_t>> scores(3, std::vector<int8_t>(16, 0));
  std::vector<std::vector<int8_t>> sums(3, std::vector<int8_t>(16, 0));
  std::vector<rkvs::TensorView> outputs;
  for (int branch = 0; branch < 3; ++branch) {
    for (int i = 0; i < 4; ++i) boxes[branch][i] = 1;
    // At branch 0, score-sum fails; at branch 1, class score fails; branch 2
    // lies exactly on the confidence boundary and must survive both gates.
    scores[branch][15] = branch == 1 ? 0 : 1;
    sums[branch][0] = branch == 0 ? 0 : 1;
    rkvs::TensorSchema box_schema{"box", rkvs::TensorDataType::kInt8,
        rkvs::TensorLayout::kNc1hwc2, {1, 1, 1, 1, 16}, 0, 1.0f};
    rkvs::TensorSchema score_schema{"class", rkvs::TensorDataType::kInt8,
        rkvs::TensorLayout::kNc1hwc2, {1, 1, 1, 1, 16}, 0, 0.25f};
    rkvs::TensorSchema sum_schema{"score_sum", rkvs::TensorDataType::kInt8,
        rkvs::TensorLayout::kNc1hwc2, {1, 1, 1, 1, 16}, 0, 0.25f};
    schema.outputs.insert(schema.outputs.end(),
        {box_schema, score_schema, sum_schema});
    outputs.push_back({box_schema, boxes[branch].data(), boxes[branch].size()});
    outputs.push_back({score_schema, scores[branch].data(), scores[branch].size()});
    outputs.push_back({sum_schema, sums[branch].data(), sums[branch].size()});
  }
  auto adapter = rkvs::make_detector_adapter(rkvs::ModelFamily::kYolo26, labels);
  rkvs::DetectorManifest manifest;
  manifest.family = rkvs::ModelFamily::kYolo26;
  manifest.input_width = manifest.input_height = 640;
  adapter->validate(schema, manifest);
  rkvs::DetectorParams params;
  params.confidence = 0.25f;
  const auto detections = adapter->decode(
      outputs, rkvs::LetterboxTransform::make(640, 640, 640, 640), params);
  require(detections.size() == 1 && detections.front().class_id == 15,
          "YOLO26 quantized score gates must preserve threshold-boundary semantics");
}

void test_fair_scheduler_latest_only() {
  std::mutex result_mutex;
  std::vector<std::string> order;
  rkvs::LatestFairScheduler scheduler(
      {{"cam0", 200}, {"cam1", 200}}, 1,
      [](rkvs::SharedFrame frame, int) {
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
        rkvs::ResultBatch result;
        result.source_id = frame->source_id;
        result.sequence = frame->sequence;
        return result;
      },
      [&](rkvs::ResultBatch result) {
        std::lock_guard<std::mutex> lock(result_mutex);
        order.push_back(result.source_id + ":" + std::to_string(result.sequence));
      });
  scheduler.start();
  auto submit = [&](const char* id, uint64_t sequence) {
    auto frame = std::make_shared<rkvs::FrameRef>();
    frame->source_id = id;
    frame->sequence = sequence;
    scheduler.submit(frame);
  };
  submit("cam0", 1);
  submit("cam0", 2);
  submit("cam1", 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  scheduler.stop();
  auto stats = scheduler.stats();
  require(stats["cam0"].replaced >= 1, "scheduler replaces stale pending frame");
  require(stats["cam0"].completed >= 1 && stats["cam1"].completed >= 1,
          "scheduler fair source service");
  require(stats["cam0"].queue_depth == 0 && stats["cam1"].queue_depth == 0,
          "scheduler queue bounded to latest frame");
}

void test_scheduler_update_sources() {
  std::atomic<int> completed{0};
  rkvs::LatestFairScheduler scheduler(
      {{"cam0", 1}}, 1,
      [](rkvs::SharedFrame frame, int) {
        rkvs::ResultBatch result;
        result.source_id = frame->source_id;
        result.sequence = frame->sequence;
        return result;
      },
      [&](rkvs::ResultBatch) { ++completed; });
  scheduler.start();
  auto submit = [&] {
    auto frame = std::make_shared<rkvs::FrameRef>();
    frame->source_id = "cam0";
    frame->capture_ts_ns = rkvs::monotonic_ns();
    frame->receive_ts_ns = frame->capture_ts_ns;
    scheduler.submit(frame);
  };
  submit();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  scheduler.update_sources({{"cam0", 200}});
  submit();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  scheduler.stop();
  require(completed >= 2, "scheduler target fps hot update");
}

}  // namespace

int main() {
  try {
    test_config();
    test_frame_hub();
    test_tracker();
    test_motion();
  test_adapter_validation_and_yolo26();
  test_yolo26_nc1hwc2_argmax();
  test_yolo26_quantized_threshold_gates();
    test_fair_scheduler_latest_only();
    test_scheduler_update_sources();
    std::cout << "rkvs_core_tests: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << "rkvs_core_tests: FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
