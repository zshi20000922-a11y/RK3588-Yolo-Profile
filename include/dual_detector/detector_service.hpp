#pragma once

#include "rknn_worker.hpp"
#include "v4l2_camera.hpp"
#include "preview_sink.hpp"
#include "pip_sink.hpp"
#include "stereo_coordinator.hpp"
#include "auto_roi_coordinator.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dual {

struct CameraConfig {
  std::string id, device;
  int capture_width=640, capture_height=360;
  int logical_width=3840, logical_height=2160;
};
struct ServiceConfig {
  struct RoiBinding { std::string camera_id, sensor, cif, isp; };
  std::string model, labels;
  std::vector<CameraConfig> cameras;
  int contexts=6, detect_fps=60, buffers=8, max_inflight_per_camera=3;
  int preprocess_threads=2, postprocess_threads=2;
  bool independent_contexts=false;
  bool preview=false;
  std::string pip_target;
  int pip_udp_port=5600, pip_rtsp_feed_port=7000;
  std::string stereo_calibration;
  std::string roi_sensor_subdev, roi_cif_subdev, roi_isp_subdev;
  int stereo_class_id=0;
  bool stereo_project_fallback=false;
  int roi_capture_fps=166;
  int roi_lost_return_ms=300;
  std::vector<RoiBinding> auto_roi_bindings;
  std::string result_socket="/run/rknn-dual-detector/detections.sock";
  std::string control_socket="/run/rknn-dual-detector/control.sock";
};

class DetectorService {
 public:
  explicit DetectorService(ServiceConfig config);
  ~DetectorService();
  void run();
  void request_stop();
  void notify_stop() { running_=false; notify_pipeline(); }
 private:
  struct WorkerSlot;
  struct PreprocessTask;
  void scheduler_loop();
  void preprocess_loop(int index);
  void inference_loop(WorkerSlot* slot);
  void postprocess_loop(int index);
  void result_server_loop();
  void result_dispatch_loop();
  void control_server_loop();
  void stereo_control_loop();
  void auto_roi_control_loop();
  void publish(DetectionBatch batch);
  void enqueue_result(DetectionBatch batch);
  std::string status_json() const;
  void set_paused(bool paused);
  void notify_pipeline();
  void fail_job(int worker_id, int tensor_slot, size_t camera_index,
                const char* stage, const std::exception& error);

  ServiceConfig config_;
  std::vector<std::unique_ptr<V4L2Camera>> cameras_;
  std::vector<std::unique_ptr<WorkerSlot>> workers_;
  std::vector<std::atomic<int>> inflight_;
  std::vector<uint64_t> last_dispatch_ns_;
  std::vector<std::atomic<uint64_t>> last_preview_ns_;
  std::vector<std::atomic<uint64_t>> camera_published_, last_published_seq_;
  std::vector<std::atomic<uint64_t>> out_of_order_results_;
  std::atomic<bool> running_{true}, paused_{false};
  std::atomic<bool> preprocess_stopped_{false};
  std::atomic<bool> inference_stopped_{false};
  std::atomic<bool> postprocess_stopped_{false};
  std::thread scheduler_, result_server_, result_dispatch_, control_server_;
  std::thread stereo_control_;
  std::thread auto_roi_control_;
  std::vector<std::thread> preprocess_threads_, postprocess_threads_;
  mutable std::mutex state_mutex_, client_mutex_;
  std::condition_variable state_cv_;

  mutable std::mutex preprocess_mutex_, postprocess_mutex_;
  std::condition_variable preprocess_cv_, postprocess_cv_;
  std::deque<std::unique_ptr<PreprocessTask>> preprocess_queue_;
  std::deque<std::unique_ptr<PipelineJob>> postprocess_queue_;
  std::atomic<uint64_t> preprocess_queue_drops_{0}, pipeline_errors_{0};

  std::vector<int> result_clients_;
  std::mutex result_mutex_;
  std::condition_variable result_cv_;
  std::deque<DetectionBatch> pending_results_;
  int result_listen_=-1, control_listen_=-1;
  uint64_t published_=0, client_drops_=0;
  std::atomic<uint64_t> result_queue_drops_{0};
  uint64_t service_start_ns_=0;
  std::unique_ptr<PreviewSink> preview_;
  std::vector<std::unique_ptr<PipSink>> stream_sinks_;
  std::unique_ptr<StereoCoordinator> stereo_;
  std::unique_ptr<RoiController> roi_controller_;
  std::atomic<bool> roi_camera_switching_{false};
  std::atomic<uint64_t> stereo_switches_{0}, stereo_switch_failures_{0};
  std::atomic<double> last_stereo_switch_ms_{0};
  std::vector<std::unique_ptr<AutoRoiCoordinator>> auto_roi_;
  std::vector<std::unique_ptr<RoiController>> auto_roi_controllers_;
  mutable std::mutex roi_switch_mutex_;
};

} // namespace dual
