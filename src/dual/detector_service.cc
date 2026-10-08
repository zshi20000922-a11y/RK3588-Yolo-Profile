#include "detector_service.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <sstream>

namespace dual {

struct DetectorService::WorkerSlot {
  int id=0;
  std::unique_ptr<RknnWorker> worker;
  std::thread inference_thread;
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<std::unique_ptr<PipelineJob>> queue;
  std::atomic<bool> infer_busy{false};
  // Number of jobs owned by this worker across preprocess, inference and
  // postprocess. The scheduler uses it to feed idle NPU cores before queuing a
  // second job behind a busy core.
  std::atomic<int> pending{0};
};

struct DetectorService::PreprocessTask {
  WorkerSlot* worker=nullptr;
  int tensor_slot=-1;
  size_t camera_index=0;
  bool make_preview=false;
  std::shared_ptr<FrameLease> lease;
};

namespace {
void pin_this_thread(int cpu) {
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu,&set);
  pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
}
void mkdir_parent(const std::string& path) {
  auto p=path.find_last_of('/');
  if(p!=std::string::npos) mkdir(path.substr(0,p).c_str(),0755);
}
int unix_listener(const std::string& path) {
  unlink(path.c_str());
  int fd=socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK,0);
  if(fd<0) throw std::runtime_error("socket failed");
  sockaddr_un a{}; a.sun_family=AF_UNIX;
  if(path.size()>=sizeof(a.sun_path)) {
    close(fd); throw std::runtime_error("socket path too long");
  }
  strncpy(a.sun_path,path.c_str(),sizeof(a.sun_path)-1);
  if(bind(fd,reinterpret_cast<sockaddr*>(&a),sizeof(a))<0 || listen(fd,8)<0) {
    int e=errno; close(fd);
    throw std::runtime_error("bind/listen: "+std::string(strerror(e)));
  }
  return fd;
}
void stop_aiq() {
  pid_t pid=fork();if(pid==0){execl("/usr/bin/killall","killall","rkaiq_3A_server",nullptr);_exit(127);}
  if(pid>0){int status=0;waitpid(pid,&status,0);}usleep(150000);
}
void start_aiq() {
  pid_t pid=fork();if(pid==0){setsid();int fd=open("/dev/null",O_RDWR);
    if(fd>=0){dup2(fd,STDIN_FILENO);dup2(fd,STDOUT_FILENO);dup2(fd,STDERR_FILENO);if(fd>2)close(fd);}
    execl("/usr/bin/rkaiq_3A_server","rkaiq_3A_server",nullptr);_exit(127);}
  usleep(500000);
}
}

DetectorService::DetectorService(ServiceConfig c)
    : config_(std::move(c)), inflight_(config_.cameras.size()),
      last_dispatch_ns_(config_.cameras.size(),0),
      last_preview_ns_(config_.cameras.size()),
      camera_published_(config_.cameras.size()),
      last_published_seq_(config_.cameras.size()),
      out_of_order_results_(config_.cameras.size()) {
  if(config_.cameras.empty() || config_.cameras.size()>2)
    throw std::invalid_argument("one or two cameras required");
  if(config_.detect_fps<1 || config_.detect_fps>60)
    throw std::invalid_argument("--detect-fps must be 1..60");
  if(config_.max_inflight_per_camera<1 || config_.max_inflight_per_camera>12)
    throw std::invalid_argument("--max-inflight must be 1..12");
  if(config_.buffers<3 || config_.buffers>32)
    throw std::invalid_argument("--buffers must be 3..32");
  if(config_.preprocess_threads<1 || config_.preprocess_threads>4 ||
     config_.postprocess_threads<1 || config_.postprocess_threads>4)
    throw std::invalid_argument("pipeline thread counts must be 1..4");
  if(!config_.stereo_calibration.empty()) {
    if(config_.cameras.size()!=2)
      throw std::invalid_argument("stereo ROI mode requires exactly two cameras");
    if(config_.roi_sensor_subdev.empty()||config_.roi_cif_subdev.empty()||
       config_.roi_isp_subdev.empty())
      throw std::invalid_argument("stereo ROI mode requires sensor/CIF/ISP subdev paths");
    StereoCalibration calibration;std::string error;
    if(!calibration.load(config_.stereo_calibration,&error))
      throw std::invalid_argument(error);
    stereo_=std::make_unique<StereoCoordinator>(std::move(calibration),
                                                config_.stereo_class_id,
                                                config_.stereo_project_fallback,
                                                config_.roi_lost_return_ms);
    roi_controller_=std::make_unique<RoiController>(
        config_.roi_sensor_subdev,config_.roi_cif_subdev,
        config_.roi_isp_subdev,config_.roi_capture_fps);
  }
  if(!config_.auto_roi_bindings.empty()) {
    if(stereo_) throw std::invalid_argument("auto ROI and stereo ROI are mutually exclusive");
    if(config_.auto_roi_bindings.size()!=config_.cameras.size())
      throw std::invalid_argument("auto ROI requires one binding per camera");
    for(const auto& camera:config_.cameras) {
      auto binding=std::find_if(config_.auto_roi_bindings.begin(),
          config_.auto_roi_bindings.end(),[&](const ServiceConfig::RoiBinding& b){
            return b.camera_id==camera.id;
          });
      if(binding==config_.auto_roi_bindings.end())
        throw std::invalid_argument("missing auto ROI binding for "+camera.id);
      auto_roi_.push_back(std::make_unique<AutoRoiCoordinator>(
          camera.id,config_.stereo_class_id,config_.roi_lost_return_ms));
      auto_roi_controllers_.push_back(std::make_unique<RoiController>(
          binding->sensor,binding->cif,binding->isp,config_.roi_capture_fps));
    }
  }
  for(auto& x:inflight_) x=0;
  for(auto& x:last_preview_ns_) x=0;
  for(auto& x:camera_published_) x=0;
  for(auto& x:last_published_seq_) x=0;
  for(auto& x:out_of_order_results_) x=0;
  for(const auto& camera:config_.cameras)
    cameras_.push_back(std::make_unique<V4L2Camera>(
        camera.id,camera.device,camera.capture_width,camera.capture_height,
        60,config_.buffers));
  RknnContextPool pool(config_.model,config_.labels,config_.contexts,
                       config_.independent_contexts);
  auto rknn_workers=pool.take_workers();
  for(auto& worker:rknn_workers) {
    auto slot=std::make_unique<WorkerSlot>();
    slot->id=workers_.size(); slot->worker=std::move(worker);
    workers_.push_back(std::move(slot));
  }
}

DetectorService::~DetectorService() { request_stop(); }

void DetectorService::notify_pipeline() {
  state_cv_.notify_all(); preprocess_cv_.notify_all();
  postprocess_cv_.notify_all(); result_cv_.notify_all();
  for(auto& worker:workers_) worker->cv.notify_all();
}

void DetectorService::run() {
  service_start_ns_=monotonic_ns();
  mkdir_parent(config_.result_socket); mkdir_parent(config_.control_socket);
  if(!config_.pip_target.empty()) {
    for(size_t index=0;index<cameras_.size();++index) {
      PipConfig stream_config;
      stream_config.target_ip=config_.pip_target;
      stream_config.camera_id=cameras_[index]->id();
      stream_config.overlay_target=index==0;
      stream_config.udp_port=config_.pip_udp_port+static_cast<int>(index);
      stream_config.rtsp_feed_port=config_.pip_rtsp_feed_port+static_cast<int>(index);
      auto sink=std::make_unique<PipSink>(stream_config);
      if(!sink->start()) throw std::runtime_error("cannot start camera stream sink");
      PipSink* sink_ptr=sink.get();
      cameras_[index]->set_frame_observer(
          [sink_ptr](std::shared_ptr<FrameLease> frame) {
            sink_ptr->submit(std::move(frame));
          });
      stream_sinks_.push_back(std::move(sink));
    }
  }
  for(auto& camera:cameras_) camera->start();
  if(config_.preview) {
    preview_=std::make_unique<PreviewSink>();
    if(!preview_->start())
      std::cerr<<"preview disabled: cannot start sink\n";
  }
  for(auto& worker:workers_)
    worker->inference_thread=
        std::thread(&DetectorService::inference_loop,this,worker.get());
  for(int i=0;i<config_.preprocess_threads;++i)
    preprocess_threads_.emplace_back(&DetectorService::preprocess_loop,this,i);
  for(int i=0;i<config_.postprocess_threads;++i)
    postprocess_threads_.emplace_back(&DetectorService::postprocess_loop,this,i);
  scheduler_=std::thread(&DetectorService::scheduler_loop,this);
  result_server_=std::thread(&DetectorService::result_server_loop,this);
  result_dispatch_=std::thread(&DetectorService::result_dispatch_loop,this);
  control_server_=std::thread(&DetectorService::control_server_loop,this);
  if(stereo_) stereo_control_=std::thread(&DetectorService::stereo_control_loop,this);
  if(!auto_roi_.empty())
    auto_roi_control_=std::thread(&DetectorService::auto_roi_control_loop,this);
  while(running_) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  request_stop();
}

void DetectorService::request_stop() {
  running_=false; notify_pipeline();
  if(result_listen_>=0) shutdown(result_listen_,SHUT_RDWR);
  if(control_listen_>=0) shutdown(control_listen_,SHUT_RDWR);
  if(scheduler_.joinable()) scheduler_.join();
  for(auto& thread:preprocess_threads_) if(thread.joinable()) thread.join();
  preprocess_stopped_=true;
  for(auto& worker:workers_) worker->cv.notify_all();
  for(auto& worker:workers_)
    if(worker->inference_thread.joinable()) worker->inference_thread.join();
  inference_stopped_=true; postprocess_cv_.notify_all();
  for(auto& thread:postprocess_threads_) if(thread.joinable()) thread.join();
  postprocess_stopped_=true; result_cv_.notify_all();
  if(result_server_.joinable()) result_server_.join();
  if(result_dispatch_.joinable()) result_dispatch_.join();
  if(control_server_.joinable()) control_server_.join();
  if(stereo_control_.joinable()) stereo_control_.join();
  if(auto_roi_control_.joinable()) auto_roi_control_.join();
  if(!stream_sinks_.empty()) {
    for(auto& camera:cameras_)camera->set_frame_observer({});
    stream_sinks_.clear();
  }
  for(auto& camera:cameras_) camera->stop();
  {
    std::lock_guard<std::mutex> lock(client_mutex_);
    for(int fd:result_clients_) close(fd);
    result_clients_.clear();
  }
  if(result_listen_>=0) { close(result_listen_); result_listen_=-1; }
  if(control_listen_>=0) { close(control_listen_); control_listen_=-1; }
  unlink(config_.result_socket.c_str()); unlink(config_.control_socket.c_str());
}

void DetectorService::scheduler_loop() {
  pin_this_thread(4);
  size_t next_camera=0, next_worker=0;
  // At the 60 FPS ceiling the camera itself is the rate limiter. An extra
  // software period loses frames through scheduler jitter (typically 59.x
  // FPS), so dispatch every newly available latest frame.
  const uint64_t period=config_.detect_fps>=60 ? 0 :
                        1000000000ULL/config_.detect_fps;
  while(running_) {
    bool dispatched=false;
    if(!paused_&&!roi_camera_switching_) {
      WorkerSlot* selected_worker=nullptr; int tensor_slot=-1;
      // A worker owns two tensor slots so RGA can overlap with NPU, but its
      // inference thread is serial. Prefer the least-loaded worker; the old
      // slot-only round robin could queue work behind a busy core while an
      // adjacent NPU core was idle.
      for(int desired_pending=0;desired_pending<2 && !selected_worker;
          ++desired_pending) {
        for(size_t n=0;n<workers_.size();++n) {
          size_t wi=(next_worker+n)%workers_.size();
          if(workers_[wi]->pending.load()!=desired_pending) continue;
          int candidate=workers_[wi]->worker->try_acquire_slot();
          if(candidate>=0) {
            selected_worker=workers_[wi].get(); tensor_slot=candidate;
            next_worker=(wi+1)%workers_.size(); break;
          }
        }
      }
      if(selected_worker) {
        size_t camera_index=cameras_.size();
        std::shared_ptr<FrameLease> frame;
        // Strict round robin while both cameras are online. If the next
        // camera has reached its in-flight cap, wait for it instead of giving
        // all newly freed workers to the faster camera. Offline cameras are
        // skipped so the healthy channel continues at full speed.
        for(size_t n=0;n<cameras_.size();++n) {
          size_t ci=next_camera;
          if(!cameras_[ci]->online()) {
            next_camera=(next_camera+1)%cameras_.size();
            continue;
          }
          uint64_t now=monotonic_ns();
          if(inflight_[ci].load()>=config_.max_inflight_per_camera ||
             now-last_dispatch_ns_[ci]<period) break;
          frame=cameras_[ci]->take_latest();
          if(!frame) break;
          camera_index=ci; last_dispatch_ns_[ci]=now;
          next_camera=(ci+1)%cameras_.size(); break;
        }
        if(frame) {
          auto task=std::make_unique<PreprocessTask>();
          task->worker=selected_worker; task->tensor_slot=tensor_slot;
          task->camera_index=camera_index; task->lease=std::move(frame);
          if(config_.preview||stereo_) {
            uint64_t now=monotonic_ns();
            uint64_t old=last_preview_ns_[camera_index].load();
            if(now-old>=100000000ULL &&
               last_preview_ns_[camera_index].compare_exchange_strong(old,now))
              task->make_preview=true;
          }
          inflight_[camera_index].fetch_add(1);
          selected_worker->pending.fetch_add(1);
          {
            std::lock_guard<std::mutex> lock(preprocess_mutex_);
            preprocess_queue_.push_back(std::move(task));
          }
          preprocess_cv_.notify_one(); dispatched=true;
        } else {
          selected_worker->worker->release_slot(tensor_slot);
        }
      }
    }
    if(!dispatched) {
      std::unique_lock<std::mutex> lock(state_mutex_);
      state_cv_.wait_for(lock,std::chrono::microseconds(500));
    }
  }
}

void DetectorService::fail_job(int worker_id, int tensor_slot,
                               size_t camera_index, const char* stage,
                               const std::exception& error) {
  ++pipeline_errors_;
  std::cerr<<stage<<" error: "<<error.what()<<'\n';
  if(worker_id>=0 && static_cast<size_t>(worker_id)<workers_.size())
  {
    workers_[worker_id]->worker->release_slot(tensor_slot);
    workers_[worker_id]->pending.fetch_sub(1);
  }
  if(camera_index<inflight_.size()) inflight_[camera_index].fetch_sub(1);
  state_cv_.notify_all();
}

void DetectorService::preprocess_loop(int index) {
  pin_this_thread(4+(index%2));
  while(true) {
    std::unique_ptr<PreprocessTask> task;
    {
      std::unique_lock<std::mutex> lock(preprocess_mutex_);
      preprocess_cv_.wait(lock,[&]{return !running_||!preprocess_queue_.empty();});
      if(preprocess_queue_.empty()) {
        if(!running_) break;
        continue;
      }
      task=std::move(preprocess_queue_.front()); preprocess_queue_.pop_front();
    }
    try {
      auto job=task->worker->worker->preprocess(
          task->lease->frame(),task->camera_index,task->tensor_slot,
          task->make_preview);
      // RGA has synchronously finished reading the camera DMA-BUF. Returning
      // the V4L2 buffer here is safe and keeps capture independent of NPU/CPU.
      task->lease.reset();
      {
        std::lock_guard<std::mutex> lock(task->worker->mutex);
        task->worker->queue.push_back(std::move(job));
      }
      task->worker->cv.notify_one();
    } catch(const std::exception& error) {
      task->lease.reset();
      fail_job(task->worker->id,task->tensor_slot,task->camera_index,
               "preprocess",error);
    }
  }
}

void DetectorService::inference_loop(WorkerSlot* worker) {
  // RKNN submission and wait still consume host CPU time. Keep the three NPU
  // workers on distinct big cores; the previous 6+(id%2) mapping placed
  // worker 0 and worker 2 on CPU6 and made the third NPU path ~2x slower.
  pin_this_thread(5+(worker->id%3));
  while(true) {
    std::unique_ptr<PipelineJob> job;
    {
      std::unique_lock<std::mutex> lock(worker->mutex);
      worker->cv.wait(lock,[&]{
        return preprocess_stopped_||!worker->queue.empty();
      });
      if(worker->queue.empty()) {
        if(preprocess_stopped_) break;
        continue;
      }
      job=std::move(worker->queue.front()); worker->queue.pop_front();
      worker->infer_busy=true;
    }
    try {
      worker->worker->run(*job);
      {
        std::lock_guard<std::mutex> lock(postprocess_mutex_);
        postprocess_queue_.push_back(std::move(job));
      }
      postprocess_cv_.notify_one();
    } catch(const std::exception& error) {
      fail_job(worker->id,job->slot,job->camera_index,"inference",error);
    }
    worker->infer_busy=false; state_cv_.notify_all();
  }
}

void DetectorService::postprocess_loop(int index) {
  pin_this_thread(2+(index%2));
  while(true) {
    std::unique_ptr<PipelineJob> job;
    {
      std::unique_lock<std::mutex> lock(postprocess_mutex_);
      postprocess_cv_.wait(lock,[&]{
        return inference_stopped_||!postprocess_queue_.empty();
      });
      if(postprocess_queue_.empty()) {
        if(inference_stopped_) break;
        continue;
      }
      job=std::move(postprocess_queue_.front()); postprocess_queue_.pop_front();
    }
    auto* worker=workers_[job->worker_id].get();
    try {
      worker->worker->postprocess(*job);
      if(stereo_) stereo_->process(&job->batch);
      if(job->camera_index<auto_roi_.size())
        auto_roi_[job->camera_index]->process(&job->batch);
      const auto& camera_config=config_.cameras[job->camera_index];
      if(job->batch.width!=camera_config.logical_width ||
         job->batch.height!=camera_config.logical_height) {
        const float sx=static_cast<float>(camera_config.logical_width)/
                       job->batch.width;
        const float sy=static_cast<float>(camera_config.logical_height)/
                       job->batch.height;
        for(auto& detection:job->batch.detections) {
          detection.left*=sx; detection.right*=sx;
          detection.top*=sy; detection.bottom*=sy;
        }
        job->batch.width=camera_config.logical_width;
        job->batch.height=camera_config.logical_height;
      }
      auto stats=cameras_[job->camera_index]->stats();
      job->batch.dropped_frames=stats.dropped;
      job->batch.v4l2_timeouts=stats.timeouts;
      job->batch.reconnects=stats.reconnects;
      uint64_t previous=last_published_seq_[job->camera_index].load();
      while(job->batch.sequence>previous &&
            !last_published_seq_[job->camera_index].compare_exchange_weak(
                previous,job->batch.sequence)) {}
      const bool stale=job->batch.sequence<=previous;
      if(stale)
        out_of_order_results_[job->camera_index].fetch_add(1);
      else {
        camera_published_[job->camera_index].fetch_add(1);
        for(auto& sink:stream_sinks_)sink->update_detections(job->batch);
        if(preview_) preview_->submit(job->batch);
        enqueue_result(std::move(job->batch));
      }
      worker->worker->release_slot(job->slot);
      worker->pending.fetch_sub(1);
      inflight_[job->camera_index].fetch_sub(1);
      state_cv_.notify_all();
    } catch(const std::exception& error) {
      fail_job(worker->id,job->slot,job->camera_index,"postprocess",error);
    }
  }
}

void DetectorService::publish(DetectionBatch batch) {
  std::string line=to_json(batch);
  std::lock_guard<std::mutex> lock(client_mutex_); ++published_;
  for(auto it=result_clients_.begin();it!=result_clients_.end();) {
    ssize_t n=send(*it,line.data(),line.size(),MSG_NOSIGNAL|MSG_DONTWAIT);
    if(n==(ssize_t)line.size()) { ++it; continue; }
    if(n<0 && (errno==EAGAIN||errno==EWOULDBLOCK)) {
      ++client_drops_; ++it; continue;
    }
    close(*it); it=result_clients_.erase(it);
  }
}

void DetectorService::enqueue_result(DetectionBatch batch) {
  std::lock_guard<std::mutex> lock(result_mutex_);
  if(pending_results_.size()>=32) {
    pending_results_.pop_front(); ++result_queue_drops_;
  }
  pending_results_.push_back(std::move(batch)); result_cv_.notify_one();
}

void DetectorService::result_dispatch_loop() {
  pin_this_thread(3);
  while(true) {
    DetectionBatch batch;
    {
      std::unique_lock<std::mutex> lock(result_mutex_);
      result_cv_.wait(lock,[&]{
        return postprocess_stopped_||!pending_results_.empty();
      });
      if(pending_results_.empty()) {
        if(postprocess_stopped_) break;
        continue;
      }
      batch=std::move(pending_results_.front()); pending_results_.pop_front();
    }
    publish(std::move(batch));
  }
}

void DetectorService::result_server_loop() {
  try { result_listen_=unix_listener(config_.result_socket); }
  catch(const std::exception& error) {
    std::cerr<<"result socket: "<<error.what()<<'\n'; running_=false;
    notify_pipeline(); return;
  }
  while(running_) {
    pollfd p{result_listen_,POLLIN,0}; if(poll(&p,1,200)<=0) continue;
    int client=accept4(result_listen_,nullptr,nullptr,
                       SOCK_CLOEXEC|SOCK_NONBLOCK);
    if(client>=0) {
      std::lock_guard<std::mutex> lock(client_mutex_);
      result_clients_.push_back(client);
    }
  }
}

std::string DetectorService::status_json() const {
  uint64_t published,drops;
  { std::lock_guard<std::mutex> lock(client_mutex_);
    published=published_; drops=client_drops_; }
  size_t pre_depth,post_depth;
  { std::lock_guard<std::mutex> lock(preprocess_mutex_);
    pre_depth=preprocess_queue_.size(); }
  { std::lock_guard<std::mutex> lock(postprocess_mutex_);
    post_depth=postprocess_queue_.size(); }
  std::ostringstream out;
  out<<"{\"running\":"<<(running_?"true":"false")
     <<",\"paused\":"<<(paused_?"true":"false")
     <<",\"contexts\":"<<workers_.size()
     <<",\"tensor_slots_per_context\":"<<RknnWorker::kTensorSlots
     <<",\"preprocess_threads\":"<<config_.preprocess_threads
     <<",\"postprocess_threads\":"<<config_.postprocess_threads
     <<",\"preprocess_queue_depth\":"<<pre_depth
     <<",\"postprocess_queue_depth\":"<<post_depth
     <<",\"pipeline_errors\":"<<pipeline_errors_
     <<",\"stereo\":"<<(stereo_?stereo_->status_json():"{\"enabled\":false}")
     <<",\"stereo_switch_ms\":"<<last_stereo_switch_ms_.load()
     <<",\"stereo_switches\":"<<stereo_switches_.load()
     <<",\"stereo_switch_failures\":"<<stereo_switch_failures_.load()
     <<",\"auto_roi\":[";
  for(size_t i=0;i<auto_roi_.size();++i) {
    if(i) out<<','; out<<auto_roi_[i]->status_json();
  }
  out<<']'
     <<",\"preview\":"<<(config_.preview?"true":"false")
     <<",\"preview_submitted\":"<<(preview_?preview_->submitted():0)
     <<",\"preview_processed\":"<<(preview_?preview_->processed():0)
     <<",\"preview_written\":"<<(preview_?preview_->written():0)
     <<",\"pip\":"<<(!stream_sinks_.empty()?"true":"false")
     <<",\"streams\":[";
  for(size_t i=0;i<stream_sinks_.size();++i){if(i)out<<',';
    out<<"{\"camera_id\":\""<<config_.cameras[i].id
       <<"\",\"frames\":"<<stream_sinks_[i]->frames()
       <<",\"dropped\":"<<stream_sinks_[i]->dropped()
       <<",\"rga_ms\":"<<stream_sinks_[i]->rga_ms()
       <<",\"mpp_ms\":"<<stream_sinks_[i]->mpp_ms()<<'}';}
  out<<']'
     <<",\"max_inflight_per_camera\":"<<config_.max_inflight_per_camera
     <<",\"published\":"<<published
     <<",\"slow_client_drops\":"<<drops
     <<",\"result_queue_drops\":"<<result_queue_drops_
     <<",\"cameras\":[";
  double elapsed=std::max(0.001,(monotonic_ns()-service_start_ns_)/1e9);
  for(size_t i=0;i<cameras_.size();++i) {
    if(i) out<<','; auto stats=cameras_[i]->stats();
    double capture_fps=0;
    if(stats.captured>1&&stats.last_capture_ts_ns>stats.first_capture_ts_ns)
      capture_fps=(stats.captured-1)*1e9/
          static_cast<double>(stats.last_capture_ts_ns-stats.first_capture_ts_ns);
    out<<"{\"id\":\""<<json_escape(cameras_[i]->id())
       <<"\",\"online\":"<<(cameras_[i]->online()?"true":"false")
       <<",\"inflight\":"<<inflight_[i].load()
       <<",\"result_fps\":"<<(camera_published_[i].load()/elapsed)
       <<",\"capture_fps\":"<<capture_fps
       <<",\"capture_width\":"<<stats.width
       <<",\"capture_height\":"<<stats.height
       <<",\"requested_fps\":"<<stats.requested_fps
       <<",\"captured\":"<<stats.captured
       <<",\"out_of_order_results\":"<<out_of_order_results_[i].load()
       <<",\"dropped\":"<<stats.dropped
       <<",\"timeouts\":"<<stats.timeouts
       <<",\"reconnects\":"<<stats.reconnects<<'}';
  }
  out<<"]}\n"; return out.str();
}

void DetectorService::set_paused(bool pause) {
  paused_=pause;
  if(pause) {
    while(running_) {
      bool busy=false;
      for(const auto& count:inflight_) busy|=count.load()>0;
      if(!busy) break;
      usleep(1000);
    }
  } else {
    state_cv_.notify_all();
  }
}

void DetectorService::control_server_loop() {
  try { control_listen_=unix_listener(config_.control_socket); }
  catch(const std::exception& error) {
    std::cerr<<"control socket: "<<error.what()<<'\n'; running_=false;
    notify_pipeline(); return;
  }
  while(running_) {
    pollfd p{control_listen_,POLLIN,0}; if(poll(&p,1,200)<=0) continue;
    int client=accept4(control_listen_,nullptr,nullptr,SOCK_CLOEXEC);
    if(client<0) continue;
    char buffer[1024]; ssize_t n=recv(client,buffer,sizeof(buffer)-1,0);
    std::string response;
    if(n>0) {
      buffer[n]=0; std::string command(buffer);
      if(command.find("pause")!=std::string::npos) {
        set_paused(true); response=status_json();
      } else if(command.find("resume")!=std::string::npos) {
        set_paused(false); response=status_json();
      } else if(command.find("shutdown")!=std::string::npos) {
        response="{\"ok\":true,\"shutdown\":true}\n";
        running_=false; notify_pipeline();
      } else if(command.find("status")!=std::string::npos) {
        response=status_json();
      } else if(command.find("unlock")!=std::string::npos&&stereo_) {
        stereo_->unlock(); response=status_json();
      } else if(command.find("projected fallback on")!=std::string::npos&&stereo_) {
        stereo_->set_projected_fallback(true); response=status_json();
      } else if(command.find("projected fallback off")!=std::string::npos&&stereo_) {
        stereo_->set_projected_fallback(false); response=status_json();
      } else if(command.find("auto on")!=std::string::npos&&!auto_roi_.empty()) {
        for(auto& coordinator:auto_roi_)coordinator->set_auto(true);
        response=status_json();
      } else if(command.find("auto off")!=std::string::npos&&!auto_roi_.empty()) {
        for(auto& coordinator:auto_roi_)coordinator->set_auto(false);
        response=status_json();
      } else if(command.find("reset")!=std::string::npos&&!auto_roi_.empty()) {
        for(auto& coordinator:auto_roi_)coordinator->force_full();
        response=status_json();
      } else if(command.find("cam0 full")!=std::string::npos&&auto_roi_.size()>0) {
        auto_roi_[0]->force_full(); response=status_json();
      } else if(command.find("cam0 roi")!=std::string::npos&&auto_roi_.size()>0) {
        auto_roi_[0]->force_roi(); response=status_json();
      } else if(command.find("cam1 full")!=std::string::npos&&auto_roi_.size()>1) {
        auto_roi_[1]->force_full(); response=status_json();
      } else if(command.find("cam1 roi")!=std::string::npos&&auto_roi_.size()>1) {
        auto_roi_[1]->force_roi(); response=status_json();
      } else {
        response="{\"error\":\"command must be status, auto on/off, cam0 full/roi, cam1 full/roi, reset, pause, resume or shutdown\"}\n";
      }
      send(client,response.data(),response.size(),MSG_NOSIGNAL);
    }
    close(client);
  }
}

void DetectorService::stereo_control_loop() {
  pin_this_thread(1);
  while(running_) {
    auto action=stereo_->take_action();
    if(!action){usleep(5000);continue;}
    const uint64_t switch_begin_ns=monotonic_ns();
    const char* action_name=action->kind==RoiActionKind::ENTER_ROI?"ENTER_ROI":
        action->kind==RoiActionKind::MOVE_ROI?"MOVE_ROI":"ENTER_FULL";
    std::cout<<"stereo action "<<action_name<<" center="<<action->center.x
             <<","<<action->center.y<<std::endl;
    bool ok=false;std::string error;
    if(action->kind==RoiActionKind::MOVE_ROI) {
      ok=roi_controller_->move(action->center,&error);
    } else {
      roi_camera_switching_=true;
      while(running_&&(inflight_[0].load()>0||inflight_[1].load()>0))usleep(1000);
      cameras_[0]->stop();cameras_[1]->stop();stop_aiq();
      if(action->kind==RoiActionKind::ENTER_ROI) {
        ok=roi_controller_->enter_roi(action->center,&error);
        if(ok){
          start_aiq();
          ok=cameras_[0]->restart(3840,2160,60)&&
             cameras_[1]->restart(640,640,config_.roi_capture_fps);
          // RKAIQ reapplies the mode's default VBLANK while starting. Apply
          // the experimental high-speed value only after Camera 1 has
          // delivered frames; V4L2Camera::restart() itself is asynchronous.
          if(ok){
            const uint64_t deadline=monotonic_ns()+8000000000ULL;
            while(running_&&monotonic_ns()<deadline&&
                  cameras_[1]->stats().captured<2)usleep(10000);
            if(cameras_[1]->stats().captured<2){
              ok=false;error="Camera 1 ROI stream did not produce frames";
            }else ok=roi_controller_->apply_roi_fps(&error);
          }
        }
        if(!ok) {
          const std::string original_error=error.empty()?"enter ROI failed":error;
          cameras_[0]->stop();cameras_[1]->stop();stop_aiq();std::string rollback_error;
          bool rollback=roi_controller_->enter_full(&rollback_error);
          if(rollback){start_aiq();rollback=cameras_[0]->restart(3840,2160,60)&&
                                               cameras_[1]->restart(3840,2160,60);}
          roi_camera_switching_=false;state_cv_.notify_all();
          stereo_->action_complete(RoiActionKind::ENTER_FULL,rollback,
              roi_controller_->roi(),original_error+
              (rollback?"; rolled back to full":"; rollback failed: "+rollback_error));
          continue;
        }
      } else {
        ok=roi_controller_->enter_full(&error);
        if(ok){start_aiq();ok=cameras_[0]->restart(3840,2160,60)&&
                              cameras_[1]->restart(3840,2160,60);}
      }
      if(!ok&&error.empty())error="camera restart failed";
      roi_camera_switching_=false;state_cv_.notify_all();
    }
    stereo_->action_complete(action->kind,ok,roi_controller_->roi(),error);
    last_stereo_switch_ms_=(monotonic_ns()-switch_begin_ns)/1e6;
    ++stereo_switches_;
    if(!ok)++stereo_switch_failures_;
    const auto applied=roi_controller_->roi();
    std::cout<<"stereo action complete "<<action_name<<" ok="<<(ok?1:0)
             <<" roi="<<applied.left<<","<<applied.top<<",640,640"
             <<(error.empty()?"":" error=")<<error<<std::endl;
  }
}

void DetectorService::auto_roi_control_loop() {
  pin_this_thread(1);
  while(running_) {
    size_t camera_index=auto_roi_.size();
    std::optional<AutoRoiAction> action;
    for(size_t i=0;i<auto_roi_.size();++i) {
      action=auto_roi_[i]->take_action();
      if(action){camera_index=i;break;}
    }
    if(!action){usleep(5000);continue;}
    auto& controller=*auto_roi_controllers_[camera_index];
    bool ok=false;std::string error;
    if(action->kind==AutoRoiActionKind::MOVE_ROI) {
      ok=controller.move(action->center,&error);
      auto_roi_[camera_index]->action_complete(action->kind,ok,controller.roi(),error);
      continue;
    }

    std::lock_guard<std::mutex> switch_lock(roi_switch_mutex_);
    roi_camera_switching_=true;
    while(running_) {
      bool busy=false;for(const auto& count:inflight_)busy|=count.load()>0;
      if(!busy)break;usleep(1000);
    }
    for(auto& camera:cameras_)camera->stop();
    stop_aiq();
    if(action->kind==AutoRoiActionKind::ENTER_ROI)
      ok=controller.enter_roi(action->center,&error);
    else
      ok=controller.enter_full(&error);

    if(ok) {
      start_aiq();
      for(size_t i=0;i<cameras_.size()&&ok;++i) {
        const bool roi=auto_roi_controllers_[i]->roi_mode();
        const int width=roi?640:config_.cameras[i].capture_width;
        const int height=roi?640:config_.cameras[i].capture_height;
        const int fps=roi?config_.roi_capture_fps:60;
        ok=cameras_[i]->restart(width,height,fps);
        if(!ok)error="camera restart failed for "+config_.cameras[i].id;
      }
      if(ok&&action->kind==AutoRoiActionKind::ENTER_ROI) {
        usleep(500000);
        ok=controller.apply_roi_fps(&error);
      }
    }

    if(!ok&&action->kind==AutoRoiActionKind::ENTER_ROI) {
      const std::string original=error.empty()?"enter ROI failed":error;
      for(auto& camera:cameras_)camera->stop();
      stop_aiq();std::string rollback_error;
      const bool configured=controller.enter_full(&rollback_error);
      bool restarted=configured;
      if(configured) {
        start_aiq();
        for(size_t i=0;i<cameras_.size()&&restarted;++i) {
          const bool roi=auto_roi_controllers_[i]->roi_mode();
          restarted=cameras_[i]->restart(
              roi?640:config_.cameras[i].capture_width,
              roi?640:config_.cameras[i].capture_height,
              roi?config_.roi_capture_fps:60);
        }
      }
      error=original+(restarted?"; rolled back to full":"; rollback failed: "+rollback_error);
    }
    roi_camera_switching_=false;state_cv_.notify_all();
    auto_roi_[camera_index]->action_complete(action->kind,ok,controller.roi(),error);
  }
}
} // namespace dual
