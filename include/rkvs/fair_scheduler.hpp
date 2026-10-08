#pragma once

#include "rkvs/types.hpp"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rkvs {

struct SchedulerSourceConfig {
  std::string source_id;
  int target_fps = 1;
};

struct SchedulerSourceStats {
  uint64_t submitted = 0;
  uint64_t replaced = 0;
  uint64_t completed = 0;
  uint64_t errors = 0;
  bool in_flight = false;
  int in_flight_count = 0;
  int queue_depth = 0;
  double detection_fps = 0;
  double latency_p50_ms = 0;
  double latency_p95_ms = 0;
};

class LatestFairScheduler {
 public:
  using Processor = std::function<ResultBatch(SharedFrame, int worker_index)>;
  using ResultCallback = std::function<void(ResultBatch)>;

  LatestFairScheduler(std::vector<SchedulerSourceConfig> sources, int workers,
                      Processor processor, ResultCallback result_callback);
  ~LatestFairScheduler();
  void start();
  void stop();
  void submit(SharedFrame frame);
  void pause();
  void resume();
  void update_sources(const std::vector<SchedulerSourceConfig>& sources);
  bool paused() const;
  std::map<std::string, SchedulerSourceStats> stats() const;

 private:
  struct SourceSlot {
    SchedulerSourceConfig config;
    SharedFrame pending;
    uint64_t next_due_ns = 0;
    SchedulerSourceStats stats;
    uint64_t first_completed_ns = 0;
    std::vector<double> latency_samples;
  };
  void worker_loop(int worker_index);

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::map<std::string, SourceSlot> sources_;
  std::vector<std::string> order_;
  size_t cursor_ = 0;
  int worker_count_ = 0;
  int max_in_flight_per_source_ = 1;
  Processor processor_;
  ResultCallback result_callback_;
  std::vector<std::thread> threads_;
  bool running_ = false;
  bool paused_ = false;
};

}  // namespace rkvs
