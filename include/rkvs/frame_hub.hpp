#pragma once

#include "rkvs/types.hpp"

#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace rkvs {

class LatestFrameHub {
 public:
  class Consumer {
   public:
    Consumer() = default;
    Consumer(LatestFrameHub* hub, std::string name);
    SharedFrame take(const std::string& source_id);
    SharedFrame wait_take(const std::string& source_id, int timeout_ms);
    std::vector<SharedFrame> take_ready();

   private:
    LatestFrameHub* hub_ = nullptr;
    std::string name_;
    std::map<std::string, uint64_t> cursors_;
  };

  Consumer subscribe(std::string name);
  void publish(SharedFrame frame);
  void close();
  uint64_t overwritten(const std::string& source_id) const;

 private:
  friend class Consumer;
  SharedFrame take_after(const std::string& source_id, uint64_t* cursor,
                         int timeout_ms, bool wait);
  std::vector<SharedFrame> take_ready_after(
      std::map<std::string, uint64_t>* cursors);

  struct Slot {
    SharedFrame frame;
    uint64_t generation = 0;
    uint64_t publishes = 0;
  };
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::map<std::string, Slot> slots_;
  bool closed_ = false;
};

}  // namespace rkvs
