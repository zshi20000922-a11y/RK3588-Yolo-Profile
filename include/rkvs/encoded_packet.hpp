#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rkvs {
struct EncodedPacket {
  std::string output_id;
  std::string codec;
  uint64_t sequence = 0;
  uint64_t pts_ns = 0;
  bool keyframe = false;
  std::vector<uint8_t> data;
};
using SharedPacket = std::shared_ptr<const EncodedPacket>;

class EncodedPacketBus {
 public:
  using Subscriber = std::function<void(SharedPacket)>;
  void subscribe(Subscriber subscriber);
  void publish(SharedPacket packet) const;
 private:
  mutable std::mutex mutex_;
  std::vector<Subscriber> subscribers_;
};
}  // namespace rkvs
