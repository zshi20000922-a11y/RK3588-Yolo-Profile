#include "rkvs/encoded_packet.hpp"
#include <stdexcept>

namespace rkvs {
void EncodedPacketBus::subscribe(Subscriber subscriber) {
  if (!subscriber) throw std::invalid_argument("packet subscriber required");
  std::lock_guard<std::mutex> lock(mutex_);
  subscribers_.push_back(std::move(subscriber));
}
void EncodedPacketBus::publish(SharedPacket packet) const {
  std::vector<Subscriber> copy;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    copy = subscribers_;
  }
  for (const auto& subscriber : copy) subscriber(packet);
}
}  // namespace rkvs
