#pragma once

#include "rkvs/config.hpp"
#include "rkvs/encoded_packet.hpp"
#include "rkvs/types.hpp"

#include <memory>

namespace rkvs {
class MppEncoder {
 public:
  MppEncoder(StreamOutputConfig config, EncodedPacketBus* bus);
  ~MppEncoder();
  void start();
  void stop();
  void submit(SharedFrame frame);
  void request_idr();
  uint64_t encoded() const;
  uint64_t dropped() const;
  double last_encode_ms() const;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace rkvs
