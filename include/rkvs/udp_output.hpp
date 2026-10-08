#pragma once

#include "rkvs/config.hpp"
#include "rkvs/encoded_packet.hpp"
#include <memory>

namespace rkvs {
class UdpMpegTsOutput {
 public:
  explicit UdpMpegTsOutput(StreamOutputConfig config);
  ~UdpMpegTsOutput();
  void start();
  void stop();
  void submit(SharedPacket packet);
  uint64_t packets() const;
  uint64_t dropped() const;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace rkvs
