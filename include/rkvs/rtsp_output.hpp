#pragma once

#include "rkvs/config.hpp"
#include "rkvs/encoded_packet.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace rkvs {
class RtspOutputServer {
 public:
  RtspOutputServer(int port, std::vector<StreamOutputConfig> outputs,
                   std::function<void()> request_idr);
  ~RtspOutputServer();
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
