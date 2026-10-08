#pragma once

#include "rkvs/config.hpp"
#include "rkvs/video_source.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

namespace rkvs {

class RtspMppSource final : public IVideoSource {
 public:
  explicit RtspMppSource(SourceConfig config);
  ~RtspMppSource() override;
  bool start(FrameCallback callback) override;
  void stop() override;
  void set_enabled(bool enabled) override;
  SourceStatus status() const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rkvs
