#pragma once

#include "rkvs/types.hpp"

namespace rkvs {

class IVideoSource {
 public:
  virtual ~IVideoSource() = default;
  virtual bool start(FrameCallback callback) = 0;
  virtual void stop() = 0;
  virtual void set_enabled(bool enabled) = 0;
  virtual SourceStatus status() const = 0;
};

}  // namespace rkvs
