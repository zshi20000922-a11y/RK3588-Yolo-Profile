#pragma once

#include "rkvs/types.hpp"

#include <memory>

namespace rkvs {

class WaylandDisplay {
 public:
  WaylandDisplay(int width, int height, int fps, bool fullscreen = true);
  ~WaylandDisplay();
  void start();
  void stop();
  void submit(SharedFrame frame);
  uint64_t displayed() const;
  uint64_t dropped() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rkvs
