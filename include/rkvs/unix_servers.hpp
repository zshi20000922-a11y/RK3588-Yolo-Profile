#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rkvs {
class JsonlBroadcastServer {
 public:
  explicit JsonlBroadcastServer(std::string path);
  ~JsonlBroadcastServer();
  void start();
  void stop();
  void publish(const std::string& line);
  size_t clients() const;
 private:
  void accept_loop();
  std::string path_;
  int listener_ = -1;
  std::atomic<bool> running_{false};
  std::thread thread_;
  mutable std::mutex mutex_;
  std::vector<int> clients_;
};

class UnixControlServer {
 public:
  using Handler = std::function<std::string(const std::string&)>;
  UnixControlServer(std::string path, Handler handler);
  ~UnixControlServer();
  void start();
  void stop();
 private:
  void loop();
  std::string path_;
  Handler handler_;
  int listener_ = -1;
  std::atomic<bool> running_{false};
  std::thread thread_;
};
}  // namespace rkvs
