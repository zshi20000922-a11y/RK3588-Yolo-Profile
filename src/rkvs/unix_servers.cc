#include "rkvs/unix_servers.hpp"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace rkvs {
namespace {
int listen_unix(const std::string& path) {
  if (path.size() >= sizeof(sockaddr_un::sun_path))
    throw std::invalid_argument("Unix socket path too long: " + path);
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (fd < 0) throw std::runtime_error("socket: " + std::string(strerror(errno)));
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
  unlink(path.c_str());
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
      listen(fd, 16) < 0) {
    const std::string error = strerror(errno);
    close(fd);
    throw std::runtime_error("listen " + path + ": " + error);
  }
  chmod(path.c_str(), 0660);
  return fd;
}
}  // namespace

JsonlBroadcastServer::JsonlBroadcastServer(std::string path) : path_(std::move(path)) {}
JsonlBroadcastServer::~JsonlBroadcastServer() { stop(); }
void JsonlBroadcastServer::start() {
  if (running_.exchange(true)) return;
  try { listener_ = listen_unix(path_); } catch (...) { running_ = false; throw; }
  thread_ = std::thread(&JsonlBroadcastServer::accept_loop, this);
}
void JsonlBroadcastServer::stop() {
  if (!running_.exchange(false)) return;
  if (listener_ >= 0) { close(listener_); listener_ = -1; }
  if (thread_.joinable()) thread_.join();
  std::lock_guard<std::mutex> lock(mutex_);
  for (int client : clients_) close(client);
  clients_.clear();
  unlink(path_.c_str());
}
void JsonlBroadcastServer::accept_loop() {
  while (running_) {
    int client = accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
    if (client >= 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      clients_.push_back(client);
    } else if (errno == EAGAIN || errno == EINTR) usleep(20000);
    else break;
  }
}
void JsonlBroadcastServer::publish(const std::string& line) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = clients_.begin(); it != clients_.end();) {
    const ssize_t sent = send(*it, line.data(), line.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
    // Never permit a slow client to block inference or receive partial JSON.
    if (sent != static_cast<ssize_t>(line.size())) {
      close(*it);
      it = clients_.erase(it);
    } else ++it;
  }
}
size_t JsonlBroadcastServer::clients() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return clients_.size();
}

UnixControlServer::UnixControlServer(std::string path, Handler handler)
    : path_(std::move(path)), handler_(std::move(handler)) {
  if (!handler_) throw std::invalid_argument("control handler required");
}
UnixControlServer::~UnixControlServer() { stop(); }
void UnixControlServer::start() {
  if (running_.exchange(true)) return;
  try { listener_ = listen_unix(path_); } catch (...) { running_ = false; throw; }
  thread_ = std::thread(&UnixControlServer::loop, this);
}
void UnixControlServer::stop() {
  if (!running_.exchange(false)) return;
  if (listener_ >= 0) { close(listener_); listener_ = -1; }
  if (thread_.joinable()) thread_.join();
  unlink(path_.c_str());
}
void UnixControlServer::loop() {
  while (running_) {
    int client = accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
    if (client < 0) {
      if (errno == EAGAIN || errno == EINTR) { usleep(20000); continue; }
      break;
    }
    timeval timeout{1, 0};
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    char buffer[4096];
    const ssize_t bytes = recv(client, buffer, sizeof(buffer) - 1, 0);
    if (bytes > 0) {
      buffer[bytes] = 0;
      std::string response;
      try { response = handler_(buffer); }
      catch (const std::exception& error) {
        response = std::string("{\"ok\":false,\"error\":\"") + error.what() + "\"}";
      }
      response.push_back('\n');
      send(client, response.data(), response.size(), MSG_NOSIGNAL);
    }
    close(client);
  }
}
}  // namespace rkvs
