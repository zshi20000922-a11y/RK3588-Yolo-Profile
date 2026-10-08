#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>

namespace {
std::string command_json(const std::string& command) {
  if (!command.empty() && command.front() == '{') return command;
  return "{\"command\":\"" + command + "\"}";
}
}  // namespace

int main(int argc, char** argv) {
  std::string socket_path = "/run/rk-vision-service/control.sock";
  std::string command = "status";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--socket" && i + 1 < argc) socket_path = argv[++i];
    else command = arg;
  }
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    std::cerr << "socket failed\n";
    return 1;
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(address.sun_path)) {
    std::cerr << "socket path too long\n";
    close(fd);
    return 1;
  }
  std::strncpy(address.sun_path, socket_path.c_str(), sizeof(address.sun_path) - 1);
  if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
    std::cerr << "connect failed: " << socket_path << "\n";
    close(fd);
    return 1;
  }
  std::string request = command_json(command) + "\n";
  if (write(fd, request.data(), request.size()) != static_cast<ssize_t>(request.size())) {
    std::cerr << "write failed\n";
    close(fd);
    return 1;
  }
  char buffer[4096];
  while (true) {
    ssize_t count = read(fd, buffer, sizeof(buffer));
    if (count < 0) {
      std::cerr << "read failed\n";
      close(fd);
      return 1;
    }
    if (count == 0) break;
    std::cout.write(buffer, count);
  }
  close(fd);
  return 0;
}
