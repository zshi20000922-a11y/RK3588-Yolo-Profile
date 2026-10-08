#include "rkvs/config.hpp"
#include "rkvs/rknn_detector.hpp"
#include "rkvs/types.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
struct DmaHeapAllocationData { uint64_t len; uint32_t fd; uint32_t fd_flags; uint64_t heap_flags; };
#define RK_BENCH_DMA_HEAP_IOC_ALLOC _IOWR('H', 0x0, DmaHeapAllocationData)

struct Options {
  std::string model, manifest, labels, family, nv12, output = "raw_timings.csv", perf_detail;
  std::string mode = "latency", input_mode = "dma-replay";
  int width = 1920, height = 1080, contexts = 1, warmup = 100;
  uint64_t iterations = 3000;
  double seconds = 60.0;
};

std::string value(int& i, int argc, char** argv) {
  if (++i >= argc) throw std::invalid_argument("missing option value");
  return argv[i];
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model") o.model = value(i, argc, argv);
    else if (a == "--manifest") o.manifest = value(i, argc, argv);
    else if (a == "--labels") o.labels = value(i, argc, argv);
    else if (a == "--family") o.family = value(i, argc, argv);
    else if (a == "--nv12") o.nv12 = value(i, argc, argv);
    else if (a == "--output") o.output = value(i, argc, argv);
    else if (a == "--perf-detail") o.perf_detail = value(i, argc, argv);
    else if (a == "--mode") o.mode = value(i, argc, argv);
    else if (a == "--input-mode") o.input_mode = value(i, argc, argv);
    else if (a == "--width") o.width = std::stoi(value(i, argc, argv));
    else if (a == "--height") o.height = std::stoi(value(i, argc, argv));
    else if (a == "--contexts") o.contexts = std::stoi(value(i, argc, argv));
    else if (a == "--warmup") o.warmup = std::stoi(value(i, argc, argv));
    else if (a == "--iterations") o.iterations = std::stoull(value(i, argc, argv));
    else if (a == "--seconds") o.seconds = std::stod(value(i, argc, argv));
    else throw std::invalid_argument("unknown option: " + a);
  }
  if (o.model.empty() || o.manifest.empty() || o.labels.empty() || o.family.empty() || o.nv12.empty())
    throw std::invalid_argument("--model --manifest --labels --family --nv12 are required");
  if (o.mode != "latency" && o.mode != "throughput") throw std::invalid_argument("mode must be latency|throughput");
  if (o.input_mode != "dma-replay" && o.input_mode != "cpu-copy")
    throw std::invalid_argument("input-mode must be dma-replay|cpu-copy");
  o.contexts = o.mode == "latency" ? 1 : 3;
  return o;
}

int allocate_dma(size_t size) {
  const char* heaps[] = {"/dev/dma_heap/cma", "/dev/dma_heap/system-uncached", "/dev/dma_heap/system"};
  for (const char* path : heaps) {
    const int heap = open(path, O_RDWR | O_CLOEXEC);
    if (heap < 0) continue;
    DmaHeapAllocationData request{};
    request.len = size; request.fd_flags = O_RDWR | O_CLOEXEC;
    const int result = ioctl(heap, RK_BENCH_DMA_HEAP_IOC_ALLOC, &request);
    close(heap);
    if (!result) return static_cast<int>(request.fd);
  }
  throw std::runtime_error("no usable DMA heap");
}

struct DmaFrame {
  int fd = -1; void* mapping = nullptr; size_t size = 0;
  std::vector<unsigned char> source;
  ~DmaFrame() { if (mapping) munmap(mapping, size); if (fd >= 0) close(fd); }
};

std::shared_ptr<DmaFrame> load_frame(const Options& o) {
  const size_t bytes = static_cast<size_t>(o.width) * o.height * 3 / 2;
  auto frame = std::make_shared<DmaFrame>();
  frame->source.resize(bytes);
  std::ifstream input(o.nv12, std::ios::binary);
  if (!input || !input.read(reinterpret_cast<char*>(frame->source.data()), bytes))
    throw std::runtime_error("NV12 input must contain one complete frame");
  frame->size = bytes; frame->fd = allocate_dma(bytes);
  frame->mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, frame->fd, 0);
  if (frame->mapping == MAP_FAILED) { frame->mapping = nullptr; throw std::runtime_error("DMA mmap failed"); }
  std::memcpy(frame->mapping, frame->source.data(), bytes);  // preload, outside timed DMA path
  return frame;
}

rkvs::ModelFamily family(const std::string& name) {
  if (name == "yolov5") return rkvs::ModelFamily::kYolov5;
  if (name == "yolov8") return rkvs::ModelFamily::kYolov8;
  if (name == "yolo11") return rkvs::ModelFamily::kYolo11;
  if (name == "yolo26") return rkvs::ModelFamily::kYolo26;
  throw std::invalid_argument("family must be yolov5|yolov8|yolo11|yolo26");
}

rkvs::SharedFrame ref(const Options& o, const std::shared_ptr<DmaFrame>& dma, uint64_t sequence) {
  auto frame = std::make_shared<rkvs::FrameRef>();
  frame->source_id = "dma-replay"; frame->sequence = sequence;
  frame->capture_ts_ns = frame->receive_ts_ns = rkvs::monotonic_ns();
  frame->format = rkvs::PixelFormat::kNv12; frame->width = o.width; frame->height = o.height;
  frame->planes.push_back({dma->fd, 0, static_cast<uint32_t>(o.width), static_cast<uint32_t>(dma->size)});
  return frame;
}

}  // namespace

int main(int argc, char** argv) try {
  const Options o = parse(argc, argv);
  if (!o.perf_detail.empty()) setenv("RKVS_COLLECT_PERF", "1", 1);
  rkvs::ModelConfig config;
  config.path = o.model; config.manifest = o.manifest; config.labels = o.labels;
  config.family = family(o.family); config.contexts = o.contexts;
  rkvs::RknnDetectorPool detector(config);
  std::vector<std::shared_ptr<DmaFrame>> dmas;
  for (int context = 0; context < o.contexts; ++context) dmas.push_back(load_frame(o));
  for (int context = 0; context < o.contexts; ++context)
    for (int i = 0; i < o.warmup; ++i) detector.process(ref(o, dmas[context], i), context);

  rusage usage_before{};
  getrusage(RUSAGE_SELF, &usage_before);

  std::ofstream csv(o.output);
  if (!csv) throw std::runtime_error("cannot open output CSV");
  csv << "sequence,context,input_mode,copy_ms,queue_ms,rga_ms,inference_ms,output_sync_ms,postprocess_ms,e2e_ms,detections\n";
  std::mutex output_mutex;
  std::atomic<uint64_t> sequence{0};
  const auto started = std::chrono::steady_clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
  auto worker = [&](int context) {
    const auto& dma = dmas[context];
    while (sequence.load() < o.iterations || elapsed() < o.seconds) {
      const uint64_t id = sequence.fetch_add(1);
      double copy_ms = 0;
      if (o.input_mode == "cpu-copy") {
        const auto before = rkvs::monotonic_ns();
        std::memcpy(dma->mapping, dma->source.data(), dma->size);
        copy_ms = static_cast<double>(rkvs::monotonic_ns() - before) / 1e6;
      }
      auto result = detector.process(ref(o, dma, id), context);
      const auto& t = result.timings;
      std::lock_guard<std::mutex> lock(output_mutex);
      csv << id << ',' << context << ',' << o.input_mode << ',' << copy_ms << ',' << t.queue_ms << ','
          << t.rga_ms << ',' << t.inference_ms << ',' << t.output_sync_ms << ',' << t.postprocess_ms << ','
          << t.capture_to_result_ms + copy_ms << ',' << result.detections.size() << '\n';
    }
  };
  std::vector<std::thread> threads;
  for (int context = 0; context < o.contexts; ++context) threads.emplace_back(worker, context);
  for (auto& thread : threads) thread.join();
  const double duration = elapsed();
  rusage usage_after{};
  getrusage(RUSAGE_SELF, &usage_after);
  auto tv_seconds = [](const timeval& value) {
    return static_cast<double>(value.tv_sec) + value.tv_usec / 1e6;
  };
  const double user_s = tv_seconds(usage_after.ru_utime) - tv_seconds(usage_before.ru_utime);
  const double sys_s = tv_seconds(usage_after.ru_stime) - tv_seconds(usage_before.ru_stime);
  if (!o.perf_detail.empty()) {
    std::ofstream perf(o.perf_detail);
    if (!perf) throw std::runtime_error("cannot open perf-detail output");
    perf << detector.perf_detail(0);
  }
  std::cout << "frames=" << sequence.load() << " seconds=" << duration
            << " fps=" << sequence.load() / duration << " contexts=" << o.contexts
            << " cores=" << (o.contexts == 1 ? "0" : "0,1,2")
            << " cpu_user_s=" << user_s << " cpu_sys_s=" << sys_s
            << " cpu_pct=" << 100.0 * (user_s + sys_s) / duration << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "benchmark failed: " << error.what() << '\n';
  return 2;
}
