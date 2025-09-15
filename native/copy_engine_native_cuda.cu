#include "copy_engine_common.hpp"

#ifdef USE_CUDA_BACKEND

#include <cuda_runtime.h>

struct CudaBackend {
  using stream_t = cudaStream_t;
  std::vector<std::vector<stream_t>> streams_; // [device][stream_id]

  void init_device_streams(int device, int streams_per_dev) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device < 0 || device >= device_count) {
      throw std::runtime_error("invalid CUDA device id");
    }
    cudaSetDevice(device);
    streams_.resize(device + 1);
    auto& vec = streams_[device];
    vec.resize(streams_per_dev);
    for (int i = 0; i < streams_per_dev; ++i) {
      cudaStreamCreateWithFlags(&vec[i], cudaStreamNonBlocking);
    }
  }

  stream_t get_stream(int device, int stream_id) {
    if (stream_id < 0) stream_id = 0;
    auto& vec = streams_.at(device);
    if (vec.empty()) throw std::runtime_error("streams not initialized");
    return vec[stream_id % static_cast<int>(vec.size())];
  }

  void* alloc_pinned(size_t bytes) {
    void* p = nullptr;
    cudaError_t st = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    if (st != cudaSuccess) return nullptr;
    return p;
  }

  void free_pinned(void* p) { cudaFreeHost(p); }

  void memcpy_h2d_async(int device, void* dst_device, const void* src_host, size_t bytes, stream_t s) {
    cudaSetDevice(device);
    cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, s);
  }

  void record_event(stream_t s, void** out_event) {
    cudaEvent_t ev;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, s);
    *out_event = reinterpret_cast<void*>(ev);
  }

  bool event_completed(void* event) {
    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(event);
    cudaError_t q = cudaEventQuery(ev);
    return q == cudaSuccess;
  }

  void destroy_event(void* event) {
    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(event);
    cudaEventDestroy(ev);
  }
};

using CopyEngineCuda = CopyEngineNative<CudaBackend>;

PYBIND11_MODULE(bodocache_agent_copy_engine, m) {
  py::class_<CopyEngineCuda>(m, "CopyEngine")
      .def(py::init<int, int>(), py::arg("device_id") = 0, py::arg("streams_per_device") = 4)
      .def("acquire_host_buffer", &CopyEngineCuda::acquire_host_buffer, py::arg("bytes"))
      .def("submit", &CopyEngineCuda::submit, py::arg("ops"), py::arg("callback"));
}

#endif // USE_CUDA_BACKEND

