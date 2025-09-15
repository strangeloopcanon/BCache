#include "copy_engine_common.hpp"

#ifdef USE_HIP_BACKEND

#include <hip/hip_runtime.h>

struct HipBackend {
  using stream_t = hipStream_t;
  std::vector<std::vector<stream_t>> streams_;

  void init_device_streams(int device, int streams_per_dev) {
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device < 0 || device >= device_count) {
      throw std::runtime_error("invalid HIP device id");
    }
    hipSetDevice(device);
    streams_.resize(device + 1);
    auto& vec = streams_[device];
    vec.resize(streams_per_dev);
    for (int i = 0; i < streams_per_dev; ++i) {
      hipStreamCreateWithFlags(&vec[i], hipStreamNonBlocking);
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
    hipError_t st = hipHostMalloc(&p, bytes, hipHostMallocDefault);
    if (st != hipSuccess) return nullptr;
    return p;
  }

  void free_pinned(void* p) { hipHostFree(p); }

  void memcpy_h2d_async(int device, void* dst_device, const void* src_host, size_t bytes, stream_t s) {
    hipSetDevice(device);
    hipMemcpyAsync(dst_device, src_host, bytes, hipMemcpyHostToDevice, s);
  }

  void record_event(stream_t s, void** out_event) {
    hipEvent_t ev;
    hipEventCreateWithFlags(&ev, hipEventDisableTiming);
    hipEventRecord(ev, s);
    *out_event = reinterpret_cast<void*>(ev);
  }

  bool event_completed(void* event) {
    hipEvent_t ev = reinterpret_cast<hipEvent_t>(event);
    hipError_t q = hipEventQuery(ev);
    return q == hipSuccess;
  }

  void destroy_event(void* event) {
    hipEvent_t ev = reinterpret_cast<hipEvent_t>(event);
    hipEventDestroy(ev);
  }
};

using CopyEngineHip = CopyEngineNative<HipBackend>;

PYBIND11_MODULE(bodocache_agent_copy_engine, m) {
  py::class_<CopyEngineHip>(m, "CopyEngine")
      .def(py::init<int, int>(), py::arg("device_id") = 0, py::arg("streams_per_device") = 4)
      .def("acquire_host_buffer", &CopyEngineHip::acquire_host_buffer, py::arg("bytes"))
      .def("submit", &CopyEngineHip::submit, py::arg("ops"), py::arg("callback"));
}

#endif // USE_HIP_BACKEND

