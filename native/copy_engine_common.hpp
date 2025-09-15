#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/pytypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct HostBuf {
  void* ptr{nullptr};
  size_t bytes{0};
};

inline void* capsule_to_ptr(const py::object& obj) {
  if (py::isinstance<py::capsule>(obj)) {
    py::capsule cap = obj.cast<py::capsule>();
    return cap.get_pointer();
  }
  // Allow int addresses as a fallback
  if (py::isinstance<py::int_>(obj)) {
    uintptr_t addr = obj.cast<uintptr_t>();
    return reinterpret_cast<void*>(addr);
  }
  // Allow objects exposing a __int__ or __index__ (e.g., torch Tensor.data_ptr())
  try {
    uintptr_t addr = obj.attr("__int__")().cast<uintptr_t>();
    return reinterpret_cast<void*>(addr);
  } catch (...) {}
  return nullptr;
}

inline bool get_bytes_view(const py::handle& h, void** out_ptr, size_t* out_size) {
  // Accept memoryview or any object supporting buffer protocol
  if (PyMemoryView_Check(h.ptr())) {
    py::memoryview mv = py::reinterpret_borrow<py::memoryview>(h);
    if (!mv || mv.readonly()) return false;
    *out_ptr = mv.data();
    *out_size = mv.nbytes();
    return true;
  }
  if (PyObject_CheckBuffer(h.ptr())) {
    py::buffer buf = py::reinterpret_borrow<py::buffer>(h);
    py::buffer_info info = buf.request();
    *out_ptr = info.ptr;
    *out_size = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
    return true;
  }
  // Fallback: if it's bytes, we need to copy into a temp pinned buf outside
  if (PyBytes_Check(h.ptr())) {
    char* data;
    Py_ssize_t len;
    if (PyBytes_AsStringAndSize(h.ptr(), &data, &len) == 0) {
      *out_ptr = data;
      *out_size = static_cast<size_t>(len);
      return true;
    }
  }
  return false;
}

// Backend-agnostic engine needs to provide these functions/types:
// - using stream_t
// - void init_device_streams(int device, int streams_per_dev)
// - stream_t get_stream(int device, int stream_id)
// - void* alloc_pinned(size_t bytes)
// - void free_pinned(void*)
// - void memcpy_h2d_async(int device, void* dst_device, const void* src_host, size_t bytes, stream_t)
// - void record_event(stream_t, void** out_event)
// - bool event_completed(void* event)
// - void destroy_event(void* event)

struct PendingOp {
  int device{0};
  void* dst_device{nullptr};
  void* src_host{nullptr};
  size_t bytes{0};
  int stream_id{0};
  int64_t deadline_ms{0};
  void* event{nullptr};
};

template <typename Backend>
class CopyEngineNative {
 public:
  CopyEngineNative(int device_id, int streams_per_device)
      : device_(device_id), streams_per_dev_(streams_per_device) {
    backend_.init_device_streams(device_, streams_per_dev_);
  }

  ~CopyEngineNative() { stop_worker(); }

  py::memoryview acquire_host_buffer(size_t bytes) {
    void* p = backend_.alloc_pinned(bytes);
    if (!p) throw std::bad_alloc();
    std::lock_guard<std::mutex> g(mu_);
    live_buffers_.push_back(p);
    // Expose as writable 1D uint8 buffer
    return py::memoryview(py::buffer_info(
        p, sizeof(uint8_t), py::format_descriptor<uint8_t>::format(), 1, {bytes}, {sizeof(uint8_t)}));
  }

  void submit(py::list ops, py::function callback) {
    std::vector<PendingOp> batch;
    batch.reserve(py::len(ops));

    for (auto item : ops) {
      py::object op = py::reinterpret_borrow<py::object>(item);
      // Extract fields by attribute or mapping
      auto get_attr = [&](const char* name) -> py::object {
        if (PyObject_HasAttrString(op.ptr(), name)) return op.attr(name);
        if (PyMapping_Check(op.ptr())) return op[name];
        throw std::runtime_error(std::string("missing field ") + name);
      };
      py::object src_obj = get_attr("src");
      py::object dst_obj = get_attr("dst");
      size_t bytes = get_attr("bytes").cast<size_t>();
      int stream_id = 0;
      if (PyObject_HasAttrString(op.ptr(), "stream_id")) stream_id = op.attr("stream_id").cast<int>();
      int device = 0;
      if (PyObject_HasAttrString(op.ptr(), "gpu_id")) device = op.attr("gpu_id").cast<int>();
      int64_t deadline_ms = 0;
      if (PyObject_HasAttrString(op.ptr(), "deadline_ms")) deadline_ms = op.attr("deadline_ms").cast<int64_t>();

      void* src_ptr = nullptr;
      size_t src_size = 0;
      if (!get_bytes_view(src_obj, &src_ptr, &src_size)) {
        throw std::runtime_error("src must be a writable buffer or bytes");
      }
      if (src_size < bytes) {
        throw std::runtime_error("src buffer too small");
      }
      void* dst_ptr = capsule_to_ptr(dst_obj);
      if (!dst_ptr) throw std::runtime_error("dst must be a device pointer capsule or int address");

      PendingOp po;
      po.device = device;
      po.dst_device = dst_ptr;
      po.src_host = src_ptr;
      po.bytes = bytes;
      po.stream_id = stream_id;
      po.deadline_ms = deadline_ms;
      batch.push_back(po);
    }

    // Submit copies
    for (auto& po : batch) {
      auto stream = backend_.get_stream(po.device, po.stream_id);
      backend_.memcpy_h2d_async(po.device, po.dst_device, po.src_host, po.bytes, stream);
      backend_.record_event(stream, &po.event);
    }

    // Start worker thread if not running
    ensure_worker();

    {
      std::lock_guard<std::mutex> g(mu_);
      for (auto& po : batch) pending_.push_back(std::move(po));
      // Update the active callback (single callback used for all ops)
      active_callback_ = callback;
    }
  }

 private:
  void ensure_worker() {
    bool expected = false;
    if (running_.compare_exchange_strong(expected, true)) {
      worker_ = std::thread([this]() { this->worker_loop(); });
    }
  }

  void stop_worker() {
    bool was_running = running_.exchange(false);
    if (was_running && worker_.joinable()) worker_.join();
  }

  void worker_loop() {
    while (running_.load()) {
      std::vector<PendingOp> done;
      py::function cb;
      {
        std::lock_guard<std::mutex> g(mu_);
        for (auto it = pending_.begin(); it != pending_.end();) {
          if (backend_.event_completed(it->event)) {
            done.push_back(*it);
            backend_.destroy_event(it->event);
            it = pending_.erase(it);
          } else {
            ++it;
          }
        }
        if (!active_callback_.is_none()) cb = active_callback_.cast<py::function>();
      }

      // Fire callbacks and free host buffers (if we own them)
      if (!done.empty()) {
        // Identify any engine-owned host buffers to free
        {
          std::lock_guard<std::mutex> g(mu_);
          for (auto& po : done) {
            // If src_host is one of our live buffers, free it
            auto it = std::find(live_buffers_.begin(), live_buffers_.end(), po.src_host);
            if (it != live_buffers_.end()) {
              backend_.free_pinned(po.src_host);
              live_buffers_.erase(it);
            }
          }
        }

        if (cb) {
          py::gil_scoped_acquire ag;
          for (auto& po : done) {
            try {
              py::dict info;
              info["gpu_id"] = po.device;
              info["bytes"] = py::int_(po.bytes);
              info["deadline_ms"] = py::int_(po.deadline_ms);
              cb(info);
            } catch (...) {
              // Swallow exceptions to keep worker alive
            }
          }
        }
      }

      // Sleep a bit
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  Backend backend_{};
  int device_{0};
  int streams_per_dev_{4};
  std::atomic<bool> running_{false};
  std::thread worker_{};
  std::mutex mu_;
  std::vector<void*> live_buffers_;
  std::vector<PendingOp> pending_;
  py::object active_callback_ = py::none();
};
