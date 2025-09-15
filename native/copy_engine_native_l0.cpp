#include "copy_engine_common.hpp"

#ifdef USE_L0_BACKEND

#include <level_zero/ze_api.h>

struct L0Backend {
  using stream_t = ze_command_queue_handle_t;
  ze_context_handle_t context_{nullptr};
  ze_device_handle_t device_{nullptr};
  ze_event_pool_handle_t event_pool_{nullptr};
  std::vector<stream_t> queues_;

  void init_device_streams(int device_index, int streams_per_dev) {
    zeInit(0);
    uint32_t nDrivers = 0;
    zeDriverGet(&nDrivers, nullptr);
    if (nDrivers == 0) throw std::runtime_error("No Level Zero drivers found");
    std::vector<ze_driver_handle_t> drivers(nDrivers);
    zeDriverGet(&nDrivers, drivers.data());

    // Choose first driver; enumerate devices
    uint32_t nDevices = 0;
    zeDeviceGet(drivers[0], &nDevices, nullptr);
    if (nDevices == 0) throw std::runtime_error("No Level Zero devices found");
    std::vector<ze_device_handle_t> devices(nDevices);
    zeDeviceGet(drivers[0], &nDevices, devices.data());
    if (device_index < 0 || (uint32_t)device_index >= nDevices) throw std::runtime_error("Invalid device index");
    device_ = devices[device_index];

    ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    zeContextCreate(drivers[0], &cdesc, &context_);

    // Create command queues (streams)
    queues_.resize(streams_per_dev);
    for (int i = 0; i < streams_per_dev; ++i) {
      ze_command_queue_desc_t qdesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
      qdesc.ordinal = 0;
      qdesc.flags = 0;
      qdesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
      qdesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
      zeCommandQueueCreate(context_, device_, &qdesc, &queues_[i]);
    }

    // Create an event pool
    ze_event_pool_desc_t pdesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    pdesc.count = 1024;
    pdesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    zeCommandListCreate(context_, device_, nullptr, nullptr); // warmup
    zeEventPoolCreate(context_, &pdesc, 0, nullptr, &event_pool_);
  }

  stream_t get_stream(int /*device*/, int stream_id) {
    if (queues_.empty()) throw std::runtime_error("queues not initialized");
    if (stream_id < 0) stream_id = 0;
    return queues_[stream_id % static_cast<int>(queues_.size())];
  }

  void* alloc_pinned(size_t bytes) {
    ze_host_mem_alloc_desc_t hdesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    void* p = nullptr;
    ze_result_t r = zeMemAllocHost(context_, &hdesc, bytes, /*alignment*/ 64, &p);
    if (r != ZE_RESULT_SUCCESS) return nullptr;
    return p;
  }

  void free_pinned(void* p) { zeMemFree(context_, p); }

  void memcpy_h2d_async(int /*device*/, void* dst_device, const void* src_host, size_t bytes, stream_t q) {
    ze_command_list_handle_t cl;
    ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    zeCommandListCreate(context_, device_, &ldesc, &cl);
    zeEventHostSynchronize(nullptr, 0); // no-op safety
    zeCommandListAppendMemoryCopy(cl, dst_device, src_host, bytes, /*signal*/ nullptr, 0, nullptr);
    zeCommandListClose(cl);
    zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr);
    // We don't synchronize here; events handled separately via record_event()
    // Hold the command list reference implicitly until event signals; user should flush queue
    zeCommandListDestroy(cl); // Allow driver to reclaim after execution
  }

  void record_event(stream_t q, void** out_event) {
    ze_event_handle_t ev;
    ze_event_desc_t edesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
    static std::atomic<uint32_t> next{1};
    edesc.index = next.fetch_add(1);
    edesc.signal = 0;
    edesc.wait = 0;
    zeEventCreate(event_pool_, &edesc, &ev);
    // Emit a barrier to signal the event after prior enqueued ops
    ze_command_list_handle_t cl;
    ze_command_list_desc_t ldesc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    zeCommandListCreate(context_, device_, &ldesc, &cl);
    zeCommandListAppendBarrier(cl, ev, 0, nullptr);
    zeCommandListClose(cl);
    zeCommandQueueExecuteCommandLists(q, 1, &cl, nullptr);
    zeCommandListDestroy(cl);
    *out_event = reinterpret_cast<void*>(ev);
  }

  bool event_completed(void* event) {
    ze_event_handle_t ev = reinterpret_cast<ze_event_handle_t>(event);
    ze_result_t r = zeEventQueryStatus(ev);
    return r == ZE_RESULT_SUCCESS;
  }

  void destroy_event(void* event) {
    ze_event_handle_t ev = reinterpret_cast<ze_event_handle_t>(event);
    zeEventDestroy(ev);
  }
};

using CopyEngineL0 = CopyEngineNative<L0Backend>;

PYBIND11_MODULE(bodocache_agent_copy_engine, m) {
  py::class_<CopyEngineL0>(m, "CopyEngine")
      .def(py::init<int, int>(), py::arg("device_id") = 0, py::arg("streams_per_device") = 4)
      .def("acquire_host_buffer", &CopyEngineL0::acquire_host_buffer, py::arg("bytes"))
      .def("submit", &CopyEngineL0::submit, py::arg("ops"), py::arg("callback"));
}

#endif // USE_L0_BACKEND

