#pragma once

#include <cstdint>
#include <functional>
#include <vector>

struct CopyOpC {
  const void* src;
  void* dst;
  uint32_t bytes;
  int stream_id;
  int gpu_id;
  int64_t deadline_ms;
};

// Stub copy engine. Production uses cudaMemcpyAsync on streams.
class CopyEngine {
 public:
  using Callback = std::function<void(const CopyOpC&)>;

  void submit(const std::vector<CopyOpC>& ops, Callback cb);
};

