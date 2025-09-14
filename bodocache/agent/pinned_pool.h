#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Stub interface for pinned host memory ring buffers.
struct PinnedBlock {
  void* ptr{nullptr};
  size_t bytes{0};
};

class PinnedPool {
 public:
  explicit PinnedPool(size_t total_bytes, size_t block_bytes);
  ~PinnedPool();

  PinnedPool(const PinnedPool&) = delete;
  PinnedPool& operator=(const PinnedPool&) = delete;

  PinnedBlock acquire();
  void release(PinnedBlock blk);

 private:
  size_t total_{};
  size_t block_{};
  std::vector<PinnedBlock> free_;
};

