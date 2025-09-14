#include "pinned_pool.h"

#include <cstdlib>

// Stubs; replace with cudaHostAlloc for pinned memory in production.

PinnedPool::PinnedPool(size_t total_bytes, size_t block_bytes)
    : total_(total_bytes), block_(block_bytes) {
  size_t n = total_bytes / block_bytes;
  free_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    void* p = std::malloc(block_bytes);
    free_.push_back({p, block_bytes});
  }
}

PinnedPool::~PinnedPool() {
  for (auto& b : free_) std::free(b.ptr);
}

PinnedBlock PinnedPool::acquire() {
  if (free_.empty()) return {nullptr, 0};
  PinnedBlock b = free_.back();
  free_.pop_back();
  return b;
}

void PinnedPool::release(PinnedBlock blk) { free_.push_back(blk); }

