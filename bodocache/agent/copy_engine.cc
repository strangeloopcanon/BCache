#include "copy_engine.h"

#include <chrono>
#include <thread>

void CopyEngine::submit(const std::vector<CopyOpC>& ops, Callback cb) {
  // Simulate async completion.
  for (auto& op : ops) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    cb(op);
  }
}

