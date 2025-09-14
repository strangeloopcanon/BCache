#pragma once

#include <cstdint>
#include <functional>
#include <vector>

struct PagePtr { void* ptr; uint32_t bytes; };
struct Ticket { uint64_t id; };

class EngineShim {
 public:
  using ReadyCb = std::function<void(Ticket, const std::vector<PagePtr>&)>;

  Ticket request_pages(const char* prefix_id, const int* layers, int n_layers, int64_t deadline_ms);
  void on_pages_ready(ReadyCb cb);
  void return_pages(const std::vector<PagePtr>& pages);
};

