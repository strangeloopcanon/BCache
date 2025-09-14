#include "engine_shim.h"

Ticket EngineShim::request_pages(const char* /*prefix_id*/, const int* /*layers*/, int /*n_layers*/, int64_t /*deadline_ms*/) {
  static uint64_t next_id = 1;
  return Ticket{next_id++};
}

void EngineShim::on_pages_ready(ReadyCb /*cb*/) {}

void EngineShim::return_pages(const std::vector<PagePtr>& /*pages*/) {}

