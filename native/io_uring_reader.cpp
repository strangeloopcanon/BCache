#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <liburing.h>

namespace py = pybind11;

static ssize_t read_range_into(const std::string& path, uint64_t offset, size_t size, py::object out_buf) {
  if (size == 0) return 0;
  // Get writable buffer
  void* dst = nullptr; size_t nbytes = 0;
  if (PyMemoryView_Check(out_buf.ptr())) {
    py::memoryview mv = py::reinterpret_borrow<py::memoryview>(out_buf);
    if (mv.readonly()) throw std::runtime_error("buffer must be writable");
    dst = mv.data(); nbytes = mv.nbytes();
  } else if (PyObject_CheckBuffer(out_buf.ptr())) {
    py::buffer buf = py::reinterpret_borrow<py::buffer>(out_buf);
    py::buffer_info info = buf.request();
    dst = info.ptr; nbytes = (size_t)info.size * (size_t)info.itemsize;
  } else {
    throw std::runtime_error("out_buf must support buffer protocol");
  }
  if (nbytes < size) throw std::runtime_error("out_buf too small");

  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open failed: " + std::string(strerror(errno)));

  io_uring ring;
  if (io_uring_queue_init(16, &ring, 0) < 0) {
    ::close(fd);
    throw std::runtime_error("io_uring_queue_init failed");
  }

  size_t submitted = 0;
  char* p = reinterpret_cast<char*>(dst);
  const size_t chunk = 1 << 20; // 1MB chunks
  size_t remaining = size;
  uint64_t off = offset;

  while (remaining > 0) {
    size_t to_read = remaining < chunk ? remaining : chunk;
    io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
      io_uring_submit(&ring);
      continue;
    }
    io_uring_prep_read(sqe, fd, p + submitted, to_read, off);
    io_uring_sqe_set_data64(sqe, submitted + to_read);
    io_uring_submit(&ring);

    // Wait for completion of this chunk before queuing more to keep it simple
    io_uring_cqe* cqe = nullptr;
    int ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret < 0) {
      io_uring_queue_exit(&ring); ::close(fd);
      throw std::runtime_error("io_uring_wait_cqe failed");
    }
    if (cqe->res < 0) {
      int err = -cqe->res;
      io_uring_cqe_seen(&ring, cqe);
      io_uring_queue_exit(&ring); ::close(fd);
      throw std::runtime_error("read failed: errno " + std::to_string(err));
    }
    submitted += static_cast<size_t>(cqe->res);
    io_uring_cqe_seen(&ring, cqe);
    remaining -= to_read;
    off += to_read;
  }

  io_uring_queue_exit(&ring);
  ::close(fd);
  return static_cast<ssize_t>(submitted);
}

PYBIND11_MODULE(bodocache_agent_io_uring, m) {
  m.def("read_range_into", &read_range_into, py::arg("path"), py::arg("offset"), py::arg("size"), py::arg("out_buf"));
}

