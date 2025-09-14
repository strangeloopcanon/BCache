// Stub CUDA kernel placeholder for page-first layout gather/scatter.
extern "C" __global__ void page_first_gather(const float* __restrict__ src,
                                             float* __restrict__ dst,
                                             int stride_src, int stride_dst,
                                             int page_elems) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < page_elems) {
    dst[idx] = src[idx];
  }
}

