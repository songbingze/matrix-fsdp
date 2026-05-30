#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>

template <typename scalar_t>
__global__ void copy_rank_segments_to_full_kernel(
    const scalar_t* __restrict__ local_tensor,
    scalar_t* __restrict__ output_tensor,
    const int64_t* __restrict__ global_starts,
    const int64_t* __restrict__ local_starts,
    const int64_t* __restrict__ numels) {
  const int64_t segment_idx = blockIdx.y;
  const int64_t global_start = global_starts[segment_idx];
  const int64_t local_start = local_starts[segment_idx];
  const int64_t numel = numels[segment_idx];
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset < numel) {
    output_tensor[global_start + offset] = local_tensor[local_start + offset];
  }
}

template <typename scalar_t>
__global__ void copy_range_kernel(
    const scalar_t* __restrict__ input_tensor,
    scalar_t* __restrict__ output_tensor,
    const int64_t input_offset,
    const int64_t numel) {
  const int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;
  for (int64_t offset = linear_idx; offset < numel; offset += stride) {
    output_tensor[offset] = input_tensor[input_offset + offset];
  }
}

void copy_rank_segments_to_full_cuda(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor global_starts,
    torch::Tensor local_starts,
    torch::Tensor numels,
    int64_t max_numel) {
  const int threads = 256;
  const int64_t num_segments = global_starts.numel();
  const int64_t tiles = (max_numel + threads - 1) / threads;
  if (num_segments == 0 || tiles == 0) {
    return;
  }
  const dim3 blocks(static_cast<unsigned int>(tiles), static_cast<unsigned int>(num_segments));
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      local_tensor.scalar_type(),
      "copy_rank_segments_to_full_cuda",
      [&] {
        copy_rank_segments_to_full_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            local_tensor.data_ptr<scalar_t>(),
            output_tensor.data_ptr<scalar_t>(),
            global_starts.data_ptr<int64_t>(),
            local_starts.data_ptr<int64_t>(),
            numels.data_ptr<int64_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_range_cuda(torch::Tensor input_tensor, torch::Tensor output_tensor, int64_t input_offset) {
  const int threads = 256;
  const int64_t numel = output_tensor.numel();
  const int blocks = static_cast<int>(std::min<int64_t>((numel + threads - 1) / threads, 4096));
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      input_tensor.scalar_type(),
      "copy_range_cuda",
      [&] {
        copy_range_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input_tensor.data_ptr<scalar_t>(),
            output_tensor.data_ptr<scalar_t>(),
            input_offset,
            numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
