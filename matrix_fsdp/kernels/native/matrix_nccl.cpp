#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <nccl.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

namespace py = pybind11;

namespace {

ncclComm_t g_comm = nullptr;
int g_rank = -1;
int g_world_size = -1;

#define MATRIX_NCCL_CHECK(cmd)                                  \
  do {                                                          \
    ncclResult_t result = (cmd);                                \
    TORCH_CHECK(result == ncclSuccess, ncclGetErrorString(result)); \
  } while (0)

#define MATRIX_CHECK_CUDA(tensor) TORCH_CHECK((tensor).is_cuda(), #tensor " must be a CUDA tensor")
#define MATRIX_CHECK_CPU(tensor) TORCH_CHECK(!(tensor).is_cuda(), #tensor " must be a CPU tensor")
#define MATRIX_CHECK_CONTIGUOUS(tensor) TORCH_CHECK((tensor).is_contiguous(), #tensor " must be contiguous")
#define MATRIX_CHECK_INT64(tensor) TORCH_CHECK((tensor).scalar_type() == at::ScalarType::Long, #tensor " must be int64")

ncclDataType_t nccl_dtype_for(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return ncclFloat16;
    case torch::kBFloat16:
      return ncclBfloat16;
    case torch::kFloat32:
      return ncclFloat32;
    case torch::kFloat64:
      return ncclFloat64;
    case torch::kInt32:
      return ncclInt32;
    case torch::kInt64:
      return ncclInt64;
    default:
      TORCH_CHECK(false, "Unsupported NCCL dtype for MatrixFSDP native fused broadcast");
  }
}

}  // namespace

py::bytes get_nccl_unique_id() {
  ncclUniqueId unique_id;
  MATRIX_NCCL_CHECK(ncclGetUniqueId(&unique_id));
  return py::bytes(unique_id.internal, NCCL_UNIQUE_ID_BYTES);
}

void init_nccl_comm(py::bytes unique_id_bytes, int rank, int world_size) {
  std::string unique_id_string = unique_id_bytes;
  TORCH_CHECK(unique_id_string.size() == NCCL_UNIQUE_ID_BYTES, "Invalid NCCL unique id size");
  if (g_comm != nullptr) {
    MATRIX_NCCL_CHECK(ncclCommDestroy(g_comm));
    g_comm = nullptr;
  }
  ncclUniqueId unique_id;
  std::memcpy(unique_id.internal, unique_id_string.data(), NCCL_UNIQUE_ID_BYTES);
  MATRIX_NCCL_CHECK(ncclCommInitRank(&g_comm, world_size, unique_id, rank));
  g_rank = rank;
  g_world_size = world_size;
}

void destroy_nccl_comm() {
  if (g_comm != nullptr) {
    MATRIX_NCCL_CHECK(ncclCommDestroy(g_comm));
    g_comm = nullptr;
    g_rank = -1;
    g_world_size = -1;
  }
}

void group_broadcast_rank_segments(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor src_ranks,
    torch::Tensor global_starts,
    torch::Tensor local_starts,
    torch::Tensor numels,
    int rank) {
  TORCH_CHECK(g_comm != nullptr, "MatrixFSDP native NCCL communicator is not initialized");
  MATRIX_CHECK_CUDA(local_tensor);
  MATRIX_CHECK_CUDA(output_tensor);
  MATRIX_CHECK_CPU(src_ranks);
  MATRIX_CHECK_CPU(global_starts);
  MATRIX_CHECK_CPU(local_starts);
  MATRIX_CHECK_CPU(numels);
  MATRIX_CHECK_CONTIGUOUS(local_tensor);
  MATRIX_CHECK_CONTIGUOUS(output_tensor);
  MATRIX_CHECK_CONTIGUOUS(src_ranks);
  MATRIX_CHECK_CONTIGUOUS(global_starts);
  MATRIX_CHECK_CONTIGUOUS(local_starts);
  MATRIX_CHECK_CONTIGUOUS(numels);
  MATRIX_CHECK_INT64(src_ranks);
  MATRIX_CHECK_INT64(global_starts);
  MATRIX_CHECK_INT64(local_starts);
  MATRIX_CHECK_INT64(numels);
  TORCH_CHECK(local_tensor.dim() == 1, "local_tensor must be 1D");
  TORCH_CHECK(output_tensor.dim() == 1, "output_tensor must be 1D");
  TORCH_CHECK(local_tensor.scalar_type() == output_tensor.scalar_type(), "source and destination dtype mismatch");
  TORCH_CHECK(src_ranks.numel() == global_starts.numel(), "metadata length mismatch");
  TORCH_CHECK(src_ranks.numel() == local_starts.numel(), "metadata length mismatch");
  TORCH_CHECK(src_ranks.numel() == numels.numel(), "metadata length mismatch");

  const int64_t range_count = src_ranks.numel();
  if (range_count == 0) {
    return;
  }

  auto dtype = nccl_dtype_for(local_tensor.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto element_size = local_tensor.element_size();
  char* local_base = static_cast<char*>(local_tensor.data_ptr());
  char* output_base = static_cast<char*>(output_tensor.data_ptr());
  const int64_t* src_rank_ptr = src_ranks.data_ptr<int64_t>();
  const int64_t* global_start_ptr = global_starts.data_ptr<int64_t>();
  const int64_t* local_start_ptr = local_starts.data_ptr<int64_t>();
  const int64_t* numel_ptr = numels.data_ptr<int64_t>();

  MATRIX_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < range_count; ++i) {
    const int64_t count = numel_ptr[i];
    if (count == 0) {
      continue;
    }
    const int src_rank = static_cast<int>(src_rank_ptr[i]);
    const int64_t global_start = global_start_ptr[i];
    void* recv_buffer = output_base + global_start * element_size;
    void* send_buffer =
        rank == src_rank ? local_base + local_start_ptr[i] * element_size : recv_buffer;
    MATRIX_NCCL_CHECK(ncclBroadcast(send_buffer, recv_buffer, count, dtype, src_rank, g_comm, stream));
  }
  MATRIX_NCCL_CHECK(ncclGroupEnd());
}

void sendrecv_rank_segments(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor src_ranks,
    torch::Tensor global_starts,
    torch::Tensor local_starts,
    torch::Tensor numels,
    int rank) {
  TORCH_CHECK(g_comm != nullptr, "MatrixFSDP native NCCL communicator is not initialized");
  MATRIX_CHECK_CUDA(local_tensor);
  MATRIX_CHECK_CUDA(output_tensor);
  MATRIX_CHECK_CPU(src_ranks);
  MATRIX_CHECK_CPU(global_starts);
  MATRIX_CHECK_CPU(local_starts);
  MATRIX_CHECK_CPU(numels);
  MATRIX_CHECK_CONTIGUOUS(local_tensor);
  MATRIX_CHECK_CONTIGUOUS(output_tensor);
  MATRIX_CHECK_CONTIGUOUS(src_ranks);
  MATRIX_CHECK_CONTIGUOUS(global_starts);
  MATRIX_CHECK_CONTIGUOUS(local_starts);
  MATRIX_CHECK_CONTIGUOUS(numels);
  MATRIX_CHECK_INT64(src_ranks);
  MATRIX_CHECK_INT64(global_starts);
  MATRIX_CHECK_INT64(local_starts);
  MATRIX_CHECK_INT64(numels);
  TORCH_CHECK(local_tensor.dim() == 1, "local_tensor must be 1D");
  TORCH_CHECK(output_tensor.dim() == 1, "output_tensor must be 1D");
  TORCH_CHECK(local_tensor.scalar_type() == output_tensor.scalar_type(), "source and destination dtype mismatch");
  TORCH_CHECK(src_ranks.numel() == global_starts.numel(), "metadata length mismatch");
  TORCH_CHECK(src_ranks.numel() == local_starts.numel(), "metadata length mismatch");
  TORCH_CHECK(src_ranks.numel() == numels.numel(), "metadata length mismatch");

  const int64_t range_count = src_ranks.numel();
  if (range_count == 0) {
    return;
  }

  auto dtype = nccl_dtype_for(local_tensor.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto element_size = local_tensor.element_size();
  char* local_base = static_cast<char*>(local_tensor.data_ptr());
  char* output_base = static_cast<char*>(output_tensor.data_ptr());
  const int64_t* src_rank_ptr = src_ranks.data_ptr<int64_t>();
  const int64_t* global_start_ptr = global_starts.data_ptr<int64_t>();
  const int64_t* local_start_ptr = local_starts.data_ptr<int64_t>();
  const int64_t* numel_ptr = numels.data_ptr<int64_t>();

  for (int64_t i = 0; i < range_count; ++i) {
    const int64_t count = numel_ptr[i];
    if (count == 0 || static_cast<int>(src_rank_ptr[i]) != rank) {
      continue;
    }
    const int64_t local_start = local_start_ptr[i];
    const int64_t global_start = global_start_ptr[i];
    C10_CUDA_CHECK(cudaMemcpyAsync(
        output_base + global_start * element_size,
        local_base + local_start * element_size,
        count * element_size,
        cudaMemcpyDeviceToDevice,
        stream));
  }

  MATRIX_NCCL_CHECK(ncclGroupStart());
  for (int64_t i = 0; i < range_count; ++i) {
    const int64_t count = numel_ptr[i];
    if (count == 0) {
      continue;
    }
    const int src_rank = static_cast<int>(src_rank_ptr[i]);
    const int64_t global_start = global_start_ptr[i];
    const int64_t local_start = local_start_ptr[i];
    void* recv_buffer = output_base + global_start * element_size;
    void* send_buffer = local_base + local_start * element_size;
    if (rank == src_rank) {
      for (int peer = 0; peer < g_world_size; ++peer) {
        if (peer == rank) {
          continue;
        }
        MATRIX_NCCL_CHECK(ncclSend(send_buffer, count, dtype, peer, g_comm, stream));
      }
    } else {
      MATRIX_NCCL_CHECK(ncclRecv(recv_buffer, count, dtype, src_rank, g_comm, stream));
    }
  }
  MATRIX_NCCL_CHECK(ncclGroupEnd());
}

void sendrecv_rank_chunks(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor shard_sizes,
    int rank) {
  TORCH_CHECK(g_comm != nullptr, "MatrixFSDP native NCCL communicator is not initialized");
  MATRIX_CHECK_CUDA(local_tensor);
  MATRIX_CHECK_CUDA(output_tensor);
  MATRIX_CHECK_CPU(shard_sizes);
  MATRIX_CHECK_CONTIGUOUS(local_tensor);
  MATRIX_CHECK_CONTIGUOUS(output_tensor);
  MATRIX_CHECK_CONTIGUOUS(shard_sizes);
  MATRIX_CHECK_INT64(shard_sizes);
  TORCH_CHECK(local_tensor.dim() == 1, "local_tensor must be 1D");
  TORCH_CHECK(output_tensor.dim() == 1, "output_tensor must be 1D");
  TORCH_CHECK(local_tensor.scalar_type() == output_tensor.scalar_type(), "source and destination dtype mismatch");
  TORCH_CHECK(shard_sizes.numel() == g_world_size, "shard_sizes length must match NCCL world size");
  TORCH_CHECK(rank == g_rank, "rank must match initialized native NCCL rank");
  TORCH_CHECK(rank >= 0 && rank < g_world_size, "rank out of range");

  const int64_t* shard_size_ptr = shard_sizes.data_ptr<int64_t>();
  std::vector<int64_t> shard_offsets(g_world_size + 1, 0);
  for (int peer = 0; peer < g_world_size; ++peer) {
    const int64_t shard_size = shard_size_ptr[peer];
    TORCH_CHECK(shard_size >= 0, "shard_sizes must be non-negative");
    shard_offsets[peer + 1] = shard_offsets[peer] + shard_size;
  }
  const int64_t total_numel = shard_offsets[g_world_size];
  TORCH_CHECK(output_tensor.numel() == total_numel, "output_tensor numel must equal sum(shard_sizes)");
  TORCH_CHECK(local_tensor.numel() == shard_size_ptr[rank], "local_tensor numel must match this rank's shard size");
  if (total_numel == 0) {
    return;
  }

  auto dtype = nccl_dtype_for(local_tensor.scalar_type());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto element_size = local_tensor.element_size();
  char* local_base = static_cast<char*>(local_tensor.data_ptr());
  char* output_base = static_cast<char*>(output_tensor.data_ptr());
  const int64_t local_count = shard_size_ptr[rank];

  if (local_count > 0) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        output_base + shard_offsets[rank] * element_size,
        local_base,
        local_count * element_size,
        cudaMemcpyDeviceToDevice,
        stream));
  }

  MATRIX_NCCL_CHECK(ncclGroupStart());
  for (int src_rank = 0; src_rank < g_world_size; ++src_rank) {
    const int64_t count = shard_size_ptr[src_rank];
    if (count == 0) {
      continue;
    }
    void* recv_buffer = output_base + shard_offsets[src_rank] * element_size;
    if (rank == src_rank) {
      for (int peer = 0; peer < g_world_size; ++peer) {
        if (peer == rank) {
          continue;
        }
        MATRIX_NCCL_CHECK(ncclSend(local_base, count, dtype, peer, g_comm, stream));
      }
    } else {
      MATRIX_NCCL_CHECK(ncclRecv(recv_buffer, count, dtype, src_rank, g_comm, stream));
    }
  }
  MATRIX_NCCL_CHECK(ncclGroupEnd());
}
