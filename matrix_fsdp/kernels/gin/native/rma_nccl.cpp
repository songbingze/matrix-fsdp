#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <nccl.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cstring>
#include <string>
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

#if defined(NCCL_MAJOR) && defined(NCCL_MINOR) && defined(NCCL_PATCH)
#define MATRIX_NCCL_COMPILED_VERSION ((NCCL_MAJOR * 10000) + (NCCL_MINOR * 100) + NCCL_PATCH)
#else
#define MATRIX_NCCL_COMPILED_VERSION 0
#endif

#if MATRIX_NCCL_COMPILED_VERSION >= 22900 && defined(NCCL_WIN_COLL_SYMMETRIC)
#define MATRIX_FSDP_HAS_NCCL_RMA 1
#else
#define MATRIX_FSDP_HAS_NCCL_RMA 0
#endif

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
      TORCH_CHECK(false, "Unsupported NCCL dtype for MatrixFSDP RMA allgatherv");
  }
}

py::dict backend_info_dict() {
  py::dict info;
  info["compiled_nccl_version"] = MATRIX_NCCL_COMPILED_VERSION;
  info["has_nccl_rma"] = static_cast<bool>(MATRIX_FSDP_HAS_NCCL_RMA);
  info["has_device_gin"] = false;
  return info;
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

py::dict backend_info() {
  return backend_info_dict();
}

bool rma_putsignal_rank_chunks(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor shard_sizes,
    int rank) {
  TORCH_CHECK(g_comm != nullptr, "MatrixFSDP GIN/RMA NCCL communicator is not initialized");
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

#if !MATRIX_FSDP_HAS_NCCL_RMA
  return false;
#else
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
    return true;
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

  ncclWindow_t output_window = nullptr;
  ncclResult_t register_result = ncclCommWindowRegister(
      g_comm,
      output_base,
      static_cast<size_t>(output_tensor.numel() * element_size),
      &output_window,
      NCCL_WIN_COLL_SYMMETRIC);
  if (register_result != ncclSuccess) {
    return false;
  }

  MATRIX_NCCL_CHECK(ncclGroupStart());
  for (int peer = 0; peer < g_world_size; ++peer) {
    if (peer == rank || local_count == 0) {
      continue;
    }
    MATRIX_NCCL_CHECK(ncclPutSignal(
        local_base,
        local_count,
        dtype,
        peer,
        output_window,
        static_cast<size_t>(shard_offsets[rank] * element_size),
        0,
        1,
        g_comm,
        stream));
  }
  for (int peer = 0; peer < g_world_size; ++peer) {
    if (peer == rank || shard_size_ptr[peer] == 0) {
      continue;
    }
    MATRIX_NCCL_CHECK(ncclWaitSignal(peer, output_window, 0, 1, g_comm, stream));
  }
  MATRIX_NCCL_CHECK(ncclGroupEnd());
  MATRIX_NCCL_CHECK(ncclCommWindowDeregister(g_comm, output_window));
  return true;
#endif
}

bool gin_device_rank_chunks(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor shard_sizes,
    int rank) {
  (void)local_tensor;
  (void)output_tensor;
  (void)shard_sizes;
  (void)rank;
  return false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_nccl_unique_id", &get_nccl_unique_id, "Create a native NCCL unique id");
  m.def("init_nccl_comm", &init_nccl_comm, "Initialize the experimental MatrixFSDP GIN/RMA NCCL communicator");
  m.def("destroy_nccl_comm", &destroy_nccl_comm, "Destroy the experimental MatrixFSDP GIN/RMA NCCL communicator");
  m.def("backend_info", &backend_info, "Report experimental GIN/RMA backend compile-time support");
  m.def("rma_putsignal_rank_chunks", &rma_putsignal_rank_chunks, "NCCL RMA put/signal rank-chunk allgatherv");
  m.def("gin_device_rank_chunks", &gin_device_rank_chunks, "Reserved NCCL Device API / GIN rank-chunk allgatherv");
}
