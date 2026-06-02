#include <torch/extension.h>

void copy_rank_segments_to_full_cuda(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor global_starts,
    torch::Tensor local_starts,
    torch::Tensor numels,
    int64_t max_numel);

void copy_range_cuda(torch::Tensor input_tensor, torch::Tensor output_tensor, int64_t input_offset);

pybind11::bytes get_nccl_unique_id();
void init_nccl_comm(pybind11::bytes unique_id_bytes, int rank, int world_size);
void destroy_nccl_comm();
void group_broadcast_rank_segments(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor src_ranks,
    torch::Tensor global_starts,
    torch::Tensor local_starts,
    torch::Tensor numels,
    int rank);
void sendrecv_rank_segments(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor src_ranks,
    torch::Tensor global_starts,
    torch::Tensor local_starts,
    torch::Tensor numels,
    int rank);
void sendrecv_rank_chunks(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor shard_sizes,
    int rank);
void reduce_rank_chunks(
    torch::Tensor packed_rank_chunks,
    torch::Tensor local_output,
    torch::Tensor shard_sizes,
    int rank,
    bool compact);

#define MATRIX_CHECK_CUDA(tensor) TORCH_CHECK((tensor).is_cuda(), #tensor " must be a CUDA tensor")
#define MATRIX_CHECK_CONTIGUOUS(tensor) TORCH_CHECK((tensor).is_contiguous(), #tensor " must be contiguous")
#define MATRIX_CHECK_INT64(tensor) TORCH_CHECK((tensor).scalar_type() == at::ScalarType::Long, #tensor " must be int64")

void copy_rank_segments_to_full(
    torch::Tensor local_tensor,
    torch::Tensor output_tensor,
    torch::Tensor global_starts,
    torch::Tensor local_starts,
    torch::Tensor numels,
    int64_t max_numel) {
  MATRIX_CHECK_CUDA(local_tensor);
  MATRIX_CHECK_CUDA(output_tensor);
  MATRIX_CHECK_CUDA(global_starts);
  MATRIX_CHECK_CUDA(local_starts);
  MATRIX_CHECK_CUDA(numels);
  MATRIX_CHECK_CONTIGUOUS(local_tensor);
  MATRIX_CHECK_CONTIGUOUS(output_tensor);
  MATRIX_CHECK_CONTIGUOUS(global_starts);
  MATRIX_CHECK_CONTIGUOUS(local_starts);
  MATRIX_CHECK_CONTIGUOUS(numels);
  MATRIX_CHECK_INT64(global_starts);
  MATRIX_CHECK_INT64(local_starts);
  MATRIX_CHECK_INT64(numels);
  TORCH_CHECK(local_tensor.dim() == 1, "local_tensor must be 1D");
  TORCH_CHECK(output_tensor.dim() == 1, "output_tensor must be 1D");
  TORCH_CHECK(local_tensor.scalar_type() == output_tensor.scalar_type(), "source and destination dtype mismatch");
  TORCH_CHECK(global_starts.numel() == local_starts.numel(), "metadata length mismatch");
  TORCH_CHECK(global_starts.numel() == numels.numel(), "metadata length mismatch");
  TORCH_CHECK(max_numel >= 0, "max_numel must be non-negative");
  if (global_starts.numel() == 0) {
    return;
  }
  copy_rank_segments_to_full_cuda(local_tensor, output_tensor, global_starts, local_starts, numels, max_numel);
}

void copy_range(torch::Tensor input_tensor, torch::Tensor output_tensor, int64_t input_offset) {
  MATRIX_CHECK_CUDA(input_tensor);
  MATRIX_CHECK_CUDA(output_tensor);
  MATRIX_CHECK_CONTIGUOUS(input_tensor);
  MATRIX_CHECK_CONTIGUOUS(output_tensor);
  TORCH_CHECK(input_tensor.dim() == 1, "input_tensor must be 1D");
  TORCH_CHECK(output_tensor.dim() == 1, "output_tensor must be 1D");
  TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type(), "source and destination dtype mismatch");
  TORCH_CHECK(input_offset >= 0, "input_offset must be non-negative");
  TORCH_CHECK(input_offset + output_tensor.numel() <= input_tensor.numel(), "copy range exceeds input tensor");
  if (output_tensor.numel() == 0) {
    return;
  }
  copy_range_cuda(input_tensor, output_tensor, input_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy_rank_segments_to_full", &copy_rank_segments_to_full, "Copy rank-owned segments into full buffer");
  m.def("copy_range", &copy_range, "Copy a contiguous 1D range into an output tensor");
  m.def("get_nccl_unique_id", &get_nccl_unique_id, "Create a native NCCL unique id");
  m.def("init_nccl_comm", &init_nccl_comm, "Initialize the native MatrixFSDP NCCL communicator");
  m.def("destroy_nccl_comm", &destroy_nccl_comm, "Destroy the native MatrixFSDP NCCL communicator");
  m.def("group_broadcast_rank_segments", &group_broadcast_rank_segments, "Grouped NCCL broadcasts for rank segments");
  m.def("sendrecv_rank_segments", &sendrecv_rank_segments, "Grouped NCCL send/recv allgatherv for rank segments");
  m.def("sendrecv_rank_chunks", &sendrecv_rank_chunks, "Grouped NCCL send/recv allgatherv for contiguous rank chunks");
  m.def("reduce_rank_chunks", &reduce_rank_chunks, "Grouped NCCL reduce-scatterv for contiguous rank chunks");
}
