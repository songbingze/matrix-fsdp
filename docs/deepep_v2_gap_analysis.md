# DeepEP V2 Gap Analysis

Last updated: 2026-06-03

This note summarizes how far MatrixFSDP currently is from a DeepEP V2-style
communication stack. The comparison is intentionally scoped to ideas that are
relevant to MatrixFSDP parameter communication. DeepEP is primarily an expert
parallelism communication library for MoE token dispatch/combine, while
MatrixFSDP manages parameter lifecycle, sharding, gradient reduction, optimizer
state, and checkpointing.

## Scope Boundary

DeepEP V2 and MatrixFSDP solve different first-order problems:

- DeepEP V2: high-throughput and low-latency expert-parallel token
  communication, especially dispatch/combine for MoE. It also exposes
  experimental PP/CP/RMA primitives.
- MatrixFSDP: FSDP2-style parameter sharding with matrix-aware layouts,
  matrix-owner Muon planning, optimizer integration, and DCP/state_dict support.

So the right target is not "replace DeepEP". The target is a parameter-side
DeepEP-like v1: persistent communication workspaces, fewer small communication
operations, topology-aware scheduling, detailed timing, and eventually a
backend that can move uneven owner segments without falling back to expensive
per-segment host orchestration.

Public DeepEP V2 references used for this comparison:

- DeepEP README: https://github.com/deepseek-ai/DeepEP
- DeepEP V2 communication/topology notes:
  https://deepwiki.com/deepseek-ai/DeepEP/3.2-internode-communication

## Current MatrixFSDP State

MatrixFSDP already has several pieces that are directionally aligned with a
DeepEP-like runtime:

- Matrix-owner planner for Muon-style full-matrix ownership.
- Role-greedy owner assignment for better rank balance than naive rotation.
- Owner-segment communication modes for uneven matrix-owner shards.
- Custom collective selection with `native_sendrecv`,
  `native_group_broadcast`, and `native_reduce` paths.
- Ordered owner prefetch queue to avoid rank-order deadlocks in custom owner
  collectives.
- Runtime event tracing for all-gather/reduce-scatter enqueue and wait phases.
- Runtime communication summary with gather/reduce backend counts, custom
  implementation counts, owner imbalance, workspace kind, workspace acquire,
  reuse, allocation, and cache-limit fields.
- Elastic workspace cache plumbing, now exposed to benchmark scripts through
  `matrix_max_cached_elastic_workspaces_per_key`.
- Default AdamW path remains close to FSDP2-style bucket copy-in and does not
  depend on owner-segment custom collectives.

The latest non-GPU change prepared a persistent workspace experiment for
custom owner-segment Muon paths. It does not prove performance yet; it only
makes the policy measurable and easy to enable during GPU runs.

## Major Gaps vs DeepEP V2

### 1. Transport Backend

DeepEP V2 moved to a lightweight NCCL Gin backend that can reuse NCCL
communicators and provide device-side put/get-like behavior. MatrixFSDP does
not have an equivalent transport. Our native custom paths are still closer to
PyTorch/NCCL process-group orchestration plus custom kernels around it.

Impact:

- More CPU-side scheduling overhead.
- Harder to reduce per-segment launch/count overhead.
- Harder to implement topology teams such as local NVLink domain, rail domain,
  and global domain.

Needed work:

- Build a dedicated parameter-communication backend abstraction.
- Avoid depending on partially exposed ProcessGroupNCCL internals.
- Decide whether the next backend should be NCCL Device API/GIN-like,
  NVSHMEM-like, or a narrower grouped NCCL kernel path.

### 2. Topology Model

DeepEP V2 models scale-up and scale-out domains separately and supports direct
and hybrid communication modes. MatrixFSDP currently treats the shard group
mostly as a flat DP shard group, with DeviceMesh support for DP shard and DP
replicate roles.

Impact:

- Single-node shard4/shard8 can work, but multi-node performance is not yet a
  first-class target.
- We cannot yet choose different algorithms for NVLink-local movement versus
  RDMA movement.
- No rail/team abstraction for corresponding ranks across nodes.

Needed work:

- Extend runtime communication metadata from `rank/world_size` to
  `scale_up_rank`, `scale_out_rank`, `local_rank`, `rail_rank`.
- Add topology-aware backend selection.
- Keep the public FSDP API DP-focused; do not pull EP token dispatch into this
  library.

### 3. Persistent Communication Buffer Architecture

DeepEP V2 uses a unified ElasticBuffer interface and emphasizes persistent
communication buffers. MatrixFSDP has flat buffers for parameter lifecycle and
an elastic workspace cache, but the custom owner-segment path still behaves
more like "allocate or borrow a temporary workspace per operation".

Impact:

- Allocation/free overhead may still appear in short-step Muon benchmarks.
- Buffer high-water behavior is not yet controlled by one global
  communication-buffer planner.
- Workspace reuse is observable now, but not yet optimized as a core invariant.

Needed work:

- Turn owner-segment workspace cache into a runtime policy with high-water
  accounting.
- Prefer one persistent owner-segment workspace per param group or per
  communication stream when shapes are stable.
- Add GPU benchmarks that separate allocation time, copy/pack time, kernel
  time, and wait time.

### 4. Granularity of Communication Operations

DeepEP V2 is built around coarse dispatch/combine kernels with carefully
managed metadata. MatrixFSDP owner-segment paths can still degenerate into many
small operations, especially with uneven matrix-owner layouts and many
transformer blocks.

Impact:

- Muon forward/backward can become communication-bound when sequence length is
  short or compute is not enough to hide communication.
- Per-segment p2p-like behavior is not competitive with FSDP2-style bulk
  all-gather/reduce-scatter when the shard pattern is close to balanced.

Needed work:

- Detect rank-contiguous chunks and route them through grouped chunk fast
  paths.
- Coalesce per-param-group owner segments into one coarser operation when the
  planner guarantees full-buffer order.
- Add counters for "collective calls per param group" and "segments per
  collective" to the runtime summary.

### 5. SM Usage and Kernel Resource Control

DeepEP V2 reports significantly lower SM usage than its V1 path for V3-like
training, while maintaining or improving performance. MatrixFSDP does not yet
have analytical SM/QP selection or an SM budget model for parameter
communication.

Impact:

- Custom communication can steal compute resources from matmul/attention.
- There is no principled way to choose communication kernel occupancy based on
  model shape, sequence length, or topology.

Needed work:

- Record communication kernel occupancy/resource usage in GPU profiles.
- Add mode-specific SM budget knobs only after measurement.
- Prefer fewer coarse operations before attempting low-level SM tuning.

### 6. Direct vs Hybrid Communication

DeepEP V2 supports direct and hybrid modes. Hybrid communication forwards data
hierarchically across scale-out and scale-up domains. MatrixFSDP has no
equivalent hierarchical owner-segment movement.

Impact:

- Cross-node shard groups may work functionally through ordinary distributed
  collectives, but they are not optimized.
- Matrix-owner layouts across nodes may be much slower than FSDP2 unless the
  shard group and planner are topology-aware.

Needed work:

- Add a direct owner-segment mode for single-node or small scale-out.
- Add a hybrid owner-segment mode for multi-node: RDMA to node proxy, then
  NVLink-local redistribution.
- Keep this below the planner/runtime boundary; the planner should expose
  owner placement and rank-contiguous chunks, not token-routing behavior.

### 7. Load Imbalance Handling

DeepEP V2 is built for imbalanced EP traffic and is working on reducing
intermediate buffer sizes by replaying EP communication. MatrixFSDP currently
uses planner-side balancing and reports owner imbalance, but runtime does not
actively compensate for imbalance.

Impact:

- Matrix-owner layouts can be unbalanced when shard count exceeds the number
  of large matrices in a local unit.
- Larger planner scope can improve balance, but runtime materialization still
  must stay local to the current compute unit to avoid peak-memory growth.

Needed work:

- Keep planner groups flexible but maintain compute/runtime units at block
  granularity.
- Add planner reports for per-rank matrix count, byte count, and estimated
  communication bytes.
- Add runtime policies that choose between owner-segment and FSDP2-like padded
  collective paths based on measured imbalance and segment count.

### 8. Zero-Copy Boundary

DeepEP has experimental zero-copy work to remove copies between PyTorch tensors
and communication buffers. MatrixFSDP has forward parameter view assignment and
flat-buffer ownership, but gradient copy-in is still intentionally FSDP2-like
for the default fast path.

Impact:

- Forward parameter materialization can be close to zero-copy in favorable
  layouts.
- Backward gradient packing still has unavoidable-looking copy-in for normal
  autograd-produced per-parameter gradients unless gradients are written
  directly into the communication buffer.

Needed work:

- Keep default AdamW on the fastest measured copy-in bucket path.
- Treat zero-copy grad bucket as experimental until it beats copy-in in GPU
  benchmarks.
- Explore autograd grad-view plumbing only after scheduler and communication
  overhead are under control.

## What Is Not a MatrixFSDP Gap

These belong to a higher-level training stack or EP runtime, not MatrixFSDP:

- MoE token routing.
- Token dispatch/combine.
- EP all-to-all for activations.
- Attention kernels.
- QKV/proj TP sharding.
- MLP matmul parallelism.
- Expert no-gather forward execution.
- Expert-local optimizer semantics outside the DP-managed parameter set.

MatrixFSDP should manage only the DP/HSDP parameter set selected by the upper
training stack. If a parameter is owned by TP/EP, it should be ignored or
handled through a clear placement contract rather than silently pulled into
MatrixFSDP.

## Practical Roadmap

### C1: Measurement and Workspace Stability

Status: partially implemented.

- Persistent elastic workspace cache is exposed to benchmark configs.
- Runtime summary shows workspace cache limit and workspace reuse/allocation.
- Collective phase tracing exists for enqueue/wait events.

Next:

- Run GPU benchmarks with `MATRIX_WORKSPACE_CACHE_PER_KEY=1`.
- Compare allocation count and total step time against cache disabled.
- Record whether Muon owner/custom paths improve without hurting AdamW.

### C2: Coarser Chunk Fast Path

Goal: reduce small-message overhead before attempting a full custom transport.

- Detect rank-contiguous owner chunks.
- Use one grouped chunk operation per param group where possible.
- Fall back to group broadcast only when the layout is truly non-contiguous.
- Add runtime summary counters for chunk path selection.

### C3: Topology-Aware Backend Selection

Goal: make single-node and multi-node choices explicit.

- Single-node: prefer NVLink-local direct owner chunks.
- Multi-node: separate local and rail domains.
- Expose enough metadata to evaluate a future hybrid mode.

### C4: DeepEP-like Parameter Backend v1

Goal: a self-contained parameter communication backend.

- Persistent owner-segment workspace.
- Fewer grouped operations.
- Backend-owned lifecycle, not half-managed by torch ProcessGroup internals.
- Fine-grained timing: pack/copy, kernel, wait, unpack/view assignment.

### C5: Long-Sequence Overlap Validation

Goal: verify whether longer sequence lengths hide owner communication.

- Test AdamW and Muon with seq4096, seq8192, and seq16384.
- Compare FSDP2 Muon, MatrixFSDP matrix-owner Muon, and MatrixFSDP AdamW.
- Measure forward, backward, optimizer step, peak memory, and reserved memory
  by rank.

## Bottom Line

MatrixFSDP is already aligned with DeepEP V2 at the planner/runtime-design
level: matrix ownership, elastic workspace accounting, custom owner-segment
communication, ordered prefetch, and detailed runtime summaries are in place.

The largest remaining gap is not the planner. It is the transport and
communication granularity:

1. DeepEP V2 has a purpose-built backend and topology model.
2. MatrixFSDP still uses a lighter custom path around existing PyTorch/NCCL
   process-group behavior.
3. The next most valuable step is to reduce small owner-segment operations and
   make persistent workspaces measurable on GPU.

After C1/C2 GPU data, we can decide whether a deeper NCCL-GIN/NVSHMEM-like
backend is worth the engineering cost.
