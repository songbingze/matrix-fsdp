# Static and Elastic Param Buffer Design

Last updated: 2026-06-03

This note records the intended buffer split for the Matrix/Ragged FSDP runtime.
The goal is to keep symmetric FSDP2-like communication simple and fast while
leaving asymmetric matrix-owner and owner-segment layouts enough room to evolve
toward a DeepEP-like backend.

## Naming

Use three distinct names:

- `MatrixFlatBuffer` or `RaggedFlatBuffer`: owns parameter lifecycle and tensor
  views.
- `StaticParamBuffer`: owns static, symmetric parameter communication buffers.
- `ElasticParamBuffer`: owns elastic, asymmetric parameter communication
  buffers.

For temporary scratch tensors, prefer `CommWorkspaceCache` or
`ParamWorkspaceCache` instead of calling the cache itself a param buffer.

`StaticParamBuffer` does not mean parameters are immutable. It means the
communication layout is static: rank chunks have a fixed equal or
padded-equal shape and can use regular FSDP2-style collectives.

`ElasticParamBuffer` means the communication layout can be uneven or
owner-based: rank payloads, owner assignments, or segment metadata may differ
across ranks.

## Component Responsibilities

```text
MatrixFSDPParamGroup
  └── MatrixFlatBuffer / RaggedFlatBuffer
        ├── local_shard
        ├── full_buffer
        ├── param.data views
        ├── param.grad views
        ├── lifecycle state
        └── ParamCommBuffer
              ├── StaticParamBuffer
              └── ElasticParamBuffer
```

### Flat Buffer

The flat buffer remains the lifecycle shell. It owns:

- local parameter shard storage;
- full parameter storage or view assignment during unshard;
- `param.data` transitions between local and full views;
- local grad shard exposure;
- autograd/lifecycle bookkeeping;
- optimizer-visible parameter views.

It should not permanently own every communication policy detail. Over time,
communication-specific code should move behind `StaticParamBuffer` and
`ElasticParamBuffer`.

### StaticParamBuffer

`StaticParamBuffer` handles symmetric or padded-symmetric communication:

- contiguous even shard;
- FSDP2-like flat bucket;
- equal `all_gather_into_tensor`;
- equal `reduce_scatter_tensor`;
- padded rank chunks when padding waste is acceptable;
- stable persistent workspaces with fixed shape.

This should be the default path for AdamW/SGD-style dense FSDP behavior.

### ElasticParamBuffer

`ElasticParamBuffer` handles asymmetric communication:

- matrix-owner layout for Muon;
- whole-parameter owner layout;
- block-owner layout;
- expert-owner layout metadata;
- unequal rank payloads;
- owner-segment allgatherv/reduce-scatterv;
- custom p2p/grouped-sendrecv/native owner collectives;
- future DeepEP-like or NCCL-GIN/NVSHMEM-style backend.

This path should be used only when the planner output cannot be expressed as a
static equal or padded-equal collective without losing the intended optimizer
or ownership semantics.

### CommWorkspaceCache

`CommWorkspaceCache` is only a scratch tensor cache. It is not a parameter
buffer. It owns:

- `acquire(reference, numel)`;
- `release(lease)`;
- per `(device, dtype, numel)` reuse;
- acquire/reuse/allocate counters;
- max cached entries per key.

Both `StaticParamBuffer` and `ElasticParamBuffer` may use a workspace cache,
but neither should expose scratch tensors as parameter storage unless a future
zero-copy path makes that contract explicit.

## Selection Rule

The planner should describe the communication layout. The runtime should pick
the buffer implementation from that layout:

```python
if layout.is_static_symmetric:
    comm_buffer = StaticParamBuffer(...)
else:
    comm_buffer = ElasticParamBuffer(...)
```

Suggested layout predicates:

- `is_static_symmetric`: every rank has the same logical chunk size, or the
  layout can be padded to equal chunks with acceptable waste.
- `is_rank_contiguous`: each rank owns one contiguous segment in flat order.
- `is_full_buffer_ordered`: packed rank shards already match full-buffer order.
- `requires_owner_semantics`: at least one parameter must remain whole on an
  owner rank.
- `requires_elastic_segments`: at least one rank has uneven or multi-segment
  ownership that cannot use the static fast path.

Owner semantics should override static convenience. For example, a
matrix-owner Muon layout should stay in `ElasticParamBuffer` even if its bytes
could be padded into an equal collective, because the optimizer semantics
depend on whole-matrix ownership.

## Expected Mapping

| Layout or training path | Buffer |
| --- | --- |
| Contiguous even shard | `StaticParamBuffer` |
| Padded equal chunk | `StaticParamBuffer` |
| FSDP2-like flat bucket | `StaticParamBuffer` |
| AdamW/SGD dense sharding | `StaticParamBuffer` |
| Matrix-owner Muon | `ElasticParamBuffer` |
| Whole-parameter owner | `ElasticParamBuffer` |
| Expert-owner metadata | `ElasticParamBuffer` |
| Block-wise asymmetric owner layout | `ElasticParamBuffer` |
| Future allgatherv/reduce-scatterv custom backend | `ElasticParamBuffer` |

## Shared Interface

Both buffer types should implement a small interface:

```python
class ParamCommBuffer:
    def start_all_gather_full_params(...): ...
    def start_reduce_grad_bucket(...): ...
    def release_after_wait(...): ...
    def communication_summary(self) -> dict[str, object]: ...
    def clear_cached_workspaces(self) -> None: ...
```

The flat buffer should call this interface instead of branching directly on
every communication backend.

## Migration Plan

### Step 1: Rename Workspace Cache

Rename the scratch cache from `ElasticParamBufferWorkspace` to
`CommWorkspaceCache` or `ParamWorkspaceCache`.

Reason:

- The cache is not elastic-only.
- Static and elastic communication can both reuse scratch tensors.
- The name avoids confusing scratch storage with parameter storage.

### Step 2: Extract StaticParamBuffer

Move equal and padded collective logic into `StaticParamBuffer`:

- direct all-gather;
- padded all-gather;
- direct reduce-scatter;
- padded reduce-scatter;
- FSDP2-like bucket copy-in summary.

Keep AdamW defaults on this path.

### Step 3: Narrow ElasticParamBuffer

Keep only asymmetric layout work in `ElasticParamBuffer`:

- owner segment metadata;
- owner imbalance and segment counts;
- custom allgatherv/reduce-scatterv selection;
- matrix-owner/whole-param/expert-owner workspace planning.

### Step 4: Add Runtime Dispatch

Add a `ParamCommBuffer` member to the flat buffer and route all param gather and
grad reduce calls through it.

The flat buffer should still own lifecycle and param views, but not backend
selection details.

### Step 5: DeepEP-like Evolution

After the split is stable, evolve only `ElasticParamBuffer` toward the harder
backend:

- persistent owner-segment workspace;
- grouped chunk fast path;
- topology-aware direct/hybrid mode;
- NCCL-GIN/NVSHMEM-like backend experiments;
- kernel/wait/copy timing split.

This keeps the static AdamW/FSDP2 path clean while allowing the asymmetric Muon
path to become more aggressive.

## Testing Plan

Unit tests:

- static/equal layout selects `StaticParamBuffer`;
- padded-equal layout selects `StaticParamBuffer`;
- matrix-owner hint selects `ElasticParamBuffer`;
- whole-param owner hint selects `ElasticParamBuffer`;
- workspace cache reuse is independent from buffer type;
- communication summary reports buffer type and workspace cache stats.

Distributed CPU/Gloo tests:

- static path two-rank correctness;
- elastic owner layout two-rank correctness;
- activation checkpoint with static path;
- activation checkpoint with elastic path.

CUDA tests:

- AdamW static path vs FSDP2 correctness and timing;
- Muon matrix-owner elastic path correctness and timing;
- workspace cache disabled vs enabled;
- runtime summary confirms path selection and reuse counters.

## Design Principle

Do not make `MatrixFlatBuffer` choose every communication detail. Let the
planner describe the layout, let the comm buffer choose the communication
implementation, and let the flat buffer keep ownership of parameter lifecycle.

The intended long-term split is:

```text
FlatBuffer = lifecycle and views
StaticParamBuffer = symmetric FSDP2-like communication
ElasticParamBuffer = asymmetric owner-segment communication
WorkspaceCache = reusable scratch tensors
```
