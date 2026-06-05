from .layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout, RankLayout, ShardPlan
from .managed_param import ManagedParam, ManagedParamRegistry, ParamShardHint
from .mesh import (
    DeviceMeshMetadata,
    infer_replicate_mesh_dim,
    infer_shard_mesh_dim,
    mesh_coordinate,
    mesh_dim_name,
    mesh_dim_names,
    mesh_metadata,
    mesh_shape,
    normalize_mesh_dim,
)
from .placement import (
    PlacementCompatibility,
    MatrixShard,
    MatrixShardPlacement,
    explain_matrix_shard_compatibility,
    is_matrix_shard_compatible_plan,
    is_matrix_shard_placement,
    matrix_shard_from_layout,
    matrix_shard_from_plan,
    shard_sizes_to_matrix_local_units,
)
from .state import MatrixShardedState, matrix_shard_metadata, matrix_sharded_state_metadata
from .torch_dtensor import make_matrix_dtensor, make_matrix_placements
