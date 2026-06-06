import unittest

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from test.fsdp.bench_fsdp2_compare import (
    BlockGroup,
    DEFAULT_FAST_MATRIX_MODE,
    DEFAULT_FULLY_SHARD_API_COMPARE_MODES,
    DEFAULT_MEMORY_ACCOUNTING_MODES,
    EXPERIMENTAL_MODES,
    DEFAULT_MUON_ADJUST_LR_FN,
    FSDP2CompareConfig,
    FSDP2CompareRow,
    FSDP2CorrectnessResult,
    FSDP2MemoryTraceRow,
    FSDP2PhaseTimingRow,
    MatrixRuntimeCommunicationRow,
    CopyInBenchmarkRow,
    ParamMaterializationBenchmarkRow,
    MixedMuonAdamWOptimizer,
    format_compare_table,
    format_correctness_result,
    format_copyin_table,
    format_memory_trace_table,
    format_runtime_communication_table,
    format_param_materialization_table,
    format_phase_timing_table,
    run_copyin_benchmark,
    run_param_materialization_benchmark,
    _memory_accounting,
    _adamw_params,
    _make_inputs,
    _make_model,
    _muon_params,
    _prefetch_budget,
    _is_unit,
)
from matrix_fsdp import MatrixFSDPOptimizer, make_muon_shard_aware_group_planner, matrix_fully_shard
from scripts.run_isolated_gpu_benchmark import (
    IsolatedBenchmarkRun,
    _last_table_row_for_mode,
    format_isolated_summary,
)


class FSDP2CompareBenchTest(unittest.TestCase):
    def test_default_config_includes_fsdp2_and_matrix(self):
        config = FSDP2CompareConfig()

        self.assertIn("fsdp2", config.modes)
        self.assertEqual(config.optimizer, "sgd")
        self.assertFalse(config.activation_checkpoint)
        self.assertFalse(config.checkpoint_use_reentrant)
        self.assertFalse(config.activation_checkpoint_wrapper)
        self.assertEqual(config.matrix_max_cached_elastic_workspaces_per_key, 0)
        self.assertEqual(DEFAULT_FAST_MATRIX_MODE, "matrix_default")
        self.assertEqual(config.modes, ("fsdp2", DEFAULT_FAST_MATRIX_MODE))
        self.assertIn("matrix_zero_copy_grad_bucket", EXPERIMENTAL_MODES)
        self.assertIn("matrix_prefetch_late_backward_fsdp2_chunk", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_pre_backward", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_matrix_all_gather", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_custom_collective", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_custom_collective_prefetch_cap2", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_custom_collective_copy_in", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_custom_collective_zero_copy_grad_bucket", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_custom_collective_memory_capped", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_role_greedy_custom_collective_post_reshard", EXPERIMENTAL_MODES)
        self.assertIn("matrix_owner_muon_cost_aware_custom_collective", EXPERIMENTAL_MODES)
        self.assertEqual(DEFAULT_FULLY_SHARD_API_COMPARE_MODES, ("fsdp2_api", "matrix_api"))

    def test_block_group_size_wraps_adjacent_benchmark_blocks(self):
        config = FSDP2CompareConfig(
            model="transformer_split_qkv",
            unit="block",
            layers=5,
            hidden=16,
            intermediate=32,
            heads=4,
            block_group_size=2,
        )
        model = _make_model(config)

        groups = [module for module in model.modules() if isinstance(module, BlockGroup)]

        self.assertEqual(len(groups), 3)
        self.assertTrue(all(_is_unit(group, config) for group in groups))
        self.assertTrue(all(not _is_unit(layer, config) for group in groups for layer in group.layers))

    def test_format_compare_table_includes_core_columns(self):
        rows = (
            FSDP2CompareRow(
                mode="fsdp2",
                unit="linear",
                device="cuda",
                world_size=2,
                layers=4,
                hidden=512,
                intermediate=2048,
                batch_size=8,
                optimizer="sgd",
                dtype="float32",
                param_count=8_388_608,
                avg_step_ms=12.345,
                peak_memory_mb=256.0,
            ),
        )

        table = format_compare_table(rows)

        self.assertIn("mode", table)
        self.assertIn("avg_step_ms", table)
        self.assertIn("model", table)
        self.assertIn("group", table)
        self.assertIn("optim", table)
        self.assertIn("budget", table)
        self.assertIn("peak_mem_mb", table)
        self.assertIn("fsdp2", table)

    def test_format_correctness_result_includes_diffs(self):
        result = FSDP2CorrectnessResult(
            reference_mode="fsdp2",
            candidate_mode="matrix_prefetch",
            unit="block",
            device="cuda",
            world_size=2,
            layers=8,
            hidden=1024,
            intermediate=4096,
            batch_size=4,
            optimizer="sgd",
            dtype="float32",
            steps=3,
            max_loss_abs_diff=1e-6,
            max_output_abs_diff=2e-6,
            max_grad_abs_diff=3e-6,
            max_grad_rel_diff=4e-6,
            grad_checked_param_count=12,
            grad_mismatched_param_count=0,
        )

        text = format_correctness_result(result)

        self.assertIn("matrix_prefetch vs fsdp2", text)
        self.assertIn("loss max abs diff", text)
        self.assertIn("output max abs diff", text)
        self.assertIn("grad max abs diff", text)
        self.assertIn("grad max rel diff", text)
        self.assertIn("grad checked params=12", text)
        self.assertIn("grad mismatched params=0", text)

    def test_format_memory_trace_table_includes_phase_memory(self):
        rows = (
            FSDP2MemoryTraceRow(
                mode="matrix_auto_finalize",
                unit="block",
                device="cuda",
                world_size=2,
                optimizer="adamw",
                dtype="float32",
                phase="after_backward",
                current_memory_mb=123.4,
                peak_memory_mb=256.7,
                prefetch_budget="f=1/b=1",
                active_full_param_buffers=2,
                full_param_buffer_mb=64.0,
                local_shard_mb=32.0,
                grad_bucket_mb=16.0,
                workspace_mb=8.0,
                optimizer_state_mb=48.0,
                pending_backward_reduces=1,
            ),
        )

        table = format_memory_trace_table(rows)

        self.assertIn("phase", table)
        self.assertIn("current_mem_mb", table)
        self.assertIn("full_bufs", table)
        self.assertIn("workspace_mb", table)
        self.assertIn("opt_state_mb", table)
        self.assertIn("adamw", table)
        self.assertIn("after_backward", table)
        self.assertIn("f=1/b=1", table)
        self.assertIn("256.7", table)
        self.assertIn("64.0", table)

    def test_memory_accounting_modes_cover_default_comparison(self):
        self.assertEqual(
            DEFAULT_MEMORY_ACCOUNTING_MODES,
            (
                "fsdp2",
                "matrix_default",
                "matrix_memory_capped",
                "matrix_zero_copy_grad_bucket",
                "matrix_no_prefetch",
            ),
        )

    def test_memory_accounting_reports_matrix_runtime_buffers(self):
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            reshard_after_forward=False,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        model(torch.randn(3, 4)).sum().backward()
        accounting = _memory_accounting(model, optimizer)
        optimizer.step()
        accounting_after_step = _memory_accounting(model, optimizer)

        self.assertGreater(accounting["local_shard_mb"], 0.0)
        self.assertGreater(accounting["full_grad_buffer_mb"], 0.0)
        self.assertGreater(accounting_after_step["optimizer_state_mb"], 0.0)

    def test_format_phase_timing_table_includes_phase_columns(self):
        rows = (
            FSDP2PhaseTimingRow(
                mode="matrix_prefetch_adaptive",
                unit="block",
                device="cuda",
                world_size=8,
                optimizer="sgd",
                dtype="bfloat16",
                avg_zero_grad_ms=1.0,
                avg_forward_ms=2.0,
                avg_backward_ms=3.0,
                avg_step_ms=4.0,
                avg_total_ms=10.0,
                peak_memory_mb=512.0,
                model="transformer",
                seq_len=4096,
                prefetch_budget="1",
            ),
        )

        table = format_phase_timing_table(rows)

        self.assertIn("fwd_ms", table)
        self.assertIn("bwd_ms", table)
        self.assertIn("step_ms", table)
        self.assertIn("total_ms", table)
        self.assertIn("4096", table)
        self.assertIn("group", table)

    def test_isolated_benchmark_summary_parses_compare_table(self):
        output = """
mode            model  unit   group  device  world  layers  hidden  inter  seq  batch  optim  dtype    params  budget   avg_step_ms  peak_mem_mb
--------------  -----  -----  -----  ------  -----  ------  ------  -----  ---  -----  -----  -------  ------  -------  -----------  -----------
matrix_default  mlp    block  2      cpu     1      4       16      32     -    2      sgd    float32  4096    f=1/b=1  0.862        0.0
"""
        row = _last_table_row_for_mode(output, "matrix_default")
        summary = format_isolated_summary(
            (
                IsolatedBenchmarkRun("matrix_default", 0, 0, avg_step_ms=float(row["avg_step_ms"]), peak_mem_mb=0.0),
                IsolatedBenchmarkRun("matrix_default", 1, 0, avg_step_ms=1.0, peak_mem_mb=2.0),
            )
        )

        self.assertEqual(row["group"], "2")
        self.assertIn("p50_ms", summary)
        self.assertIn("matrix_default", summary)

    def test_prefetch_budget_reports_elastic_workspace_cache_limit(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        optimizer = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_cached_elastic_workspaces_per_key=1,
        )

        self.assertIn("elastic_ws=1", _prefetch_budget(optimizer))

    def test_format_runtime_communication_table_includes_workspace_cache_limit(self):
        row = MatrixRuntimeCommunicationRow(
            mode="matrix_owner_muon_role_greedy_custom_collective",
            model="transformer_split_qkv",
            unit="block",
            device="cuda",
            world_size=8,
            optimizer="muon",
            dtype="bfloat16",
            param_groups=16,
            gather_backend_counts="owner_broadcast:16",
            resolved_custom_allgatherv_counts="native_sendrecv:16",
            grad_reduce_backend_counts="native_reduce:16",
            resolved_custom_reduce_scatterv_counts="native_reduce:16",
            rank_chunk_fast_paths=16,
            packed_full_order=16,
            max_segment_count=8,
            max_segments_per_rank=1,
            max_padding_waste_ratio=0.0,
            max_owner_imbalance_ratio=0.125,
            workspace_preferred_kind_counts="owner_segment:16",
            max_workspace_preferred_numel=1024,
            max_workspace_padded_numel=1024,
            max_workspace_padding_waste_ratio=0.0,
            workspace_acquires=32,
            workspace_reuses=16,
            workspace_allocates=16,
            workspace_cache_limit=1,
            max_workspace_allocated_numel=1024,
            min_shard_size=512,
            max_shard_size=1024,
        )

        table = format_runtime_communication_table((row,))

        self.assertIn("workspace_cache", table)
        self.assertIn("native_reduce:16", table)
        self.assertIn("owner_segment:16", table)

    def test_transformer_benchmark_model_uses_sequence_inputs(self):
        config = FSDP2CompareConfig(
            model="transformer",
            layers=2,
            hidden=16,
            intermediate=32,
            heads=4,
            batch_size=3,
            seq_len=5,
        )
        model = _make_model(config)
        x, target = _make_inputs(config, device="cpu")

        self.assertEqual(tuple(x.shape), (3, 5, 16))
        self.assertEqual(tuple(target.shape), (3, 5, 16))
        self.assertEqual(tuple(model(x).shape), (3, 5, 16))

    def test_transformer_benchmark_model_supports_activation_checkpoint(self):
        config = FSDP2CompareConfig(
            model="transformer",
            layers=2,
            hidden=16,
            intermediate=32,
            heads=4,
            batch_size=3,
            seq_len=5,
            activation_checkpoint=True,
        )
        model = _make_model(config)
        x, _target = _make_inputs(config, device="cpu")

        self.assertEqual(tuple(model(x).shape), (3, 5, 16))

    def test_transformer_benchmark_model_supports_checkpoint_wrapper(self):
        config = FSDP2CompareConfig(
            model="transformer",
            layers=2,
            hidden=16,
            intermediate=32,
            heads=4,
            batch_size=3,
            seq_len=5,
            activation_checkpoint=True,
            activation_checkpoint_wrapper=True,
        )
        model = _make_model(config)
        x, _target = _make_inputs(config, device="cpu")

        self.assertIsInstance(model.layers[0], CheckpointWrapper)
        self.assertEqual(tuple(model(x).shape), (3, 5, 16))

    def test_split_qkv_transformer_uses_ordered_matrix_params_for_shard8(self):
        config = FSDP2CompareConfig(
            model="transformer_split_qkv",
            layers=1,
            hidden=16,
            intermediate=64,
            heads=4,
            batch_size=3,
            seq_len=5,
        )
        model = _make_model(config)
        x, target = _make_inputs(config, device="cpu")

        self.assertEqual(tuple(x.shape), (3, 5, 16))
        self.assertEqual(tuple(target.shape), (3, 5, 16))
        self.assertEqual(tuple(model(x).shape), (3, 5, 16))
        self.assertEqual(
            [name for name, param in model.layers[0].named_parameters() if param.ndim == 2],
            [
                "q.weight",
                "k.weight",
                "v.weight",
                "proj.weight",
                "mlp.up0.weight",
                "mlp.up1.weight",
                "mlp.down.weight",
            ],
        )

    def test_muon_optimizer_param_split_uses_adamw_for_non_2d_params(self):
        matrix = torch.nn.Parameter(torch.ones(4, 4))
        vector = torch.nn.Parameter(torch.ones(4))
        empty_matrix = torch.nn.Parameter(torch.empty(0))

        params = [matrix, vector, empty_matrix]

        self.assertEqual(_muon_params(params), [matrix])
        self.assertEqual(_adamw_params(params), [vector])

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_muon_optimizer_defaults_to_moonshot_rms_matching(self):
        matrix = torch.nn.Parameter(torch.ones(4, 4))

        optimizer = MixedMuonAdamWOptimizer([matrix], [])

        self.assertEqual(DEFAULT_MUON_ADJUST_LR_FN, "match_rms_adamw")
        self.assertIsNotNone(optimizer.muon)
        self.assertEqual(optimizer.muon.param_groups[0]["adjust_lr_fn"], DEFAULT_MUON_ADJUST_LR_FN)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_muon_optimizer_can_delay_state_allocation_until_step(self):
        matrix = torch.nn.Parameter(torch.ones(4, 4))

        optimizer = MixedMuonAdamWOptimizer([matrix], [], lazy_muon_init=True)

        self.assertEqual(optimizer.state, {})
        self.assertEqual(optimizer.muon.state_dict()["state"], {})
        self.assertEqual(optimizer.muon.param_groups[0]["adjust_lr_fn"], DEFAULT_MUON_ADJUST_LR_FN)

    def test_rotating_muon_group_planner_rotates_owner_ranks(self):
        config = FSDP2CompareConfig(model="transformer", hidden=16, intermediate=64, heads=4)
        block = _make_model(config).layers[0]
        from matrix_fsdp.managed_param import ManagedParamRegistry
        from matrix_fsdp.shard_hint import build_shard_hints

        params = ManagedParamRegistry.from_module(block, shard_hints=build_shard_hints(block)).as_list()
        planner = make_muon_shard_aware_group_planner(rotation_strategy="round_robin")

        first = planner(params, 4).layout
        second = planner(params, 4).layout

        self.assertEqual(first.owner_ranks("qkv.weight"), (0,))
        self.assertEqual(second.owner_ranks("qkv.weight"), (1,))
        self.assertEqual(first.owner_ranks("mlp.down.weight"), (3,))
        self.assertEqual(second.owner_ranks("mlp.down.weight"), (0,))

    def test_split_qkv_muon_planner_maps_one_block_across_shard8(self):
        config = FSDP2CompareConfig(model="transformer_split_qkv", hidden=16, intermediate=64, heads=4)
        block = _make_model(config).layers[0]
        from matrix_fsdp.managed_param import ManagedParamRegistry
        from matrix_fsdp.shard_hint import build_shard_hints

        params = ManagedParamRegistry.from_module(block, shard_hints=build_shard_hints(block)).as_list()
        evaluation = make_muon_shard_aware_group_planner()(params, 8)
        layout = evaluation.layout

        self.assertEqual(evaluation.name, "matrix_owner_tail")
        self.assertEqual(layout.owner_ranks("q.weight"), (0,))
        self.assertEqual(layout.owner_ranks("k.weight"), (1,))
        self.assertEqual(layout.owner_ranks("v.weight"), (2,))
        self.assertEqual(layout.owner_ranks("proj.weight"), (3,))
        self.assertEqual(layout.owner_ranks("mlp.up0.weight"), (4,))
        self.assertEqual(layout.owner_ranks("mlp.up1.weight"), (5,))
        self.assertEqual(layout.owner_ranks("mlp.down.weight"), (6,))
        self.assertEqual(layout.owner_ranks("norm1.weight"), (7,))
        self.assertEqual(layout.owner_ranks("norm1.bias"), (7,))
        self.assertEqual(layout.owner_ranks("norm2.weight"), (7,))
        self.assertEqual(layout.owner_ranks("norm2.bias"), (7,))

    def test_copyin_benchmark_runs_on_cpu_flat_layout(self):
        rows = run_copyin_benchmark(
            device="cpu",
            dtype="float32",
            world_size=2,
            param_count=2,
            param_numel=8,
            steps=1,
            warmup_steps=0,
            layout="flat",
            backends=("auto", "foreach_copy", "segment_copy"),
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].total_numel, 16)
        self.assertGreaterEqual(rows[0].avg_ms, 0.0)

    def test_copyin_benchmark_runs_on_cpu_chunk_cat_layout(self):
        rows = run_copyin_benchmark(
            device="cpu",
            dtype="float32",
            world_size=2,
            param_count=2,
            param_numel=5,
            steps=1,
            warmup_steps=0,
            layout="chunk_cat",
            backends=("auto", "chunk_cat", "segment_copy"),
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].total_numel, 10)
        self.assertGreaterEqual(rows[0].avg_ms, 0.0)

    def test_format_copyin_table_includes_backend_and_bandwidth(self):
        table = format_copyin_table(
            (
                CopyInBenchmarkRow(
                    backend="auto",
                    resolved_layout="flat_contiguous",
                    layout="flat",
                    device="cuda",
                    dtype="bfloat16",
                    world_size=8,
                    param_count=64,
                    param_numel=1024,
                    total_numel=65536,
                    avg_ms=0.123,
                    bandwidth_gb_s=42.0,
                ),
            )
        )

        self.assertIn("backend", table)
        self.assertIn("resolved", table)
        self.assertIn("flat_contiguous", table)
        self.assertIn("auto", table)
        self.assertIn("GB/s", table)
        self.assertIn("42.00", table)

    def test_param_materialization_benchmark_compares_view_and_copy(self):
        rows = run_param_materialization_benchmark(
            device="cpu",
            dtype="float32",
            param_count=2,
            param_numel=8,
            steps=1,
            warmup_steps=0,
            backends=("view_assign", "copy_out"),
        )

        self.assertEqual([row.backend for row in rows], ["view_assign", "copy_out"])
        self.assertEqual(rows[0].total_numel, 16)
        self.assertGreaterEqual(rows[0].avg_ms, 0.0)

    def test_format_param_materialization_table_includes_backends(self):
        table = format_param_materialization_table(
            (
                ParamMaterializationBenchmarkRow(
                    backend="view_assign",
                    device="cuda",
                    dtype="bfloat16",
                    param_count=64,
                    param_numel=1024,
                    total_numel=65536,
                    avg_ms=0.001,
                    bandwidth_gb_s=1000.0,
                ),
                ParamMaterializationBenchmarkRow(
                    backend="copy_out",
                    device="cuda",
                    dtype="bfloat16",
                    param_count=64,
                    param_numel=1024,
                    total_numel=65536,
                    avg_ms=0.100,
                    bandwidth_gb_s=10.0,
                ),
            )
        )

        self.assertIn("view_assign", table)
        self.assertIn("copy_out", table)
        self.assertIn("GB/s", table)


if __name__ == "__main__":
    unittest.main()
