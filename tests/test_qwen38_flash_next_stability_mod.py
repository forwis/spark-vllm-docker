#!/usr/bin/env python3
"""Behavior tests for the fail-closed Qwen3.8 Flash Next runtime patcher."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PATCHER = PROJECT / "mods/qwen38-flash-next-stability/patch_vllm.py"

SOURCES = {
    "v1/core/single_type_kv_cache_manager.py": '''
def mamba_hit(block_hashes, max_length, kv_cache_group_ids, block_pool,
              kv_cache_spec, drop_eagle_block, alignment_tokens):
    block_hashes = resolve_block_hashes(
        block_hashes,
        block_pool.hash_block_size,
        kv_cache_spec.block_size,
        supports_fine_grained_hash_lookup=True,
        alignment_tokens=alignment_tokens,
    )
''',
    "v1/worker/gpu/model_states/mamba_hybrid.py": '''
def add_request(self, req_index, new_req_data):
    if self._align_mode:
        self._mamba_state_idx_gpu[req_index].fill_(
            (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
        )
''',
    "v1/core/sched/scheduler.py": '''
def init(self, kv_cache_config):
    self.mamba_partial_cache_hit = (
        self.need_mamba_block_aligned_split
        and self.hash_block_size < self.block_size
        and self.kv_cache_manager.coordinator.enable_partial_hash_hits
    )

def split(self):
    block_size = self.cache_config.block_size
''',
    "v1/worker/block_table.py": '''
def legacy(mask, is_local, block_indices, block_table_stride):
    block_numbers = tl.load(
        block_table_ptr + row_offset + block_indices,
        mask=mask & is_local,
        other=0,
    ).to(tl.int64)
    slot_offsets = local_block_offsets % block_size
    slot_ids = block_numbers * block_size + slot_offsets
    slot_ids = tl.where(is_local, slot_ids, PAD_ID)
    tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
''',
    "v1/worker/gpu/block_table.py": '''
def modern(block_indices, block_table_stride):
    block_numbers = tl.load(
        block_table_ptr + req_state_idx * block_table_stride + block_indices,
        mask=is_local,
        other=0,
    )
    slot_ids = block_numbers * kernel_block_size + block_offsets
    if CP_SIZE != 1:
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)

    slot_ids = tl.where(mapping_enabled, slot_ids, PAD_ID)
''',
    "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py": '''
def apply():
    _ = flashinfer_cutlass_fused_moe(
        input=hidden_states,
        use_deepseek_fp8_block_scale=self.use_deepseek_fp8_block_scale,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        use_w4_group_scaling=use_w4_group_scaling,
    )
''',
}


class StabilityModTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for relative, source in SOURCES.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_patcher(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PATCHER), "--vllm-root", str(self.root), *args],
            text=True,
            capture_output=True,
        )

    def test_applies_all_stability_guards_and_is_idempotent(self) -> None:
        first = self.run_patcher()
        self.assertEqual(first.returncode, 0, first.stderr)

        prefix = (self.root / "v1/core/single_type_kv_cache_manager.py").read_text()
        self.assertIn("max_length = max(0, max_length - kv_cache_spec.block_size)", prefix)

        worker = (self.root / "v1/worker/gpu/model_states/mamba_hybrid.py").read_text()
        self.assertIn("mamba_bs = (", worker)
        self.assertIn("else self.cache_config.block_size", worker)
        self.assertIn("// mamba_bs", worker)

        scheduler = (self.root / "v1/core/sched/scheduler.py").read_text()
        self.assertIn("mamba_state_block_sizes = {", scheduler)
        self.assertIn("self.mamba_state_block_size", scheduler)
        self.assertIn("if self.mamba_state_block_size is not None", scheduler)

        legacy = (self.root / "v1/worker/block_table.py").read_text()
        modern = (self.root / "v1/worker/gpu/block_table.py").read_text()
        for source in (legacy, modern):
            self.assertIn("in_range = block_indices < block_table_stride", source)
            self.assertIn("is_local & in_range", source)

        moe = (self.root / "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py").read_text()
        self.assertIn("use_fused_finalize=False", moe)

        before = {p: p.read_bytes() for p in self.root.rglob("*.py")}
        second = self.run_patcher()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("already installed", second.stdout)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.py")})

        check = self.run_patcher("--check")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertIn("already installed", check.stdout)

    def test_check_reports_changes_without_writing(self) -> None:
        before = {p: p.read_bytes() for p in self.root.rglob("*.py")}
        result = self.run_patcher("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compatible:", result.stdout)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.py")})

    def test_unknown_layout_fails_before_any_write(self) -> None:
        target = self.root / "v1/core/sched/scheduler.py"
        target.write_text(target.read_text().replace(
            "block_size = self.cache_config.block_size",
            "block_size = choose_block_size(self.cache_config)",
        ))
        before = {p: p.read_bytes() for p in self.root.rglob("*.py")}
        result = self.run_patcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected exactly one", result.stderr)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.py")})

    def test_corrupt_already_patched_layout_fails_closed(self) -> None:
        first = self.run_patcher()
        self.assertEqual(first.returncode, 0, first.stderr)
        target = self.root / "v1/worker/gpu/model_states/mamba_hybrid.py"
        target.write_text(target.read_text().replace("// mamba_bs", "// 7"))
        before = {p: p.read_bytes() for p in self.root.rglob("*.py")}
        result = self.run_patcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid already-patched", result.stderr)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*.py")})


if __name__ == "__main__":
    unittest.main()
