#!/usr/bin/env python3
"""Apply fail-closed Qwen3.8 Flash Next stability fixes to installed vLLM."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path


class PatchError(RuntimeError):
    """The installed vLLM tree does not match the qualified source layout."""


def validate_patched(text: str, label: str, marker: str, *required: str) -> str:
    if text.count(marker) != 1 or any(text.count(block) != 1 for block in required):
        raise PatchError(f"invalid already-patched {label}")
    return text


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise PatchError(f"expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def shifted(text: str, spaces: int) -> str:
    padding = " " * spaces
    return "\n".join(padding + line if line else line for line in text.split("\n"))


def replace_indented(
    text: str, old: str, new: str, label: str, *extra_indents: int
) -> str:
    if new in text or any(shifted(new, n) in text for n in extra_indents):
        return text
    matches = [(old, new)] + [
        (shifted(old, n), shifted(new, n)) for n in extra_indents
    ]
    present = [(candidate, replacement) for candidate, replacement in matches if candidate in text]
    if len(present) != 1:
        raise PatchError(f"expected exactly one {label}; found {len(present)}")
    candidate, replacement = present[0]
    if text.count(candidate) != 1:
        raise PatchError(f"expected exactly one {label}; found {text.count(candidate)}")
    return text.replace(candidate, replacement, 1)


def patch_mamba_cache_hit(text: str) -> str:
    marker = "# qwen38-stability: reserve the EAGLE/MTP replay block"
    if marker in text:
        return validate_patched(
            text,
            "Mamba cache-hit guard",
            marker,
            "if drop_eagle_block:",
            "max_length = max(0, max_length - kv_cache_spec.block_size)",
        )

    start = text.find("class MambaManager(")
    if start >= 0:
        end = text.find("\nclass ", start + 1)
        end = len(text) if end < 0 else end
        prefix, section, suffix = text[:start], text[start:end], text[end:]
    else:
        prefix, section, suffix = "", text, ""
    old = """        block_hashes = resolve_block_hashes(
"""
    new = """        # qwen38-stability: reserve the EAGLE/MTP replay block
        if drop_eagle_block:
            max_length = max(0, max_length - kv_cache_spec.block_size)
        block_hashes = resolve_block_hashes(
"""
    if old in section:
        section = replace_once(section, old, new, "Mamba cache-hit resolver")
    else:
        old_fixture = "    block_hashes = resolve_block_hashes(\n"
        new_fixture = """    # qwen38-stability: reserve the EAGLE/MTP replay block
    if drop_eagle_block:
        max_length = max(0, max_length - kv_cache_spec.block_size)
    block_hashes = resolve_block_hashes(
"""
        section = replace_once(
            section, old_fixture, new_fixture, "Mamba cache-hit resolver"
        )
    return prefix + section + suffix


def patch_mamba_state_seed(text: str) -> str:
    marker = "# qwen38-stability: seed on the Mamba state grid"
    old = """        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )
"""
    old_fixture = """    if self._align_mode:
        self._mamba_state_idx_gpu[req_index].fill_(
            (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
        )
"""
    new = """        if self._align_mode:
            # qwen38-stability: seed on the Mamba state grid
            mamba_bs = (
                self._mamba_spec.block_size
                if self._mamba_spec is not None
                else self.cache_config.block_size
            )
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_bs
            )
"""
    new_fixture = """    if self._align_mode:
        # qwen38-stability: seed on the Mamba state grid
        mamba_bs = (
            self._mamba_spec.block_size
            if self._mamba_spec is not None
            else self.cache_config.block_size
        )
        self._mamba_state_idx_gpu[req_index].fill_(
            (new_req_data.num_computed_tokens - 1) // mamba_bs
        )
"""
    if marker in text:
        return validate_patched(
            text,
            "Mamba state seed",
            marker,
            "mamba_bs = (",
            "if self._mamba_spec is not None",
            "// mamba_bs",
        )
    if old in text:
        return replace_once(text, old, new, "Mamba state seed")
    return replace_once(text, old_fixture, new_fixture, "Mamba state seed")


def patch_scheduler(text: str) -> str:
    marker = "# qwen38-stability: schedule on the Mamba state grid"
    if marker in text:
        return validate_patched(
            text,
            "Mamba scheduler alignment",
            marker,
            "mamba_state_block_sizes = {",
            "self.mamba_state_block_size = (",
            "if self.mamba_state_block_size is not None",
        )
    init_old = """    self.mamba_partial_cache_hit = (
        self.need_mamba_block_aligned_split
        and self.hash_block_size < self.block_size
        and self.kv_cache_manager.coordinator.enable_partial_hash_hits
    )
"""
    init_real = init_old.replace("    ", "        ", 1).replace("\n    ", "\n        ")
    body = """    # qwen38-stability: schedule on the Mamba state grid
    mamba_state_block_sizes = {
        group.kv_cache_spec.block_size
        for group in kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    }
    assert len(mamba_state_block_sizes) <= 1, (
        "mamba align scheduling requires one Mamba state block size, "
        f"got {sorted(mamba_state_block_sizes)}"
    )
    self.mamba_state_block_size = (
        next(iter(mamba_state_block_sizes)) if mamba_state_block_sizes else None
    )
"""
    if init_real in text:
        real_body = body.replace("    ", "        ", 1).replace("\n    ", "\n        ")
        text = replace_once(
            text, init_real, init_real + real_body, "scheduler Mamba initialization"
        )
    else:
        text = replace_once(text, init_old, init_old + body, "scheduler Mamba initialization")
    split_old = """    block_size = self.cache_config.block_size
"""
    split_new = """    block_size = (
        self.mamba_state_block_size
        if self.mamba_state_block_size is not None
        else self.cache_config.block_size
    )
"""
    return replace_once(text, split_old, split_new, "scheduler Mamba split block size")


def patch_legacy_slot_mapping(text: str) -> str:
    old = """    block_numbers = tl.load(
        block_table_ptr + row_offset + block_indices,
        mask=mask & is_local,
        other=0,
    ).to(tl.int64)
    slot_offsets = local_block_offsets % block_size
    slot_ids = block_numbers * block_size + slot_offsets
    slot_ids = tl.where(is_local, slot_ids, PAD_ID)
"""
    new = """    # qwen38-stability: never read beyond a request's block-table row
    in_range = block_indices < block_table_stride
    block_numbers = tl.load(
        block_table_ptr + row_offset + block_indices,
        mask=mask & is_local & in_range,
        other=0,
    ).to(tl.int64)
    slot_offsets = local_block_offsets % block_size
    slot_ids = block_numbers * block_size + slot_offsets
    slot_ids = tl.where(is_local & in_range, slot_ids, PAD_ID)
"""
    return replace_indented(text, old, new, "legacy slot-mapping load", 8)


def patch_modern_slot_mapping(text: str) -> str:
    old = """    block_numbers = tl.load(
        block_table_ptr + req_state_idx * block_table_stride + block_indices,
        mask=is_local,
        other=0,
    )
    slot_ids = block_numbers * kernel_block_size + block_offsets
    if CP_SIZE != 1:
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
"""
    new = """    # qwen38-stability: never read beyond a request's block-table row
    in_range = block_indices < block_table_stride
    block_numbers = tl.load(
        block_table_ptr + req_state_idx * block_table_stride + block_indices,
        mask=is_local & in_range,
        other=0,
    )
    slot_ids = block_numbers * kernel_block_size + block_offsets
    slot_ids = tl.where(is_local & in_range, slot_ids, PAD_ID)
"""
    return replace_indented(text, old, new, "Model Runner V2 slot-mapping load", 4)


def patch_moe_finalize(text: str) -> str:
    old = """        use_w4_group_scaling=use_w4_group_scaling,
    )
"""
    new = """        use_w4_group_scaling=use_w4_group_scaling,
        # qwen38-stability: deterministic NVFP4 reduction on GB10/SM12x
        use_fused_finalize=False,
    )
"""
    return replace_indented(text, old, new, "FlashInfer CUTLASS MoE call", 4)


PATCHERS: dict[str, Callable[[str], str]] = {
    "v1/core/single_type_kv_cache_manager.py": patch_mamba_cache_hit,
    "v1/worker/gpu/model_states/mamba_hybrid.py": patch_mamba_state_seed,
    "v1/core/sched/scheduler.py": patch_scheduler,
    "v1/worker/block_table.py": patch_legacy_slot_mapping,
    "v1/worker/gpu/block_table.py": patch_modern_slot_mapping,
    "model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py": patch_moe_finalize,
}


def patch_tree(vllm_root: Path, *, check: bool = False) -> list[Path]:
    planned: dict[Path, str] = {}
    for relative, transform in PATCHERS.items():
        target = vllm_root / relative
        if not target.is_file():
            raise PatchError(f"required qualified vLLM source is missing: {target}")
        original = target.read_text()
        patched = transform(original)
        compile(patched, relative, "exec")
        if patched != original:
            planned[target] = patched

    if not check:
        for target, content in planned.items():
            temporary = target.with_suffix(target.suffix + ".qwen38-stability.tmp")
            temporary.write_text(content)
            temporary.replace(target)
    return sorted(planned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        changed = patch_tree(args.vllm_root, check=args.check)
    except (PatchError, SyntaxError) as error:
        parser.error(str(error))
    if not changed:
        print("already installed")
    else:
        state = "compatible" if args.check else "changed"
        for path in changed:
            print(f"{state}: {path.relative_to(args.vllm_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
