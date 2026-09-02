#!/usr/bin/env python3
"""Transactional launch-time port for DeepSeek V4 Flash Vision-Exp."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path


class PatchError(RuntimeError):
    """The installed vLLM tree is not the qualified JASL v2 source layout."""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """Replace one qualified source anchor, accepting an already-patched result."""
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise PatchError(f"expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def _compiled_exact(text: str, relative: str, marker: str, *blocks: str) -> str:
    compile(text, relative, "exec")
    if text.count(marker) != 1 or any(text.count(block) != 1 for block in blocks):
        raise PatchError(f"invalid already-patched {relative}")
    return text


def patch_router(text: str) -> str:
    relative = "model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
    marker = "# dsv4f-vision-exp: modality-specific fused routing"
    old = '''        \"\"\"Compute routing using fused top-k with bias.\"\"\"
        topk_weights, topk_ids = fused_topk_bias(
'''
    new = '''        \"\"\"Compute routing using fused top-k with bias.\"\"\"
        # dsv4f-vision-exp: modality-specific fused routing
        bias_vl = getattr(self, "bias_vl", None)
        if bias_vl is not None and input_ids is not None:
            topk_weights, topk_ids = self._compute_routing_vision(
                router_logits, input_ids, indices_type
            )
            return self._append_fused_shared_experts(topk_weights, topk_ids)
        topk_weights, topk_ids = fused_topk_bias(
'''
    anchor = '''    def _compute_routing(
'''
    method = '''    def _compute_routing_vision(
        self,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Route Vision sentinels with the checkpoint's ``bias_vl``.\"\"\"
        scores = router_logits.to(torch.float32)
        scores = torch.nn.functional.softplus(scores).sqrt()
        original_scores = scores
        image_mask = input_ids >= self.vl_vocab_size
        if self._hash_indices_table is not None:
            safe_ids = torch.where(
                image_mask, torch.zeros_like(input_ids), input_ids
            ).to(dtype=self._hash_indices_table.dtype)
            indices = self._hash_indices_table[safe_ids]
            vl_indices = (scores + self.bias_vl).topk(self.top_k, dim=-1)[1]
            indices = torch.where(
                image_mask.unsqueeze(-1),
                vl_indices.to(indices.dtype),
                indices,
            )
        else:
            bias = torch.where(
                image_mask.unsqueeze(-1),
                self.bias_vl,
                self.e_score_correction_bias,
            )
            indices = (scores + bias).topk(self.top_k, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.routed_scaling_factor
        if indices_type is not None:
            indices = indices.to(indices_type)
        return weights, indices

'''
    helper = '''    def _append_fused_shared_experts(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_fused_shared_experts > 0:
            m = topk_ids.shape[0]
            n = self.num_fused_shared_experts
            # global_num_experts counts only the routed experts; the fused
            # shared experts occupy the slots immediately after them, i.e. ids
            # [global_num_experts, global_num_experts + n).
            base = self.global_num_experts
            shared_ids = torch.arange(
                base, base + n, dtype=topk_ids.dtype, device=topk_ids.device
            ).expand(m, n)
            shared_w = torch.full(
                (m, n),
                self.shared_expert_weight,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
            topk_ids = torch.cat([topk_ids, shared_ids], dim=-1)
            topk_weights = torch.cat([topk_weights, shared_w], dim=-1)

        return topk_weights, topk_ids

'''
    stock_append = '''
        if self.num_fused_shared_experts > 0:
            m = topk_ids.shape[0]
            n = self.num_fused_shared_experts
            # global_num_experts counts only the routed experts; the fused
            # shared experts occupy the slots immediately after them, i.e. ids
            # [global_num_experts, global_num_experts + n).
            base = self.global_num_experts
            shared_ids = torch.arange(
                base, base + n, dtype=topk_ids.dtype, device=topk_ids.device
            ).expand(m, n)
            shared_w = torch.full(
                (m, n),
                self.shared_expert_weight,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
            topk_ids = torch.cat([topk_ids, shared_ids], dim=-1)
            topk_weights = torch.cat([topk_weights, shared_w], dim=-1)

        return topk_weights, topk_ids
'''
    shared_return = '''
        return self._append_fused_shared_experts(topk_weights, topk_ids)
'''
    if marker in text:
        return _compiled_exact(
            text,
            relative,
            marker,
            new,
            method,
            helper,
            shared_return,
        )
    result = replace_once(text, old, new, "fused router dispatch")
    result = replace_once(
        result,
        stock_append,
        shared_return,
        "fused shared-expert append",
    )
    result = replace_once(
        result,
        anchor,
        method + helper + anchor,
        "fused router methods",
    )
    return _compiled_exact(
        result,
        relative,
        marker,
        new,
        method,
        helper,
        shared_return,
    )


def patch_registry(text: str) -> str:
    relative = "model_executor/models/registry.py"
    marker = "# dsv4f-vision-exp: DeepSeek V4 Vision-Exp"
    old = '''_MULTIMODAL_MODELS = {
    # [Decoder-only]
'''
    new = '''_MULTIMODAL_MODELS = {
    # dsv4f-vision-exp: DeepSeek V4 Vision-Exp
    "DeepseekV4VForConditionalGeneration": (
        "vllm.models.deepseek_v4.vision_model",
        "DeepseekV4VForConditionalGeneration",
    ),
    # [Decoder-only]
'''
    if marker in text:
        return _compiled_exact(text, relative, marker, new)
    return _compiled_exact(
        replace_once(text, old, new, "multimodal registry"),
        relative,
        marker,
        new,
    )


def patch_model(text: str) -> str:
    relative = "models/deepseek_v4/nvidia/model.py"
    marker = "# dsv4f-vision-exp: Vision gate state"
    gate_old = '''        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
'''
    gate_new = '''        self.gate.e_score_correction_bias = None
        # dsv4f-vision-exp: Vision gate state
        self.vl_vocab_size = config.vocab_size
        self.gate.bias_vl = None
        if int(getattr(config, "vision_n_layers", 0) or 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        self.gate.tid2eid = None
'''
    sync_old = '''        self._sync_fused_moe_metadata()
'''
    sync_new = '''        self._sync_fused_moe_metadata()
        if self.gate.bias_vl is not None:
            _router = getattr(self.experts, "router", None)
            if _router is not None:
                _router.bias_vl = self.gate.bias_vl
                _router.vl_vocab_size = self.vl_vocab_size
'''
    load_old = '''        for name, loaded_weight in weights:
            if pad_shared_expert and ".shared_experts." in name:
'''
    load_new = '''        for name, loaded_weight in weights:
            if (
                name.endswith(".ffn.gate.e_score_correction_bias")
                and name not in params_dict
            ):
                continue
            if pad_shared_expert and ".shared_experts." in name:
'''
    guard_old = '''        if not self.use_mega_moe:
            return self._forward_fused_moe(hidden_states, input_ids)

'''
    guard_new = '''        if not self.use_mega_moe:
            return self._forward_fused_moe(hidden_states, input_ids)

        # Image sentinel ids are outside the hash table's vocabulary.
        # Keep this replacement branch-free for torch.compile on the
        # qualified mega-MoE path; non-mega routing already returned raw ids.
        image_mask = None
        if input_ids is not None and getattr(self.gate, "bias_vl", None) is not None:
            image_mask = input_ids >= self.vl_vocab_size
            input_ids = torch.where(
                image_mask, torch.zeros_like(input_ids), input_ids)
'''
    if marker in text:
        return _compiled_exact(
            text,
            relative,
            marker,
            gate_new,
            sync_new,
            load_new,
            guard_new,
        )
    result = replace_once(
        text,
        gate_old,
        gate_new,
        "Vision gate initialization",
    )
    result = replace_once(
        result,
        sync_old,
        sync_new,
        "JASL fused MoE metadata synchronization",
    )
    result = replace_once(
        result,
        load_old,
        load_new,
        "hash-gate routing bias loader",
    )
    result = replace_once(
        result,
        guard_old,
        guard_new,
        "mega-MoE image sentinel guard",
    )
    return _compiled_exact(
        result,
        relative,
        marker,
        gate_new,
        sync_new,
        load_new,
        guard_new,
    )


def patch_dspark(text: str) -> str:
    relative = "models/deepseek_v4/nvidia/dspark.py"
    marker = "# dsv4f-vision-exp: DSpark modality-specific gate bias"
    old = '''                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                param = params_dict[name]
'''
    new = '''                # dsv4f-vision-exp: DSpark modality-specific gate bias
                if name.endswith(".ffn.gate.bias_vl"):
                    pass
                elif name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                if name not in params_dict:
                    continue
                param = params_dict[name]
'''
    if marker in text:
        return _compiled_exact(text, relative, marker, new)
    return _compiled_exact(
        replace_once(text, old, new, "dspark bias loader"),
        relative,
        marker,
        new,
    )


def patch_input_processor(text: str) -> str:
    relative = "v1/engine/input_processor.py"
    marker = "# dsv4f-vision-exp: Vision sentinel validation"
    # Keep the source's indentation because the fixture intentionally uses a
    # compact validator while JASL nests this block under ``if prompt_ids``.
    max_line = re.compile(r"(?m)^(?P<indent>[ \t]*)max_input_id = max\(prompt_ids, default=0\)$")
    matches = list(max_line.finditer(text))
    if len(matches) != 1:
        raise PatchError(f"expected exactly one input token bounds; found {len(matches)}")
    match = matches[0]
    indent = match.group("indent")
    min_block = (
        f"{indent}min_input_id = min(prompt_ids, default=0)\n"
        f"{indent}if min_input_id < 0:\n"
        f"{indent}    raise VLLMValidationError(\n"
        f"{indent}        f\"Token id {{min_input_id}} is out of vocabulary\"\n"
        f"{indent}    )\n"
        f"{indent}max_input_id = max(prompt_ids, default=0)"
    )
    model_line = re.compile(
        r"(?m)^(?P<indent>[ \t]*)model_vocab_size = model_config\.get_vocab_size\(\)$"
    )
    matches = list(model_line.finditer(text))
    if len(matches) != 1:
        raise PatchError(
            f"expected exactly one input model vocabulary lookup; found {len(matches)}"
        )
    match = matches[0]
    indent = match.group("indent")
    allowed = (
        f"{match.group(0)}\n"
        f"{indent}# dsv4f-vision-exp: Vision sentinel validation\n"
        f"{indent}allowed_max = max(tokenizer.max_token_id, model_vocab_size - 1)\n"
        f"{indent}hf_cfg = getattr(model_config, \"hf_config\", None)\n"
        f"{indent}if int(getattr(hf_cfg, \"vision_n_layers\", 0) or 0) > 0:\n"
        f"{indent}    allowed_max = max(allowed_max, model_vocab_size + 4)"
    )
    comparison = "max_input_id > allowed_max"
    if marker in text:
        return _compiled_exact(
            text,
            relative,
            marker,
            min_block,
            allowed,
            comparison,
        )

    max_match = list(max_line.finditer(text))[0]
    result = (
        text[: max_match.start()]
        + min_block
        + text[max_match.end() :]
    )
    matches = list(model_line.finditer(result))
    match = matches[0]
    result = result[: match.start()] + allowed + result[match.end() :]
    result = replace_once(
        result,
        "max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1)",
        comparison,
        "input maximum token validation",
    )
    return _compiled_exact(
        result,
        relative,
        marker,
        min_block,
        allowed,
        comparison,
    )


PATCHERS: dict[str, Callable[[str], str]] = {
    "model_executor/layers/fused_moe/router/fused_topk_bias_router.py": patch_router,
    "model_executor/models/registry.py": patch_registry,
    "models/deepseek_v4/nvidia/model.py": patch_model,
    "models/deepseek_v4/nvidia/dspark.py": patch_dspark,
    "v1/engine/input_processor.py": patch_input_processor,
}

OVERLAY_FILES = (
    "models/deepseek_v4/mm_preprocess.py",
    "models/deepseek_v4/vision.py",
    "models/deepseek_v4/vision_model.py",
)


def patch_tree(vllm_root: Path, overlay_root: Path, *, check: bool = False) -> list[Path]:
    """Validate the complete port first, then atomically install every change."""
    planned: dict[Path, str] = {}
    for relative, transform in PATCHERS.items():
        target = vllm_root / relative
        if not target.is_file():
            raise PatchError(f"required JASL v2 source is missing: {target}")
        original = target.read_text()
        patched = transform(original)
        if patched != original:
            planned[target] = patched
    for relative in OVERLAY_FILES:
        source = overlay_root / relative
        target = vllm_root / relative
        if not source.is_file():
            raise PatchError(f"required Vision overlay is missing: {source}")
        desired = source.read_text()
        compile(desired, relative, "exec")
        if target.exists() and target.read_text() != desired:
            raise PatchError(f"refusing to overwrite foreign Vision module: {target}")
        if not target.exists():
            planned[target] = desired
    if check:
        return sorted(planned)
    for target, content in planned.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".dsv4f-vision.tmp")
        temporary.write_text(content)
        temporary.replace(target)
    return sorted(planned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        changed = patch_tree(args.vllm_root, args.overlay_root, check=args.check)
    except (PatchError, SyntaxError) as error:
        parser.error(str(error))
    state = "compatible" if args.check else "changed"
    if not changed:
        print("already installed")
    else:
        for path in changed:
            print(f"{state}: {path.relative_to(args.vllm_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
