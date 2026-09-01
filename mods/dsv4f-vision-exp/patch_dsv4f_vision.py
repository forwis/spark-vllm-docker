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


def _compiled(text: str, relative: str, marker: str, *required: str) -> str:
    compile(text, relative, "exec")
    if marker not in text or any(value not in text for value in required):
        raise PatchError(f"invalid already-patched {relative}")
    return text


def patch_router(text: str) -> str:
    relative = "model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
    marker = "# dsv4f-vision-exp: modality-specific fused routing"
    if marker in text:
        return _compiled(text, relative, marker, "def _compute_routing_vision", "return self._compute_routing_vision")
    old = '''        \"\"\"Compute routing using fused top-k with bias.\"\"\"
        topk_weights, topk_ids = fused_topk_bias(
'''
    new = '''        \"\"\"Compute routing using fused top-k with bias.\"\"\"
        # dsv4f-vision-exp: modality-specific fused routing
        bias_vl = getattr(self, "bias_vl", None)
        if bias_vl is not None and input_ids is not None:
            return self._compute_routing_vision(router_logits, input_ids, indices_type)
        topk_weights, topk_ids = fused_topk_bias(
'''
    result = replace_once(text, old, new, "fused router dispatch")
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
    result = replace_once(result, anchor, method + anchor, "fused router method")
    return _compiled(result, relative, marker, "def _compute_routing_vision", "return self._compute_routing_vision")


def patch_registry(text: str) -> str:
    relative = "model_executor/models/registry.py"
    marker = "# dsv4f-vision-exp: DeepSeek V4 Vision-Exp"
    if marker in text:
        return _compiled(text, relative, marker, "DeepseekV4VForConditionalGeneration")
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
    return _compiled(replace_once(text, old, new, "multimodal registry"), relative, marker, "DeepseekV4VForConditionalGeneration")


def patch_cache_utils(text: str) -> str:
    relative = "models/deepseek_v4/common/ops/cache_utils.py"
    marker = "# dsv4f-vision-exp: Vision-visible sparse attention window"
    if marker in text:
        return _compiled(text, relative, marker, "def compute_vision_visible_window", "WINDOW_SIZE")
    if text.count("def _build_flashinfer_mixed_sparse_indices_kernel") != 1:
        raise PatchError("expected exactly one cache utility kernel; found " + str(text.count("def _build_flashinfer_mixed_sparse_indices_kernel")))
    result = text.rstrip() + '''


# dsv4f-vision-exp: Vision-visible sparse attention window
def compute_vision_visible_window(
    input_ids: torch.Tensor,
    vocab_size: int,
    window_size: int,
    max_image_tokens: int = 384,
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Return per-token visible extents for Vision image spans.\"\"\"
    idx = torch.arange(input_ids.shape[-1], dtype=torch.int32, device=input_ids.device)
    is_start = input_ids == (vocab_size + 0)
    is_end = input_ids == (vocab_size + 4)
    valid = (is_start.cumsum(-1) > is_end.cumsum(-1)) | is_end
    starts = torch.where(is_start, idx, torch.zeros_like(idx)).cummax(-1)[0]
    left = (idx - starts) * valid
    ends = (
        torch.where(is_end, idx, torch.full_like(idx, input_ids.shape[-1]))
        .flip(-1)
        .cummin(-1)[0]
        .flip(-1)
    )
    right = (ends - idx) * valid
    left = left.clamp(max=max_image_tokens - 1)
    right = right.clamp(max=max_image_tokens)
    return left.to(torch.int32), right.to(torch.int32)
'''
    return _compiled(result, relative, marker, "def compute_vision_visible_window", "WINDOW_SIZE")


def patch_model(text: str) -> str:
    relative = "models/deepseek_v4/nvidia/model.py"
    marker = "# dsv4f-vision-exp: Vision gate state"
    if marker in text:
        return _compiled(text, relative, marker, "_router.bias_vl = self.gate.bias_vl", "e_score_correction_bias_vl")
    result = replace_once(
        text,
        '''        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
''',
        '''        self.gate.e_score_correction_bias = None
        # dsv4f-vision-exp: Vision gate state
        self.vl_vocab_size = config.vocab_size
        self.gate.bias_vl = None
        if int(getattr(config, "vision_n_layers", 0) or 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        self.gate.tid2eid = None
''',
        "Vision gate initialization",
    )
    result = replace_once(
        result,
        '''        self._sync_fused_moe_metadata()
''',
        '''        self._sync_fused_moe_metadata()
        if self.gate.bias_vl is not None:
            _router = getattr(self.experts, "router", None)
            if _router is not None:
                _router.bias_vl = self.gate.bias_vl
                _router.vl_vocab_size = self.vl_vocab_size
''',
        "JASL fused MoE metadata synchronization",
    )
    result = replace_once(
        result,
        '''        for name, loaded_weight in weights:
            if pad_shared_expert and ".shared_experts." in name:
''',
        '''        for name, loaded_weight in weights:
            if (
                name.endswith(".ffn.gate.e_score_correction_bias")
                and name not in params_dict
            ):
                continue
            if pad_shared_expert and ".shared_experts." in name:
''',
        "hash-gate routing bias loader",
    )
    result = replace_once(
        result,
        '''            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
''',
        '''            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
            ".ffn.gate.bias_vl": ".ffn.gate.e_score_correction_bias_vl",
''',
        "Vision weights mapper suffix",
    )
    return _compiled(result, relative, marker, "_router.bias_vl = self.gate.bias_vl", "e_score_correction_bias_vl")


def patch_dspark(text: str) -> str:
    relative = "models/deepseek_v4/nvidia/dspark.py"
    marker = "# dsv4f-vision-exp: DSpark modality-specific gate bias"
    if marker in text:
        return _compiled(text, relative, marker, ".ffn.gate.e_score_correction_bias_vl", "if name not in params_dict")
    old = '''                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                param = params_dict[name]
'''
    new = '''                # dsv4f-vision-exp: DSpark modality-specific gate bias
                if name.endswith(".ffn.gate.bias_vl"):
                    name = name.replace(
                        ".ffn.gate.bias_vl",
                        ".ffn.gate.e_score_correction_bias_vl",
                    )
                elif name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias",
                        ".ffn.gate.e_score_correction_bias",
                    )
                if name not in params_dict:
                    continue
                param = params_dict[name]
'''
    return _compiled(replace_once(text, old, new, "dspark bias loader"), relative, marker, ".ffn.gate.e_score_correction_bias_vl", "if name not in params_dict")


def patch_input_processor(text: str) -> str:
    relative = "v1/engine/input_processor.py"
    marker = "# dsv4f-vision-exp: Vision sentinel validation"
    if marker in text:
        return _compiled(text, relative, marker, "model_vocab_size + 4", "min_input_id")
    # Keep the source's indentation because the fixture intentionally uses a
    # compact validator while JASL nests this block under ``if prompt_ids``.
    max_line = re.compile(r"(?m)^(?P<indent>[ \t]*)max_input_id = max\(prompt_ids, default=0\)$")
    matches = list(max_line.finditer(text))
    if len(matches) != 1:
        raise PatchError(f"expected exactly one input token bounds; found {len(matches)}")
    match = matches[0]
    indent = match.group("indent")
    new = (
        f"{indent}min_input_id = min(prompt_ids, default=0)\n"
        f"{indent}if min_input_id < 0:\n"
        f"{indent}    raise VLLMValidationError(\n"
        f"{indent}        f\"Token id {{min_input_id}} is out of vocabulary\"\n"
        f"{indent}    )\n"
        f"{indent}max_input_id = max(prompt_ids, default=0)"
    )
    result = text[: match.start()] + new + text[match.end() :]
    model_line = re.compile(
        r"(?m)^(?P<indent>[ \t]*)model_vocab_size = model_config\.get_vocab_size\(\)$"
    )
    matches = list(model_line.finditer(result))
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
    result = result[: match.start()] + allowed + result[match.end() :]
    result = replace_once(
        result,
        "max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1)",
        "max_input_id > allowed_max",
        "input maximum token validation",
    )
    return _compiled(result, relative, marker, "model_vocab_size + 4", "min_input_id")


PATCHERS: dict[str, Callable[[str], str]] = {
    "model_executor/layers/fused_moe/router/fused_topk_bias_router.py": patch_router,
    "model_executor/models/registry.py": patch_registry,
    "models/deepseek_v4/common/ops/cache_utils.py": patch_cache_utils,
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
