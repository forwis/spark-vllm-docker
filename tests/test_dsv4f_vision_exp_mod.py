#!/usr/bin/env python3
"""Behavior tests for the launch-time DeepSeek V4 Vision-Exp port."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path


PATCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "mods/dsv4f-vision-exp/patch_dsv4f_vision.py"
)
SPEC = importlib.util.spec_from_file_location("dsv4f_vision_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


class Dsv4fVisionExpModTests(unittest.TestCase):
    """The patcher must be complete, transactional, and repeatable."""

    def make_fixture(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "vllm"
        overlay = PATCHER_PATH.parent / "overlay/vllm"
        files = {
            "model_executor/layers/fused_moe/router/fused_topk_bias_router.py": """
                import torch

                def fused_topk_bias(**kwargs):
                    return (), ()

                class BaseRouter:
                    pass

                class FusedTopKBiasRouter(BaseRouter):
                    def _compute_routing(
                        self,
                        hidden_states: torch.Tensor,
                        router_logits: torch.Tensor,
                        indices_type: torch.dtype | None,
                        *,
                        input_ids: torch.Tensor | None = None,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        \"\"\"Compute routing using fused top-k with bias.\"\"\"
                        topk_weights, topk_ids = fused_topk_bias(
                            hidden_states=hidden_states,
                            gating_output=router_logits,
                            scoring_func=self.scoring_func,
                            e_score_correction_bias=self.e_score_correction_bias.data
                            if self.e_score_correction_bias is not None
                            else None,
                            topk=self.top_k,
                            renormalize=self.renormalize,
                            indices_type=indices_type,
                            input_tokens=input_ids,
                            hash_indices_table=self._hash_indices_table,
                            routed_scaling_factor=self.routed_scaling_factor,
                        )
                        return topk_weights, topk_ids
            """,
            "model_executor/models/registry.py": """
                _MULTIMODAL_MODELS = {
                    # [Decoder-only]
                    \"AriaForConditionalGeneration\": (\"aria\", \"AriaForConditionalGeneration\"),
                }
            """,
            "models/deepseek_v4/common/ops/cache_utils.py": """
                import torch

                WINDOW_SIZE = 128

                def _build_flashinfer_mixed_sparse_indices_kernel():
                    return WINDOW_SIZE
            """,
            "models/deepseek_v4/nvidia/model.py": """
                import torch
                import torch.nn as nn

                class DeepseekV4MoE(nn.Module):
                    def __init__(self, config):
                        super().__init__()
                        self.gate.e_score_correction_bias = None
                        self.gate.tid2eid = None
                        if self.gate.tid2eid is not None:
                            pass
                        self.experts = make_experts(
                            e_score_correction_bias=self.gate.e_score_correction_bias,
                            hash_indices_table=self.gate.tid2eid,
                        )
                        self._sync_fused_moe_metadata()

                    def forward(self, hidden_states, input_ids=None):
                        if self.gate.tid2eid is not None and input_ids is None:
                            raise ValueError("input_ids required")
                        if not self.use_mega_moe:
                            return self._forward_fused_moe(hidden_states, input_ids)

                    def _sync_fused_moe_metadata(self):
                        pass

                class DeepseekV4Model:
                    def load_weights(self, weights):
                        params_dict = {}
                        pad_shared_expert = False
                        for name, loaded_weight in weights:
                            if pad_shared_expert and \".shared_experts.\" in name:
                                pass

                def get_deepseek_v4_weights_mapper():
                    return dict(
                        orig_to_new_suffix={
                            \".ffn.gate.bias\": \".ffn.gate.e_score_correction_bias\",
                        },
                    )
            """,
            "models/deepseek_v4/nvidia/dspark.py": """
                class DSpark:
                    def load_weights(self, weights):
                        params_dict = {}
                        if True:
                            for name, loaded_weight in weights:
                                if name.endswith(\".ffn.gate.bias\"):
                                    name = name.replace(
                                        \".ffn.gate.bias\",
                                        \".ffn.gate.e_score_correction_bias\",
                                    )
                                param = params_dict[name]
            """,
            "v1/engine/input_processor.py": """
                class VLLMValidationError(Exception):
                    pass

                class InputProcessor:
                    def validate(self, prompt_ids, tokenizer, model_config):
                        max_input_id = max(prompt_ids, default=0)
                        model_vocab_size = model_config.get_vocab_size()
                        if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                            raise VLLMValidationError(
                                f\"Token id {max_input_id} is out of vocabulary\"
                            )
            """,
        }
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(textwrap.dedent(content).lstrip())
        return root, overlay

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text()

    @staticmethod
    def hash_tree(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def test_patch_tree_installs_complete_port_idempotently(self):
        self.root, overlay = self.make_fixture()
        changed = PATCHER.patch_tree(self.root, overlay)
        self.assertEqual(
            {path.relative_to(self.root).as_posix() for path in changed},
            {
                "model_executor/layers/fused_moe/router/fused_topk_bias_router.py",
                "model_executor/models/registry.py",
                "models/deepseek_v4/common/ops/cache_utils.py",
                "models/deepseek_v4/nvidia/model.py",
                "models/deepseek_v4/nvidia/dspark.py",
                "v1/engine/input_processor.py",
                "models/deepseek_v4/mm_preprocess.py",
                "models/deepseek_v4/vision.py",
                "models/deepseek_v4/vision_model.py",
            },
        )

        self.assertIn("DeepseekV4VForConditionalGeneration", self.read("model_executor/models/registry.py"))
        self.assertIn("def _compute_routing_vision", self.read("model_executor/layers/fused_moe/router/fused_topk_bias_router.py"))
        self.assertIn("self._sync_fused_moe_metadata()", self.read("models/deepseek_v4/nvidia/model.py"))
        self.assertIn("_router.bias_vl = self.gate.bias_vl", self.read("models/deepseek_v4/nvidia/model.py"))
        self.assertIn("WINDOW_SIZE", self.read("models/deepseek_v4/common/ops/cache_utils.py"))
        self.assertIn("def compute_vision_visible_window", self.read("models/deepseek_v4/common/ops/cache_utils.py"))
        self.assertIn("model_vocab_size + 4", self.read("v1/engine/input_processor.py"))
        model = self.read("models/deepseek_v4/nvidia/model.py")
        self.assertIn("self.gate.bias_vl = None", model)
        self.assertIn("input_ids = torch.where(", model)
        self.assertIn("if not self.use_mega_moe:", model)
        self.assertNotIn('".ffn.gate.bias_vl": ".ffn.gate.e_score_correction_bias_vl"', model)
        dspark = self.read("models/deepseek_v4/nvidia/dspark.py")
        self.assertIn('name.endswith(".ffn.gate.bias_vl")', dspark)
        self.assertIn("if name not in params_dict", dspark)
        self.assertNotIn("e_score_correction_bias_vl", dspark)

        before = self.hash_tree(self.root)
        self.assertEqual(PATCHER.patch_tree(self.root, overlay), [])
        self.assertEqual(self.hash_tree(self.root), before)

    def test_check_mode_validates_without_writing(self):
        root, overlay = self.make_fixture()
        before = self.hash_tree(root)
        expected = PATCHER.patch_tree(root, overlay, check=True)
        self.assertEqual(len(expected), 9)
        self.assertEqual(self.hash_tree(root), before)

    def test_nonmega_forward_keeps_raw_ids_before_mega_guard(self):
        self.root, overlay = self.make_fixture()
        PATCHER.patch_tree(self.root, overlay)
        model = self.read("models/deepseek_v4/nvidia/model.py")
        raw_return = model.index("return self._forward_fused_moe(hidden_states, input_ids)")
        masked_ids = model.index("input_ids = torch.where(")
        self.assertLess(raw_return, masked_ids)
        router = self.read(
            "model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
        )
        self.assertIn("image_mask = input_ids >= self.vl_vocab_size", router)

    def test_unknown_anchor_fails_without_partial_writes(self):
        root, overlay = self.make_fixture()
        target = root / "models/deepseek_v4/nvidia/dspark.py"
        target.write_text("class UnknownDSpark: pass\n")
        before = self.hash_tree(root)
        with self.assertRaisesRegex(PATCHER.PatchError, "dspark bias loader"):
            PATCHER.patch_tree(root, overlay)
        self.assertEqual(self.hash_tree(root), before)

    def test_foreign_overlay_collision_is_rejected(self):
        root, overlay = self.make_fixture()
        target = root / "models/deepseek_v4/vision.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("FOREIGN = True\n")
        with self.assertRaisesRegex(PATCHER.PatchError, "refusing to overwrite"):
            PATCHER.patch_tree(root, overlay)


if __name__ == "__main__":
    unittest.main()
