#!/usr/bin/env python3
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "mods/glm53-fp8-ds-mla/patch_mla.py"
EXPECTED_SHA256 = "12ed6565328c8b72edae62207b2640436355475d033729046170cdf5db96954f"

MLA_FIXTURE = '''\
class Layer:
    def __init__(self, qk_nope_head_dim, qk_rope_head_dim):
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.mla_attn = dict(
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )

    def forward(self, q, k_pe):
        attn_out = self.mla_attn(
            q=q,
            k_pe=k_pe,
        )
        return attn_out
'''

SM120_FIXTURE = '''\
class Impl:
    def __init__(self, vllm_config, model_type):
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

    def forward(self, topk_indices, topk_indices_physical, attn_metadata):
        call(
            max_seq_len=attn_metadata.topk_tokens,
            sparse_mla_top_k=attn_metadata.topk_tokens,
        )
        topk_indices_physical = convert(
            triton_convert(
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )
        return topk_indices_physical
'''

FLASHINFER_FIXTURE = '''\
_DECODE_DSV3_2_DISPATCH = frozenset()
_DECODE_DSV3_2_PAGE_BLOCK_SIZE = 64
'''

TARGET_LITERALS = {
    'P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla.py"': "P",
    'P2 = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"': "P2",
    'P3 = "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py"': "P3",
}


def prepare_case(root: Path, mla_source: str = MLA_FIXTURE):
    mla = root / "mla.py"
    sm120 = root / "flashinfer_mla_sparse_sm120.py"
    flashinfer = root / "_sparse_mla_sm120.py"
    mla.write_text(mla_source)
    sm120.write_text(SM120_FIXTURE)
    flashinfer.write_text(FLASHINFER_FIXTURE)

    replacements = {
        "P": f"P = {str(mla)!r}",
        "P2": f"P2 = {str(sm120)!r}",
        "P3": f"P3 = {str(flashinfer)!r}",
    }
    relocated_text = PATCH.read_text()
    for literal, key in TARGET_LITERALS.items():
        assert relocated_text.count(literal) == 1
        relocated_text = relocated_text.replace(literal, replacements[key])
    relocated = root / "patch_mla.py"
    relocated.write_text(relocated_text)
    return relocated, mla, sm120, flashinfer


def execute(path: Path):
    return subprocess.run(
        [sys.executable, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == EXPECTED_SHA256


def main():
    with tempfile.TemporaryDirectory() as temporary_directory:
        relocated, mla, sm120, flashinfer = prepare_case(Path(temporary_directory))
        result = execute(relocated)

        assert result.returncode == 0, result.stderr
        mla_text = mla.read_text()
        sm120_text = sm120.read_text()
        flashinfer_text = flashinfer.read_text()
        assert "VLLM_MLA_NOPE_PAD_ROPE" in mla_text
        assert "self.pe_pad = 64" in mla_text
        assert "torch.nn.functional.pad(q, (0, self.pe_pad))" in mla_text
        assert "q.new_zeros((k_pe.shape[0], 1, self.pe_pad))" in mla_text
        assert "_glm_kpool_tail" in sm120_text
        assert "topk_indices_physical.shape[-1]" in sm120_text
        assert "GLM5_NEXT_WIDTH" in flashinfer_text
        assert "(h, 2176) for h in (8, 16, 32, 64, 128)" in flashinfer_text

        second_result = execute(relocated)
        assert second_result.returncode == 0, second_result.stderr
        assert "already patched" in second_result.stdout

    with tempfile.TemporaryDirectory() as temporary_directory:
        malformed_source = MLA_FIXTURE.replace(
            "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n", ""
        )
        relocated, _, _, _ = prepare_case(
            Path(temporary_directory), malformed_source
        )
        result = execute(relocated)

        assert result.returncode != 0
        assert "anchor1 count 0" in result.stderr

    with tempfile.TemporaryDirectory() as temporary_directory:
        duplicate_anchor = "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n"
        duplicate_source = MLA_FIXTURE.replace(
            duplicate_anchor, duplicate_anchor * 2
        )
        relocated, _, _, _ = prepare_case(
            Path(temporary_directory), duplicate_source
        )
        result = execute(relocated)

        assert result.returncode != 0
        assert "anchor1 count 2" in result.stderr


if __name__ == "__main__":
    main()
