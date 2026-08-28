#!/bin/bash
# Runtime patches for GLM-5.3-Flash on GB10 (SM121).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 -c '
import flashinfer
expected = "0.6.18.dev20260819"
if not flashinfer.__version__.startswith(expected):
    raise SystemExit(
        f"GLM53 GB10 patches require FlashInfer {expected}*, found "
        f"{flashinfer.__version__}"
    )
'

# NoPE MLA: make the SM90 backend available on SM121 and use FA2 there.
python3 - <<'PY'
from pathlib import Path

base = Path("/usr/local/lib/python3.12/dist-packages/vllm")

p = base / "platforms/cuda.py"
s = p.read_text()
old = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
new = """        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]"""
if s.count(old) != 1:
    raise SystemExit("unexpected cuda.py capability-12 MLA candidate list; refusing to patch")
p.write_text(s.replace(old, new))

p = base / "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"
s = p.read_text()

old = "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major == 9\n"
new = "    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:\n        return capability.major in (9, 12)\n"
if s.count(old) != 1:
    raise SystemExit("unexpected sm90 capability gate; refusing to patch")
s = s.replace(old, new)

old = '            backend="fa3",\n'
new = '            backend=("fa3" if torch.cuda.get_device_capability()[0] == 9 else "fa2"),\n'
if s.count(old) != 1:
    raise SystemExit("unexpected sm90 wrapper backend literal; refusing to patch")
s = s.replace(old, new)

old = """        if not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 requires FlashInfer with SM90 "
                "MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
new = """        if kv_cache_dtype in ("fp8", "fp8_e4m3") and not has_flashinfer_sm90_nope_mla():
            return (
                "FLASHINFER_MLA_SPARSE_SM90 fp8 KV requires FlashInfer with "
                "SM90 MLA support (ckv_scale_arr in "
                "BatchMLAPagedAttentionWrapper.run, FlashInfer >= 0.6.18)"
            )"""
if s.count(old) != 1:
    raise SystemExit("unexpected sm90 flashinfer version gate; refusing to patch")
s = s.replace(old, new)

p.write_text(s)
print("sm121 NoPE MLA patches applied")
PY

# PDL is not validated on SM12x and races in KDA recurrent-state kernels.
python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py")
s = p.read_text()
old = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        return major >= 9
"""
new = """    @classmethod
    def is_arch_support_pdl(cls) -> bool:
        try:
            device = torch.cuda.current_device()
            major, _ = torch.cuda.get_device_capability(device)
        except Exception:
            return False
        # PDL lowering is unvalidated on SM12x (GB10) and races on KDA
        # state kernels there; keep it to Hopper/Blackwell-datacenter.
        return major in (9, 10)
"""
if s.count(old) != 1:
    raise SystemExit("unexpected is_arch_support_pdl source; refusing to patch")
p.write_text(s.replace(old, new))
print("PDL gated off on SM12x")
PY

python3 "$SCRIPT_DIR/patch_v7.py"
python3 "$SCRIPT_DIR/patch_v8_fp8.py"
