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

# TileLang's fused MHC kernels compile on CUDA by default, but its SM12x
# lowering is not qualified for GB10.  GLM's first text prefill otherwise
# JIT-compiles mhc_fused_tilelang and terminates the worker.  vLLM already
# provides a native MHC fallback when TileLang is unavailable; select it only
# on SM12x without changing the qualified stack on other architectures.
python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py")
s = p.read_text()
old = '''    if current_platform.is_cuda():
        return True
'''
new = '''    if current_platform.is_cuda():
        try:
            major, _ = torch.cuda.get_device_capability()
        except Exception:
            return False
        # TileLang MHC is unvalidated on SM12x (GB10); use vLLM's native
        # correctness fallback there until this lowering is qualified.
        return major not in (12,)
'''
if s.count(old) != 1:
    raise SystemExit("unexpected TileLang MHC CUDA gate; refusing to patch")
p.write_text(s.replace(old, new))
print("TileLang MHC disabled on SM12x")
PY

# FlashInfer's planner must receive the logical FP8 dtype, not vLLM's uint8
# storage dtype for the E4M3 KV pages. The forward path already reinterprets
# the pages as E4M3 before calling FlashInfer; this makes the plan path match.
python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py")
s = p.read_text()
old = """        topk_indices_buffer = impl.topk_indices_buffer
        assert topk_indices_buffer is not None
        self.state = _SM90State(
            device,
            impl.num_heads,
            kv_cache_spec.dtype,
            vllm_config.scheduler_config.max_num_batched_tokens,
"""
new = """        topk_indices_buffer = impl.topk_indices_buffer
        assert topk_indices_buffer is not None
        # vLLM stores E4M3 KV pages as raw bytes. FlashInfer's planner needs
        # the logical element dtype instead of that uint8 storage dtype.
        kv_dtype = (
            torch.float8_e4m3fn
            if kv_cache_spec.dtype == torch.uint8
            else kv_cache_spec.dtype
        )
        self.state = _SM90State(
            device,
            impl.num_heads,
            kv_dtype,
            vllm_config.scheduler_config.max_num_batched_tokens,
"""
if s.count(old) != 1:
    raise SystemExit("unexpected SM90 MLA planner dtype handoff; refusing to patch")
p.write_text(s.replace(old, new))
print("SM90 MLA FP8 KV planner dtype normalized")
PY

# Sparse MLA on GB10 cannot safely execute a long prefill alongside an MTP
# decode step.  The CUDA failure otherwise surfaces later as a cuBLAS error.
# Defer peer prefills until decoding completes; the policy can be disabled or
# capped with GLM53_MIXED_PREFILL_CHUNK for explicit experiments.
python3 - <<'PY'
import os
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
s = p.read_text()
mark = "# [glm53-decode-floor]"
if mark not in s:
    old_import = "import itertools\nimport time\n"
    new_import = "import itertools\nimport os\nimport time\n"
    helper = '''def _glm53_mixed_prefill_policy(running, current):
    """Return a mixed-prefill cap when another request is decoding."""
    raw = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip").strip().lower()
    if raw in ("0", "off", "no"):
        return None
    if raw in ("skip", "-1"):
        cap = 0
    else:
        try:
            cap = int(raw)
        except ValueError:
            cap = 0
        if cap <= 0:
            return None
    current_id = getattr(current, "request_id", None)
    for running_request in running:
        if running_request is current or getattr(running_request, "request_id", None) == current_id:
            continue
        if running_request.num_computed_tokens >= running_request.num_prompt_tokens:
            return cap
    return None


'''
    running_old = '''            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
'''
    running_new = '''            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
'''
    waiting_old = '''                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
'''
    waiting_new = '''                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
                    if mixed_cap is not None and num_computed_tokens < request.num_prompt_tokens:
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
'''
    anchors = ((old_import, new_import, "scheduler import"),
               ("from vllm.compilation.cuda_graph import CUDAGraphStat\n", helper + "from vllm.compilation.cuda_graph import CUDAGraphStat\n", "scheduler helper"),
               (running_old, running_new, "running prefill"),
               (waiting_old, waiting_new, "waiting prefill"))
    for old, new, label in anchors:
        if s.count(old) != 1:
            raise SystemExit(f"unexpected {label} anchor; refusing to patch")
        s = s.replace(old, new, 1)
    p.write_text(s)
print("GLM mixed-prefill decode floor applied")
PY

# vLLM's post-KV-cache warmup constructs a synthetic mixed MTP/non-MTP
# decode batch.  GLM's KDA path reclassifies its non-MTP member as prefill;
# on GB10 this is the same unsafe sparse-MLA shape that can kill the worker.
# The real scheduler has the corresponding peer-prefill guard above.  Keep
# the all-MTP and single-request warmups, but omit this synthetic mixed case.
python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/warmup.py")
s = p.read_text()
old = '''        if num_reqs >= 2:
            # Mixed spec / non-spec: GDN and KDA reclassify the non-spec decode
            # as a prefill and split the batch into spec/non-spec token indices.
            decode_steps.append(([0, 1], [use_spec_decode, False]))
            if use_spec_decode:
                # Exercise the model paths that split a batch by whether each
                # request received draft tokens.
                decode_steps.append(([0, 1], [False, False]))
'''
new = '''        model_type = getattr(model_runner.model_config.hf_config, "model_type", "")
        is_glm5_mtp = model_type == "glm5next"
        if num_reqs >= 2 and not (is_glm5_mtp and use_spec_decode):
            # Mixed spec / non-spec: GDN and KDA reclassify the non-spec decode
            # as a prefill and split the batch into spec/non-spec token indices.
            decode_steps.append(([0, 1], [use_spec_decode, False]))
            if use_spec_decode:
                # Exercise the model paths that split a batch by whether each
                # request received draft tokens.
                decode_steps.append(([0, 1], [False, False]))
'''
if s.count(old) != 1:
    raise SystemExit("unexpected mixed MTP warmup anchor; refusing to patch")
p.write_text(s.replace(old, new))
print("GLM MTP mixed warmup excluded on GB10")
PY

python3 "$SCRIPT_DIR/patch_v7.py"
python3 "$SCRIPT_DIR/patch_v8_fp8.py"
