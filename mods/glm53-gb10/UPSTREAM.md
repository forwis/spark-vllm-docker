# GLM-5.3 GB10 patch provenance

These runtime patches are vendored from
[`tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
at commit `2815bcb63cd28daa3c52501da0c62dad0927e99b`.

| Source path | Vendored location / installed target |
| --- | --- |
| `docker/Dockerfile.glm53-sm121` | `apply.sh`: exact-match NoPE MLA patches for `/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py` and `/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` |
| `docker/Dockerfile.glm53-sm121-v6` | `apply.sh`: exact-match PDL patch for `/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py` |
| `docker/patch_v7.py` | `patch_v7.py`: `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py` and `/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/ops/kpool_compress.py` |
| `docker/patch_v8_fp8.py` | `patch_v8_fp8.py`: `/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/mla.cuh` and `/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py` |

`apply.sh` requires `flashinfer.__version__` to start with
`0.6.18.dev20260819`, then applies the patches in source-stage order: NoPE MLA,
PDL, indexer hardening, and FP8 MLA. Every patch has an exact single-anchor
check and exits nonzero if the installed source differs.

The upstream repository's `overlay-dflash2/` content is deliberately omitted.
DFlash2 is a separate speculative-decoding integration and is outside this
GB10 runtime patch set.
