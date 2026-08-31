# GLM-5.3 Sparse-MLA Reimplementation Design

## Goal

Replace every active part of the existing GLM-5.3 Flash DFlash2/SM121 patch
chain with a clean, reproducible integration of the sparse-MLA CUDA extension
from `/home/arbusto/git/glm53-flash-vllm-gb10` at commit
`617d0ccc7a6cd95b5a76b4b7a73f038409150dc1`.

The replacement remains a two-node DGX Spark profile exposed as
`--glm53-gb10`, built as `vllm-node-glm`, and launched by
`recipes/glm-5.3-flash-nvfp4.yaml`.

## Scope

The implementation will:

- Delete `Dockerfile.glm53-dflash2` and `mods/glm53-dflash2/`.
- Remove DFlash2-specific image construction, recipe behavior, tests, and
  current public documentation.
- Vendor the minimal production sparse-MLA extension and its two vLLM plugin
  entry points from the pinned reference commit.
- Build the extension inside the official GLM-5.3 CUDA 13 image for SM121.
- Replace the GLM recipe with the reference source's executable TP2/BF16
  contract and conservative GB10 serving defaults.
- Keep host cache reclamation as an operator-owned prerequisite, without adding
  privileged cache-clearing behavior.

The work is development-only. It will not build an image, download weights,
discover hosts, launch containers, or modify cluster resources.

## Source of Truth

The pinned reference repository's executable source is authoritative when its
README or example deployment files disagree with it. At the pinned commit:

- `glm53_sparse_mla.backend` accepts 32 attention heads per rank, which is TP2
  for the 64-head GLM-5.3 model.
- The backend advertises `auto` and `bfloat16` KV cache and explicitly rejects
  quantized KV cache.
- The CUDA kernel consumes BF16 query and KV tensors.
- The latest corrective commit states that a constant
  `VLLM_GLM53_MOE_INPUT_SCALE` is not a sound replacement for per-projection
  calibration scales.

Therefore the new profile will not copy the reference README's broader TP/FP8
claims, will not set `VLLM_GLM53_MOE_INPUT_SCALE`, and will not retain any old
DFlash2 compatibility code.

## Vendored Runtime

The new `mods/glm53-sparse-mla/` tree will contain only runtime build inputs:

- `setup.py`
- `csrc/sparse_mla.cu`
- `glm53_sparse_mla/__init__.py`
- `glm53_sparse_mla/backend.py`
- `glm53_sparse_mla/build.py`
- `glm53_sparse_mla/moe_fix.py`
- `UPSTREAM.md`

`UPSTREAM.md` will record the source repository, exact commit, license, copied
paths, intended target architecture, and deliberate omissions. Experimental
kernel revisions, probes, deployment scripts, SGLang fixtures, correctness
harnesses, and RTX-specific provisioning are not runtime dependencies and will
not be vendored.

The vendored code will remain byte-for-byte equivalent to the pinned runtime
files except for narrowly documented packaging changes required by this
repository. Any such changes must be covered by focused tests and listed in
`UPSTREAM.md`.

## Image Build

`Dockerfile.glm53-sparse-mla` will start from
`vllm/vllm-openai:glm53-flash-arm64-cu130`. It will copy the vendored extension
to an image-local build directory and install it with:

```bash
GLM53_ARCHS=121a MAX_JOBS="${BUILD_JOBS}" \
  pip install --no-build-isolation --no-deps <vendored directory>
```

Building inside the serving image ensures the extension matches that image's
Python, PyTorch, CUDA, and libtorch ABI. The build must not use fast math. The
final image will verify that the package metadata exposes both
`vllm.general_plugins` entry points:

- `glm53_sparse_mla = glm53_sparse_mla.backend:register`
- `glm53_moe_fix = glm53_sparse_mla.moe_fix:register`

No installed vLLM source file will be patched. The package is inert unless its
environment gates are enabled.

`build-and-copy.sh --glm53-gb10` will retain the current qualified-profile
override rejection and image tag, but it will invoke the new Dockerfile and
describe the sparse-MLA plugin build rather than DFlash2. It will forward
`BUILD_JOBS` and the fixed `GLM53_ARCHS=121a` target.

## Recipe

`recipes/glm-5.3-flash-nvfp4.yaml` will remain cluster-only and use:

- Model: `LibertAIDAI/GLM-5.3-Flash-NVFP4`
- Image: `vllm-node-glm`
- Tensor parallelism: 2
- GPU memory utilization: 0.80
- Maximum model length: 65,536
- Maximum sequences: 2
- Maximum batched tokens: 1,024
- Block size: 256
- KV cache dtype: `bfloat16`
- MoE backend: `flashinfer_cutlass`
- Eager execution
- MTP with three speculative tokens
- Tool parser: `glm47`
- Reasoning parser: `deepseek_r1`

The recipe will set `VLLM_GLM53_CUDA_SPARSE_MLA=1` so vLLM selects the custom
NoPE sparse-MLA backend. It will not set `VLLM_GLM53_MOE_INPUT_SCALE`; calibrated
checkpoint data, rather than a global constant, must supply any required input
scales. Existing RoCE/NCCL safety settings and the 3,600-second engine readiness
timeout remain because they are cluster orchestration constraints independent
of the discarded DFlash2 implementation.

The recipe will use the normal model identifier rather than reading a specific
Hugging Face snapshot path in shell. The runner already mounts the Hugging Face
cache and vLLM resolves the downloaded revision.

## Tests

Implementation will follow red-green-refactor cycles.

The build-script test will first be changed to require the new Dockerfile,
SM121 build argument, image tag, and sparse-MLA build description. It must fail
against the old DFlash2 implementation before production code changes.

A focused vendored-runtime test will validate observable packaging behavior:

- building package metadata exposes both expected vLLM plugin entry points;
- `GLM53_ARCHS=121a` produces the expected SM121 compiler target without
  `--use_fast_math`;
- the old DFlash2 artifacts are absent from active build inputs.

The recipe test will first require the new model, BF16 KV, conservative memory
limits, sparse-MLA environment gate, MTP configuration, and `deepseek_r1`
parser. It will reject the old DFlash2 model, DFlash speculative method, Marlin
backend, FP8 KV, unsafe scale override, and `glm45` parser.

Final development verification will run:

```bash
./tests/test_build_and_copy.sh
./tests/test_glm53_sparse_mla_mod.sh
./tests/test_recipes.sh -v
./tests/test_launch_cluster_image_sync.sh
./tests/test_launch_cluster_vllm_pr.sh
bash -n build-and-copy.sh
python3 -m py_compile mods/glm53-sparse-mla/glm53_sparse_mla/*.py
./run-recipe.sh recipes/glm-5.3-flash-nvfp4.yaml \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
```

These checks do not prove GPU kernel correctness or ABI compatibility. A real
image build and live two-node inference validation remain explicitly outside
this development request.

## Documentation and Historical Records

The README and repository instructions will describe only the new sparse-MLA
profile. Active documentation will not direct users to the old DFlash2 drafter,
RedHatAI checkpoint, FP8 KV path, Marlin backend, or `glm45` parser.

Existing dated Superpowers design and plan documents are historical records
that also cover unrelated Qwen work. They will not be treated as active
implementation, but their GLM sections will receive a clear superseded notice
pointing to this specification so repository-wide searches do not mistake them
for current guidance.

## Safety and Failure Behavior

- Qualified-profile overrides remain rejected so a user cannot silently replace
  the base image, CUDA architecture, vLLM, FlashInfer, or Torch combination.
- A failed extension build, missing plugin entry point, unsupported KV dtype, or
  unsupported head count fails explicitly; there is no fallback to the old
  patch chain.
- No credentials, local IP addresses, `.env` values, model artifacts, or built
  binaries will be added to the repository.
- The implementation preserves unrelated Qwen and general orchestration
  behavior.
