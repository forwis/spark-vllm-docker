# GLM-5.3 Sparse-MLA Reimplementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the active GLM-5.3 DFlash2 implementation and replace it with a reproducible SM121 sparse-MLA vLLM plugin image and conservative two-node recipe.

**Architecture:** Vendor the pinned reference repository's minimal CUDA extension and Python plugin package, compile it inside the official GLM-5.3 CUDA 13 image, and enable it through vLLM's general-plugin interface. Preserve the public build flag, image tag, and recipe filename while replacing every GLM-specific implementation detail and test expectation.

**Tech Stack:** Bash, Dockerfile, Python 3.12/setuptools, PyTorch CUDA extensions, CUDA C++, YAML, mocked shell integration tests

**Spec:** `docs/superpowers/specs/2026-08-31-glm53-sparse-mla-reimplementation-design.md`

## Global Constraints

- Reference source: `/home/arbusto/git/glm53-flash-vllm-gb10` at commit `617d0ccc7a6cd95b5a76b4b7a73f038409150dc1`.
- Base image: `vllm/vllm-openai:glm53-flash-arm64-cu130`.
- Target architecture: SM121 (`GLM53_ARCHS=121a`).
- Runtime contract: TP2, BF16 KV, block size 256, eager execution, MTP with three speculative tokens.
- Do not set `VLLM_GLM53_MOE_INPUT_SCALE`; the pinned source retracts a global constant as unsafe.
- Do not retain or fall back to DFlash2, the old patch chain, FP8 KV, Marlin, or `glm45`.
- Do not run a real Docker build, model download, discovery, image copy, or container launch.
- The existing uncommitted GLM recipe and recipe-test edits are superseded by this plan and may be overwritten; preserve all unrelated user changes.
- Operator-managed host filesystem-cache reclamation remains documentation only.

---

### Task 1: Vendor the sparse-MLA runtime and replace the focused mod test

**Files:**
- Delete: `mods/glm53-dflash2/`
- Delete: `tests/test_glm53_dflash2_mod.sh`
- Create: `mods/glm53-sparse-mla/setup.py`
- Create: `mods/glm53-sparse-mla/csrc/sparse_mla.cu`
- Create: `mods/glm53-sparse-mla/glm53_sparse_mla/__init__.py`
- Create: `mods/glm53-sparse-mla/glm53_sparse_mla/backend.py`
- Create: `mods/glm53-sparse-mla/glm53_sparse_mla/build.py`
- Create: `mods/glm53-sparse-mla/glm53_sparse_mla/moe_fix.py`
- Create: `mods/glm53-sparse-mla/UPSTREAM.md`
- Create: `tests/test_glm53_sparse_mla_mod.sh`

**Interfaces:**
- Consumes: `GLM53_ARCHS=121a` and `MAX_JOBS` during package installation; `VLLM_GLM53_CUDA_SPARSE_MLA=1` at runtime.
- Produces: Python distribution `glm53_sparse_mla==0.1.0`, CUDA op `torch.ops.glm53_sparse_mla.sparse_fwd`, and vLLM general-plugin entry points `glm53_sparse_mla` and `glm53_moe_fix`.

- [ ] **Step 1: Write the failing packaging behavior test**

Replace the old focused test with `tests/test_glm53_sparse_mla_mod.sh`. The test must run `setup.py` under stubbed `setuptools` and `torch.utils.cpp_extension` modules, then assert the captured setup behavior:

```bash
#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOD_DIR="$PROJECT_DIR/mods/glm53-sparse-mla"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

GLM53_ARCHS=121a python3 - "$MOD_DIR" <<'PY'
import os
import runpy
import sys
import types
from pathlib import Path

root = Path(sys.argv[1])
captured = {}

setuptools = types.ModuleType("setuptools")
setuptools.setup = lambda **kwargs: captured.update(kwargs)
cpp_extension = types.ModuleType("torch.utils.cpp_extension")
cpp_extension.BuildExtension = object
cpp_extension.CUDAExtension = lambda **kwargs: kwargs
torch_utils = types.ModuleType("torch.utils")
torch_utils.cpp_extension = cpp_extension
torch = types.ModuleType("torch")
torch.utils = torch_utils
sys.modules.update({
    "setuptools": setuptools,
    "torch": torch,
    "torch.utils": torch_utils,
    "torch.utils.cpp_extension": cpp_extension,
})

os.chdir(root)
runpy.run_path(str(root / "setup.py"), run_name="__main__")

assert captured["name"] == "glm53_sparse_mla"
assert captured["packages"] == ["glm53_sparse_mla"]
assert captured["entry_points"]["vllm.general_plugins"] == [
    "glm53_sparse_mla = glm53_sparse_mla.backend:register",
    "glm53_moe_fix = glm53_sparse_mla.moe_fix:register",
]
nvcc = captured["ext_modules"][0]["extra_compile_args"]["nvcc"]
assert "-gencode=arch=compute_121a,code=sm_121a" in nvcc
assert "-gencode=arch=compute_120,code=compute_120" in nvcc
assert "--use_fast_math" not in nvcc
PY

python3 -m py_compile "$MOD_DIR"/glm53_sparse_mla/*.py
echo "GLM53 sparse-MLA package test passed."
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `bash tests/test_glm53_sparse_mla_mod.sh`

Expected: FAIL because `mods/glm53-sparse-mla/setup.py` does not exist.

- [ ] **Step 3: Remove the old runtime and vendor the pinned production files**

Use `apply_patch` to delete `mods/glm53-dflash2/` and add the six runtime files exactly from the pinned reference checkout. Verify the copy against these independent hashes:

```text
bb79c53cc4828ec79dab08251de7eb5dc5876e5563db719120bc2a4deb52000c  setup.py
3bbf7abde1131a916b85b75dbf65182c1936dd36e6651ed6a492262aab1e2da8  csrc/sparse_mla.cu
ad76a0456d029f2f516b1b13b950c88c5ceb229a1be8040a1f950b0ed31668b6  glm53_sparse_mla/__init__.py
edb6a44ccdef9e740f47b8d8723ae6fbba149b98897cb4fa47842ce5790b35b3  glm53_sparse_mla/backend.py
2af9c8979e054e17d873c02233725f26e681efbb1d0f2238fa32233019aa1ee3  glm53_sparse_mla/build.py
31c626d74bcaa6d166cd51025b27336444ae041d8b85e6986713f50a87ac618f  glm53_sparse_mla/moe_fix.py
```

Add `UPSTREAM.md` with this content:

```markdown
# GLM-5.3 sparse-MLA runtime provenance

The runtime files in this directory are vendored from
`Libertai/glm53-flash-vllm-gb10` at commit
`617d0ccc7a6cd95b5a76b4b7a73f038409150dc1` under Apache-2.0.

The source-to-vendored paths are `kernel/setup.py` to `setup.py`,
`kernel/csrc/sparse_mla.cu` to `csrc/sparse_mla.cu`, and
`kernel/glm53_sparse_mla/` to `glm53_sparse_mla/`. These six runtime files are
copied without modification; their SHA-256 values are recorded in the
implementation plan.

This repository's qualified `--glm53-gb10` profile compiles only for SM121 and
uses the reference backend's executable TP2/BF16 contract. Experimental kernel
revisions, probes, SGLang fixtures, correctness harnesses, deployment scripts,
and RTX provisioning are deliberately omitted because they are not runtime
dependencies of the DGX Spark image.
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
bash tests/test_glm53_sparse_mla_mod.sh
sha256sum \
  mods/glm53-sparse-mla/setup.py \
  mods/glm53-sparse-mla/csrc/sparse_mla.cu \
  mods/glm53-sparse-mla/glm53_sparse_mla/__init__.py \
  mods/glm53-sparse-mla/glm53_sparse_mla/backend.py \
  mods/glm53-sparse-mla/glm53_sparse_mla/build.py \
  mods/glm53-sparse-mla/glm53_sparse_mla/moe_fix.py
```

Expected: focused test PASS and all six hashes equal the pinned values above.

- [ ] **Step 5: Commit the runtime replacement**

```bash
git add mods/glm53-sparse-mla mods/glm53-dflash2 tests/test_glm53_sparse_mla_mod.sh tests/test_glm53_dflash2_mod.sh
git commit -m "feat: vendor GLM53 sparse MLA plugin"
```

---

### Task 2: Replace the dedicated image and build profile

**Files:**
- Delete: `Dockerfile.glm53-dflash2`
- Create: `Dockerfile.glm53-sparse-mla`
- Modify: `build-and-copy.sh`
- Modify: `tests/test_build_and_copy.sh`

**Interfaces:**
- Consumes: `build-and-copy.sh --glm53-gb10`, optional `--build-jobs`, and the qualified profile's existing incompatibility checks.
- Produces: `docker build -f Dockerfile.glm53-sparse-mla -t vllm-node-glm --build-arg BUILD_JOBS=7 --build-arg GLM53_ARCHS=121a .` for the focused fixture invocation.

- [ ] **Step 1: Change the mocked build test to express the new behavior**

Update `setup_fixture` to copy `Dockerfile.glm53-sparse-mla`. Replace the old GLM test with:

```bash
test_glm53_gb10_profile_builds_sparse_mla_plugin_image() {
    setup_fixture
    run_build --glm53-gb10 --build-jobs 7 || fail "--glm53-gb10 run failed"
    assert_log_not_contains '^docker pull '
    assert_log_contains '^docker build -f Dockerfile\.glm53-sparse-mla -t vllm-node-glm .*--build-arg BUILD_JOBS=7 .*--build-arg GLM53_ARCHS=121a \.$'
    assert_log_not_contains '^docker build --target (flashinfer-export|vllm-export) '
    assert_output_contains 'Building qualified GLM-5\.3 sparse-MLA plugin image'
    pass "--glm53-gb10 builds the SM121 sparse-MLA plugin image"
}
```

Rename the invocation at the bottom of the test file to the new function. Keep the qualified-profile override test unchanged.

- [ ] **Step 2: Run the focused build test and verify RED**

Run: `bash tests/test_build_and_copy.sh`

Expected: FAIL because the fixture/build script still refers to `Dockerfile.glm53-dflash2` and does not pass `GLM53_ARCHS=121a`.

- [ ] **Step 3: Add the minimal dedicated Dockerfile**

Create `Dockerfile.glm53-sparse-mla` with:

```dockerfile
FROM vllm/vllm-openai:glm53-flash-arm64-cu130

ARG BUILD_JOBS=16
ARG GLM53_ARCHS=121a
ENV MAX_JOBS=${BUILD_JOBS}

COPY mods/glm53-sparse-mla /opt/spark-vllm/glm53-sparse-mla

RUN GLM53_ARCHS="${GLM53_ARCHS}" \
    pip install --no-build-isolation --no-deps \
        /opt/spark-vllm/glm53-sparse-mla \
    && python3 - <<'PY'
from importlib.metadata import entry_points

plugins = {
    (entry.name, entry.value)
    for entry in entry_points(group="vllm.general_plugins")
}
assert (
    "glm53_sparse_mla",
    "glm53_sparse_mla.backend:register",
) in plugins
assert (
    "glm53_moe_fix",
    "glm53_sparse_mla.moe_fix:register",
) in plugins
PY
```

Delete `Dockerfile.glm53-dflash2`.

- [ ] **Step 4: Replace DFlash2 build-script references**

In `build-and-copy.sh`:

- change help text to `Build the qualified GLM-5.3 GB10 sparse-MLA plugin image`;
- describe the qualified official base and plugin package rather than a patch chain;
- build with `Dockerfile.glm53-sparse-mla`;
- append `--build-arg BUILD_JOBS=$BUILD_JOBS` and `--build-arg GLM53_ARCHS=$DEFAULT_GPU_ARCH_LIST` before `.`;
- emit `Building qualified GLM-5.3 sparse-MLA plugin image with command: ...`;
- keep every override rejection and the `vllm-node-glm` default tag intact.

- [ ] **Step 5: Run focused verification and verify GREEN**

Run:

```bash
bash tests/test_build_and_copy.sh
bash -n build-and-copy.sh
```

Expected: all build-and-copy tests PASS and Bash syntax check exits 0.

- [ ] **Step 6: Commit the image/build replacement**

```bash
git add Dockerfile.glm53-sparse-mla Dockerfile.glm53-dflash2 build-and-copy.sh tests/test_build_and_copy.sh
git commit -m "feat: build GLM53 sparse MLA image"
```

---

### Task 3: Replace the GLM recipe contract

**Files:**
- Modify: `recipes/glm-5.3-flash-nvfp4.yaml`
- Modify: `tests/test_recipes.sh`

**Interfaces:**
- Consumes: two cluster nodes, `vllm-node-glm`, locally cached `LibertAIDAI/GLM-5.3-Flash-NVFP4`, and the installed general plugin.
- Produces: a `vllm serve` command using TP2, BF16 KV, sparse MLA, FlashInfer CUTLASS MoE, eager mode, native MTP3, `glm47`, and `deepseek_r1`.

- [ ] **Step 1: Replace recipe expectations before editing the recipe**

Change `test_glm53_flash_nvfp4_profile` to require these literal command fragments:

```bash
for expected in \
    "LibertAIDAI/GLM-5.3-Flash-NVFP4" \
    "--tensor-parallel-size 2" \
    "--gpu-memory-utilization 0.8" \
    "--max-model-len 65536" \
    "--max-num-seqs 2" \
    "--max-num-batched-tokens 1024" \
    "--block-size 256" \
    "--moe-backend flashinfer_cutlass" \
    "--kv-cache-dtype bfloat16" \
    "--enforce-eager" \
    "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'" \
    "--reasoning-parser deepseek_r1" \
    "--tool-call-parser glm47"; do
```

Require `-e VLLM_GLM53_CUDA_SPARSE_MLA=1` in the launch environment. Add negative assertions for `GLM-5.3-Flash-DFlash2`, `--moe-backend marlin`, `--kv-cache-dtype fp8`, `VLLM_GLM53_MOE_INPUT_SCALE`, and `--reasoning-parser glm45`. Rename log messages from “DFlash2” to “sparse-MLA”.

- [ ] **Step 2: Run the recipe test and verify RED**

Run: `bash tests/test_recipes.sh`

Expected: FAIL because the current recipe still emits the old checkpoint, memory, block-size, MoE, KV, speculation, and reasoning settings.

- [ ] **Step 3: Replace the recipe with the approved runtime contract**

Retain the cluster-only structure and operator cache warning, but set the following exact recipe behavior:

```yaml
recipe_version: "1"
name: GLM-5.3-Flash-NVFP4 sparse MLA (TP=2)
description: Qualified GB10 TP2 profile using the vendored sparse-MLA vLLM plugin

model: LibertAIDAI/GLM-5.3-Flash-NVFP4
container: vllm-node-glm
cluster_only: true

env:
  TZ: Asia/Seoul
  TORCH_CUDA_ARCH_LIST: "12.1a"
  FLASHINFER_CUDA_ARCH_LIST: "12.1a"
  FLASHINFER_DISABLE_VERSION_CHECK: "1"
  VLLM_GLM53_CUDA_SPARSE_MLA: "1"
  NCCL_CUMEM_ENABLE: "0"
  NCCL_NVLS_ENABLE: "0"
  NCCL_CROSS_NIC: "0"
  NCCL_IB_MERGE_NICS: "0"
  TORCH_NCCL_ASYNC_ERROR_HANDLING: "1"
  VLLM_ENGINE_READY_TIMEOUT_S: "3600"

build_args:
  - --glm53-gb10

defaults:
  port: 54351
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.80
  max_num_seqs: 2
  max_num_batched_tokens: 1024
  block_size: 256

command: |
  vllm serve LibertAIDAI/GLM-5.3-Flash-NVFP4 \
    --served-model-name glm-5.3-flash \
    --host {host} \
    --port {port} \
    --trust-remote-code \
    --tensor-parallel-size {tensor_parallel} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-model-len 65536 \
    --max-num-seqs {max_num_seqs} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --block-size {block_size} \
    --moe-backend flashinfer_cutlass \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser deepseek_r1 \
    --kv-cache-dtype bfloat16 \
    --speculative-config '{{"method":"mtp","num_speculative_tokens":3}}'
```

The header comments must reference the local sparse-MLA source repository, instruct users to download only `LibertAIDAI/GLM-5.3-Flash-NVFP4`, and preserve the operator-managed cache-reclamation warning.

- [ ] **Step 4: Run recipe tests and explicit dry-run; verify GREEN**

Run:

```bash
bash tests/test_recipes.sh
./run-recipe.sh recipes/glm-5.3-flash-nvfp4.yaml \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
```

Expected: recipe suite PASS; dry-run exits 0 and prints the exact two-node command without discovery or deployment.

- [ ] **Step 5: Commit the recipe replacement**

```bash
git add recipes/glm-5.3-flash-nvfp4.yaml tests/test_recipes.sh
git commit -m "feat: replace GLM53 recipe with sparse MLA profile"
```

---

### Task 4: Replace active documentation and mark historical plans superseded

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-08-29-qwen-glm53-gb10-build-design.md`
- Modify: `docs/superpowers/plans/2026-08-29-qwen-glm53-gb10-builds.md`

**Interfaces:**
- Consumes: the final build and recipe behavior from Tasks 1–3.
- Produces: public and agent guidance that names only the active sparse-MLA profile, plus explicit supersession markers on historical mixed Qwen/GLM records.

- [ ] **Step 1: Replace the README GLM changelog section**

Document the official base, pinned LibertAI sparse-MLA plugin source, BF16 KV/TP2 constraints, MTP3, and the single-model setup workflow:

```bash
./build-and-copy.sh --glm53-gb10 -c --copy-parallel
./hf-download.sh LibertAIDAI/GLM-5.3-Flash-NVFP4 -c --copy-parallel
./run-recipe.sh recipes/glm-5.3-flash-nvfp4.yaml
```

Remove instructions to download `RedHatAI/GLM-5.3-Flash-NVFP4` or
`incoai/GLM-5.3-Flash-DFlash2` and remove claims about DFlash2, FP8 KV, and the
old patch chain.

- [ ] **Step 2: Update repository agent constraints**

Replace the GLM bullet in `AGENTS.md` with the new qualified profile:

```markdown
- `--glm53-gb10` locally builds `vllm-node-glm` from the official GLM-5.3 vLLM
  base plus the vendored sparse-MLA vLLM plugin pinned from
  `Libertai/glm53-flash-vllm-gb10`. Its base image, SM121 build target, TP2/BF16
  kernel contract, checkpoint, and serving flags are one qualified profile; do
  not mix it with manual vLLM, FlashInfer, Torch, GPU-architecture, KV-dtype, or
  experimental-profile overrides.
```

Keep the cluster-only/cache-reclamation paragraph intact.

- [ ] **Step 3: Mark mixed historical documents superseded**

Immediately below each older document title, add:

```markdown
> **GLM-5.3 note:** The GLM portions of this historical document are superseded
> by `docs/superpowers/specs/2026-08-31-glm53-sparse-mla-reimplementation-design.md`.
> Its Qwen portions remain historical records of the Qwen implementation.
```

- [ ] **Step 4: Check documentation consistency**

Run:

```bash
rg -n 'RedHatAI/GLM-5.3|incoai/GLM-5.3|glm53-dflash2|Dockerfile.glm53-dflash2' \
  README.md AGENTS.md build-and-copy.sh recipes tests mods Dockerfile.glm53-sparse-mla
```

Expected: no matches. DFlash2 mentions for unrelated Qwen recipes are allowed outside these GLM-specific patterns.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md AGENTS.md \
  docs/superpowers/specs/2026-08-29-qwen-glm53-gb10-build-design.md \
  docs/superpowers/plans/2026-08-29-qwen-glm53-gb10-builds.md
git commit -m "docs: document GLM53 sparse MLA profile"
```

---

### Task 5: Full development verification and cleanup audit

**Files:**
- Verify only; modify the smallest responsible file if a check exposes a defect.

**Interfaces:**
- Consumes: completed Tasks 1–4.
- Produces: fresh evidence that the mocked build, vendored package, recipe, launcher integration, syntax, and repository state agree.

- [ ] **Step 1: Run all required automated checks**

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

Expected: every command exits 0. No command contacts a real cluster, builds an image, or downloads a model.

- [ ] **Step 2: Audit removal and scope**

```bash
test ! -e Dockerfile.glm53-dflash2
test ! -e mods/glm53-dflash2
test ! -e tests/test_glm53_dflash2_mod.sh
rg -n 'glm53-dflash2|Dockerfile\.glm53-dflash2|RedHatAI/GLM-5\.3|incoai/GLM-5\.3' \
  build-and-copy.sh recipes tests mods README.md AGENTS.md
git diff --check
git status --short
```

Expected: the three absence checks pass; the search returns no active GLM implementation references; the diff check is clean; status contains only intended changes or pre-existing unrelated user work.

- [ ] **Step 3: Review the final diff against the specification**

Verify explicitly that:

- the old Dockerfile/mod/test tree is deleted;
- all six vendored runtime hashes match the pinned reference;
- the Docker build uses the official base and `GLM53_ARCHS=121a`;
- the recipe uses TP2/BF16 and never sets a global MoE input scale;
- no privileged cache command was added;
- no operational build, copy, download, or launch was performed.

- [ ] **Step 4: Commit any verification-only corrections**

If verification required a correction, stage only the responsible files and commit with a narrowly scoped message. If no corrections were required, do not create an empty commit.
