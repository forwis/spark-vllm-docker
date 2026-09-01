# GLM-5.3 Native FP8 DS-MLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the qualified GLM-5.3 GB10 image's custom BF16 sparse-MLA plugin with the cluster reference's native `fp8_ds_mla` NoPE padding chain while retaining the LibertAIDAI checkpoint and stable public interfaces.

**Architecture:** Vendor the exact environment-gated `patch_mla.py` from `glm53-flash-cluster@50e344433076efce702029e3f446c54a80916dc9`, apply it to a digest-pinned official GLM image, and let native vLLM/FlashInfer own sparse MLA. Remove the custom CUDA/plugin implementation, keep the existing local-snapshot processor fix, and switch the conservative TP2 recipe from BF16 KV to explicit `fp8_ds_mla` without adopting the lab checkpoint or 512K profile.

**Tech Stack:** Bash, YAML, Python 3.10+, Dockerfile, vLLM, FlashInfer, Hugging Face cache layout, mocked shell integration tests.

**Spec:** `docs/superpowers/specs/2026-09-01-glm53-native-fp8-ds-mla-design.md`

## Global Constraints

- Preserve `--glm53-gb10`, image tag `vllm-node-glm`, recipe path `recipes/glm-5.3-flash-nvfp4.yaml`, model `LibertAIDAI/GLM-5.3-Flash-NVFP4`, served name `glm-5.3-flash`, TP2, and port `54351`.
- Pin the base image to `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`.
- Vendor `docker/labbuild/patch_mla.py` from `/home/arbusto/git/glm53-flash-cluster` commit `50e344433076efce702029e3f446c54a80916dc9`; its required SHA-256 is `12ed6565328c8b72edae62207b2640436355475d033729046170cdf5db96954f`.
- Retain context 65,536, GPU utilization 0.80, two sequences, 1,024 batched tokens, block size 256, eager execution, MTP3, `glm47`, `deepseek_r1`, and multimodal processing.
- Use `VLLM_MLA_NOPE_PAD_ROPE=1` and exact KV dtype `fp8_ds_mla`; do not add `--kv-cache-memory-bytes` or `--language-model-only`.
- Do not import `model.patch`, `modelopt.patch`, either CC12 indexer resource patch, the lab checkpoint, 512K settings, Compose/launch/watchdog code, or host-management behavior.
- Do not add privileged filesystem-cache clearing. Cache reclamation remains an operator-managed pre-launch requirement.
- Preserve unrelated work. In particular, the existing unstaged local-snapshot changes in `recipes/glm-5.3-flash-nvfp4.yaml` and `tests/test_recipes.sh` are intentional and must be committed before the FP8 recipe edits.
- Development verification must not build an image, download a model, or launch a real container/cluster.

---

### Task 1: Capture the Existing Local-Snapshot Startup Fix

**Files:**
- Modify (already present and unstaged): `recipes/glm-5.3-flash-nvfp4.yaml:46`
- Modify (already present and unstaged): `tests/test_recipes.sh:1419-1512`

**Interfaces:**
- Consumes: Hugging Face cache ref `/root/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/refs/main`.
- Produces: shell variable `glm53_model_snapshot` and a `vllm serve` model argument pointing to the corresponding local snapshot directory.

This fix already completed a red-green cycle before this plan was written: the recipe test failed when the command passed the repository ID and passed after local snapshot resolution was restored. Do not rewrite or discard the pending diff.

- [ ] **Step 1: Confirm the pending diff is limited to the startup fix**

Run:

```bash
git diff -- recipes/glm-5.3-flash-nvfp4.yaml tests/test_recipes.sh
```

Expected: the recipe reads `refs/main` and passes `/root/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/$glm53_model_snapshot`; the test requires that local path and rejects a direct repository-ID `vllm serve` argument.

- [ ] **Step 2: Re-run the focused observable behavior**

Run:

```bash
./tests/test_recipes.sh -v
./run-recipe.sh recipes/glm-5.3-flash-nvfp4.yaml \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
```

Expected: 69 recipe tests pass; the dry-run launch script contains:

```bash
glm53_model_snapshot="$(cat /root/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/refs/main)"
vllm serve "/root/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/$glm53_model_snapshot"
```

- [ ] **Step 3: Commit only the startup fix**

```bash
git add recipes/glm-5.3-flash-nvfp4.yaml tests/test_recipes.sh
git diff --cached --check
git commit -m "fix: serve GLM53 from local snapshot"
```

---

### Task 2: Vendor and Test the Native FP8 NoPE Patch Chain

**Files:**
- Create: `mods/glm53-fp8-ds-mla/patch_mla.py`
- Create: `mods/glm53-fp8-ds-mla/UPSTREAM.md`
- Create: `tests/test_glm53_fp8_ds_mla_patch.py`

**Interfaces:**
- Consumes: the installed files `vllm/model_executor/layers/mla.py`, `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`, and `flashinfer/mla/_sparse_mla_sm120.py` from the pinned image.
- Produces: an environment-gated 64-wide NoPE pad, corrected SM120 top-k width handling, the GLM dispatch geometry, and k-pool tail compaction.

- [ ] **Step 1: Write the failing executable patch test**

Create `tests/test_glm53_fp8_ds_mla_patch.py`. It must execute the real vendored script against temporary syntactically valid fixtures, rather than grep the patch source. Use this structure:

```python
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
```

Use `prepare_case` to create a temporary case, execute it once, then read the
three returned paths. The successful test must assert:

```python
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
```

Run the same relocated patch a second time against the already transformed
fixtures and assert exit status zero plus `already patched` in stdout. The
source script treats the NoPE marker as the whole-chain idempotence marker and
exits immediately, so it does not print the later section messages on the
second invocation.

Add a second case that removes `self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` from `MLA_FIXTURE`, runs the relocated patch, and asserts nonzero status and `anchor1 count 0` in stderr. At module startup, assert the vendored file checksum equals `EXPECTED_SHA256`.

- [ ] **Step 2: Run the new test to verify RED**

Run:

```bash
python3 tests/test_glm53_fp8_ds_mla_patch.py
```

Expected: FAIL because `mods/glm53-fp8-ds-mla/patch_mla.py` does not exist.

- [ ] **Step 3: Vendor the exact patch and provenance**

Use `apply_patch` to add a byte-for-byte copy of:

```text
/home/arbusto/git/glm53-flash-cluster/docker/labbuild/patch_mla.py
```

at `mods/glm53-fp8-ds-mla/patch_mla.py`. Do not reformat or adapt it.

Create `mods/glm53-fp8-ds-mla/UPSTREAM.md` with these exact facts:

```markdown
# GLM-5.3 native FP8 DS-MLA patch provenance

`patch_mla.py` is vendored without modification from
`/home/arbusto/git/glm53-flash-cluster/docker/labbuild/patch_mla.py` at tree
commit `50e344433076efce702029e3f446c54a80916dc9`. It was introduced by commit
`ac425d3ae51e6e1472a9000f4e242e63b6510987` and is distributed under the
source repository's Apache-2.0 license.

SHA-256:
`12ed6565328c8b72edae62207b2640436355475d033729046170cdf5db96954f`.

Only this generic NoPE/native-FP8 patch is imported. The lab checkpoint,
`model.patch`, `modelopt.patch`, the CC12 512K indexer resource patches, and
the reference repository's orchestration are deliberately excluded.
```

- [ ] **Step 4: Verify provenance and GREEN behavior**

Run:

```bash
sha256sum mods/glm53-fp8-ds-mla/patch_mla.py
python3 tests/test_glm53_fp8_ds_mla_patch.py
python3 -m py_compile mods/glm53-fp8-ds-mla/patch_mla.py \
  tests/test_glm53_fp8_ds_mla_patch.py
```

Expected: checksum matches the pinned value; both successful/idempotent and missing-anchor cases pass; Python compilation succeeds.

- [ ] **Step 5: Commit the vendored patch unit**

```bash
git add mods/glm53-fp8-ds-mla tests/test_glm53_fp8_ds_mla_patch.py
git diff --cached --check
git commit -m "feat: vendor GLM53 native FP8 MLA patch"
```

---

### Task 3: Replace the Custom Plugin Image with the Native FP8 Image

**Files:**
- Create: `Dockerfile.glm53-fp8-ds-mla`
- Modify: `build-and-copy.sh:630,792-819,1161-1177`
- Modify: `tests/test_build_and_copy.sh:43-56,286-310,1352-1353`
- Delete: `Dockerfile.glm53-sparse-mla`
- Delete: `mods/glm53-sparse-mla/`
- Delete: `tests/test_glm53_sparse_mla_mod.sh`

**Interfaces:**
- Consumes: `mods/glm53-fp8-ds-mla/patch_mla.py` and `build-and-copy.sh --glm53-gb10`.
- Produces: digest-pinned, patched image `vllm-node-glm` with no custom CUDA extension or vLLM general-plugin entry points.

- [ ] **Step 1: Change the mocked build test first**

Rename `test_glm53_gb10_profile_builds_sparse_mla_plugin_image` to `test_glm53_gb10_profile_builds_native_fp8_mla_image`. Update `setup_fixture` to copy `Dockerfile.glm53-fp8-ds-mla` instead of the old Dockerfile. Require:

```bash
assert_log_contains '^docker build -f Dockerfile\.glm53-fp8-ds-mla -t vllm-node-glm \.$'
assert_output_contains 'Building qualified GLM-5\.3 native FP8 DS-MLA image'
assert_log_not_contains 'BUILD_JOBS='
assert_log_not_contains 'GLM53_ARCHS='
```

Keep the existing assertions that no prebuilt image, FlashInfer export, or vLLM export is invoked. Rename the final `pass` message to `--glm53-gb10 builds the native FP8 DS-MLA image` and update the test invocation at the bottom of the file.

- [ ] **Step 2: Run the build test to verify RED**

Run:

```bash
./tests/test_build_and_copy.sh
```

Expected: FAIL because `build-and-copy.sh` still invokes `Dockerfile.glm53-sparse-mla` with plugin compiler arguments.

- [ ] **Step 3: Create the native-FP8 Dockerfile**

Create `Dockerfile.glm53-fp8-ds-mla` with:

```dockerfile
FROM vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce

COPY mods/glm53-fp8-ds-mla/patch_mla.py /tmp/glm53-patch-mla.py

RUN set -eux; \
    python3 /tmp/glm53-patch-mla.py; \
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm; \
    FLASHINFER_ROOT=/usr/local/lib/python3.12/dist-packages/flashinfer; \
    grep -Fq 'VLLM_MLA_NOPE_PAD_ROPE' "$VLLM_ROOT/model_executor/layers/mla.py"; \
    grep -Fq '_glm_kpool_tail' "$VLLM_ROOT/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"; \
    grep -Fq 'topk_indices_physical.shape[-1]' "$VLLM_ROOT/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"; \
    grep -Fq 'GLM5_NEXT_WIDTH' "$FLASHINFER_ROOT/mla/_sparse_mla_sm120.py"; \
    python3 -m py_compile \
        "$VLLM_ROOT/model_executor/layers/mla.py" \
        "$VLLM_ROOT/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py" \
        "$FLASHINFER_ROOT/mla/_sparse_mla_sm120.py"; \
    rm -f /tmp/glm53-patch-mla.py
```

- [ ] **Step 4: Switch the build profile and remove obsolete implementation**

In `build-and-copy.sh`:

- change help text to `Build the qualified GLM-5.3 GB10 native FP8 DS-MLA image`;
- describe the qualified profile as the digest-pinned official base plus native FP8 NoPE patch chain;
- build with `docker build -f Dockerfile.glm53-fp8-ds-mla -t "$IMAGE_TAG" .`;
- preserve `--progress=plain` and `--network` insertion before the final `.`;
- stop forwarding `BUILD_JOBS` and `GLM53_ARCHS` for this profile;
- log `Building qualified GLM-5.3 native FP8 DS-MLA image with command: ...`;
- retain every current qualified override rejection.

Delete the old Dockerfile, `mods/glm53-sparse-mla/`, and `tests/test_glm53_sparse_mla_mod.sh`. Do not retain the MoE plugin separately: it is inactive in the qualified recipe, and the checkpoint's calibrated scale data remains authoritative.

- [ ] **Step 5: Verify the migrated image contract**

Run:

```bash
./tests/test_build_and_copy.sh
python3 tests/test_glm53_fp8_ds_mla_patch.py
bash -n build-and-copy.sh
rg -n 'glm53_sparse_mla|VLLM_GLM53_CUDA_SPARSE_MLA|Dockerfile\.glm53-sparse-mla' \
  Dockerfile.glm53-fp8-ds-mla build-and-copy.sh mods tests/test_build_and_copy.sh
```

Expected: build and patch tests pass; syntax passes; the final `rg` returns no matches.

- [ ] **Step 6: Commit the image migration**

```bash
git add build-and-copy.sh tests/test_build_and_copy.sh \
  Dockerfile.glm53-fp8-ds-mla Dockerfile.glm53-sparse-mla \
  mods/glm53-sparse-mla tests/test_glm53_sparse_mla_mod.sh
git diff --cached --check
git commit -m "feat: build GLM53 with native FP8 MLA"
```

---

### Task 4: Switch the Qualified Recipe to Native `fp8_ds_mla`

**Files:**
- Modify: `recipes/glm-5.3-flash-nvfp4.yaml:1-66`
- Modify: `tests/test_recipes.sh:1417-1512`

**Interfaces:**
- Consumes: `vllm-node-glm` from Task 3 and a complete local LibertAIDAI Hugging Face snapshot from Task 1.
- Produces: TP2 native-FP8 launch command with `VLLM_MLA_NOPE_PAD_ROPE=1` and `--kv-cache-dtype fp8_ds_mla`.

- [ ] **Step 1: Replace recipe expectations before production YAML**

In `test_glm53_flash_nvfp4_profile`:

- rename display/pass/fail text from `sparse-MLA` to `native FP8 DS-MLA`;
- preserve Task 1's local-snapshot behavior, updating its assertions for the
  `glm53_model_cache`, `glm53_model_ref`, and `glm53_model_path` variables used
  by the new fail-fast preflight;
- replace required `--kv-cache-dtype bfloat16` with `--kv-cache-dtype fp8_ds_mla`;
- replace required launch environment `-e VLLM_GLM53_CUDA_SPARSE_MLA=1` with `-e VLLM_MLA_NOPE_PAD_ROPE=1`;
- require generated-script checks for a readable `refs/main` and a present
  `processor_config.json`, with explicit error text naming the missing item;
- require the unchanged TP2, 0.80 utilization, 65,536 context, two sequences, 1,024 batched tokens, block 256, FlashInfer CUTLASS MoE, eager, MTP3, `deepseek_r1`, and `glm47` arguments;
- reject `VLLM_GLM53_CUDA_SPARSE_MLA`, `--kv-cache-dtype bfloat16`, `--kv-cache-memory-bytes`, `--language-model-only`, lab checkpoint names, and old DFlash artifacts;
- retain the assertions for image `vllm-node-glm`, build arg `--glm53-gb10`, and cluster-only mode.

Use exact fixed-string checks so rejecting `--kv-cache-dtype fp8` does not accidentally reject the required `fp8_ds_mla` value.

- [ ] **Step 2: Run recipe tests to verify RED**

Run:

```bash
./tests/test_recipes.sh -v
```

Expected: one failure in the GLM native-FP8 profile because the YAML still emits BF16 KV and the old plugin gate.

- [ ] **Step 3: Implement the minimal recipe change**

In `recipes/glm-5.3-flash-nvfp4.yaml`:

- identify `/home/arbusto/git/glm53-flash-cluster` as the native-FP8 patch source in the header;
- retain the download and operator-managed cache-reclamation instructions;
- set the name to `GLM-5.3-Flash-NVFP4 native FP8 DS-MLA (TP=2)`;
- set the description to `Qualified GB10 TP2 profile using native FP8 DS-MLA with the vendored NoPE patch chain`;
- replace `VLLM_GLM53_CUDA_SPARSE_MLA: "1"` with `VLLM_MLA_NOPE_PAD_ROPE: "1"`;
- retain local snapshot resolution and a local-path `vllm serve` argument;
- expand snapshot resolution into this fail-fast preflight before `vllm serve`:

```bash
glm53_model_cache=/root/.cache/huggingface/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4
glm53_model_ref="$glm53_model_cache/refs/main"
if [ ! -r "$glm53_model_ref" ]; then
  echo "Error: GLM-5.3 snapshot ref is missing: $glm53_model_ref" >&2
  exit 1
fi
glm53_model_snapshot="$(cat "$glm53_model_ref")"
glm53_model_path="$glm53_model_cache/snapshots/$glm53_model_snapshot"
if [ ! -f "$glm53_model_path/processor_config.json" ]; then
  echo "Error: GLM-5.3 snapshot is incomplete: missing processor_config.json in $glm53_model_path" >&2
  exit 1
fi
vllm serve "$glm53_model_path" \
```

- replace `--kv-cache-dtype bfloat16` with `--kv-cache-dtype fp8_ds_mla`;
- do not change any other serving limit or add a KV memory pin.

- [ ] **Step 4: Verify GREEN and generated command**

Run:

```bash
./tests/test_recipes.sh -v
./run-recipe.sh recipes/glm-5.3-flash-nvfp4.yaml \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
```

Expected: 69 tests pass. The dry-run includes the local snapshot path, `-e VLLM_MLA_NOPE_PAD_ROPE=1`, and `--kv-cache-dtype fp8_ds_mla`; it contains no old plugin gate, BF16 KV flag, KV-memory pin, or text-only flag.

- [ ] **Step 5: Commit the recipe contract**

```bash
git add recipes/glm-5.3-flash-nvfp4.yaml tests/test_recipes.sh
git diff --cached --check
git commit -m "feat: serve GLM53 with native FP8 MLA"
```

---

### Task 5: Update Active Documentation and Supersession Notes

**Files:**
- Modify: `README.md:184-199`
- Modify: `AGENTS.md:37-48`
- Modify: `docs/superpowers/specs/2026-08-31-glm53-sparse-mla-reimplementation-design.md:1-6`
- Modify: `docs/superpowers/plans/2026-08-31-glm53-sparse-mla-reimplementation.md:1-7`

**Interfaces:**
- Consumes: the final image and recipe contracts from Tasks 3 and 4.
- Produces: public/operator/agent documentation that describes only the active native-FP8 profile while retaining historical records with explicit supersession.

- [ ] **Step 1: Update public and agent documentation**

Change the README section title to `GLM-5.3 Flash native FP8 DS-MLA profile`. State that the image uses the digest-pinned official GLM base and the vendored NoPE/native-FP8 patch chain from `glm53-flash-cluster`, retains LibertAIDAI, targets SM121 TP2, and uses `fp8_ds_mla`. Keep the three build/download/run commands unchanged.

Replace the GLM paragraph in `AGENTS.md` with this contract:

```markdown
- `--glm53-gb10` locally builds `vllm-node-glm` from the digest-pinned official
  GLM-5.3 vLLM base plus the vendored NoPE/native-`fp8_ds_mla` patch chain from
  `glm53-flash-cluster`. Its base digest, retained LibertAIDAI checkpoint,
  native FP8 cache, block size 256, SM121 TP2 topology, and serving flags are a
  single qualified profile; do not mix it with manual vLLM, FlashInfer, Torch,
  GPU-architecture, KV-dtype, or experimental-profile overrides.
```

Preserve the separate cache-reclamation rule exactly.

- [ ] **Step 2: Mark the previous design and plan as historical**

Immediately after each old document title, add:

```markdown
> **Superseded:** The active GLM implementation is defined by
> `docs/superpowers/specs/2026-09-01-glm53-native-fp8-ds-mla-design.md` and its
> implementation plan. This document remains as history for the removed
> BF16 custom-plugin profile.
```

- [ ] **Step 3: Verify documentation consistency**

Run:

```bash
rg -n 'sparse-MLA vLLM plugin|TP2/BF16|VLLM_GLM53_CUDA_SPARSE_MLA|glm53-sparse-mla' \
  README.md AGENTS.md recipes/glm-5.3-flash-nvfp4.yaml build-and-copy.sh \
  Dockerfile.glm53-fp8-ds-mla mods/glm53-fp8-ds-mla tests
```

Expected: no matches. Historical design and plan files are intentionally excluded from this active-surface scan.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md AGENTS.md \
  docs/superpowers/specs/2026-08-31-glm53-sparse-mla-reimplementation-design.md \
  docs/superpowers/plans/2026-08-31-glm53-sparse-mla-reimplementation.md
git diff --cached --check
git commit -m "docs: document GLM53 native FP8 profile"
```

---

### Task 6: Complete Development Verification

**Files:**
- Verify only; modify a file only to correct a failure caused by Tasks 1-5, then rerun the affected red-green cycle and commit that correction separately.

**Interfaces:**
- Consumes: the complete native-FP8 implementation.
- Produces: fresh evidence that the source, generated command, and mocked orchestration contracts pass without performing an operational deployment.

- [ ] **Step 1: Run focused patch and image tests**

```bash
python3 tests/test_glm53_fp8_ds_mla_patch.py
./tests/test_build_and_copy.sh
python3 -m py_compile mods/glm53-fp8-ds-mla/patch_mla.py \
  tests/test_glm53_fp8_ds_mla_patch.py
bash -n build-and-copy.sh
```

Expected: all commands exit zero.

- [ ] **Step 2: Run recipe and launcher suites**

```bash
./tests/test_recipes.sh -v
./tests/test_launch_cluster_image_sync.sh
./tests/test_launch_cluster_vllm_pr.sh
```

Expected: 69 recipe tests, 3 image-consistency tests, and 6 runtime-vLLM-PR tests pass.

- [ ] **Step 3: Inspect the final dry-run and active surface**

```bash
./run-recipe.sh recipes/glm-5.3-flash-nvfp4.yaml \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
rg -n 'glm53_sparse_mla|VLLM_GLM53_CUDA_SPARSE_MLA|Dockerfile\.glm53-sparse-mla|--kv-cache-dtype bfloat16' \
  build-and-copy.sh recipes/glm-5.3-flash-nvfp4.yaml \
  Dockerfile.glm53-fp8-ds-mla mods/glm53-fp8-ds-mla tests README.md AGENTS.md
```

Expected: dry-run shows local snapshot resolution, native pad gate, `fp8_ds_mla`, and all conservative limits. The `rg` command returns no matches.

- [ ] **Step 4: Review repository state**

```bash
git diff --check
git status --short --branch
git log --oneline -8
```

Expected: no unstaged or staged implementation changes remain; the branch contains the snapshot, vendor, image, recipe, and documentation commits after the design and plan commits. Do not build/copy the image or launch the cluster as part of development verification.
