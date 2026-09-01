# DeepSeek V4 Flash Vision-Exp on DSpark v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a locally built, two-DGX-Spark recipe for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` by porting focused multimodal support onto the pinned `vllm-node-dsv4f:v2` JASL/DSpark runtime.

**Architecture:** Keep the native JASL v2 SM121, DSpark, and `fp8_ds_mla` implementation unchanged, then apply an idempotent launch-time Python mod that installs three Apache-2.0 Vision modules and patches six exact JASL source sites. A cluster-only recipe reproduces the v2 source-build pins when needed and launches the larger 1M-context DSpark-6 profile; no published final runtime image is used.

**Tech Stack:** Bash, Python 3.10+, PyYAML recipes, vLLM Python overlays, Docker source builds, unittest, mocked recipe dry-runs.

**Spec:** `docs/superpowers/specs/2026-09-02-dsv4f-vision-exp-design.md`

## Global Constraints

- Use only the locally built image tag `vllm-node-dsv4f:v2`; do not introduce a published final image.
- Pin vLLM to `https://github.com/jasl/vllm.git` commit `9ad62027bc84ca0ccbcc40853179312de770220c`.
- Pin FlashInfer to commit `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7` and GPU architecture to `12.1a`.
- Preserve Torch 2.13.0, torchvision 0.28.0, torchaudio 2.11.0, CUTLASS DSL 4.7, Transformers 5, and the non-B12X v2 profile.
- Port only the production changes from `a939984672/vllm` commits `f9ff02d`, `b706973`, `979e41f`, `ce9dd9c`, and `7996ea0`.
- Preserve JASL's `_sync_fused_moe_metadata()`, `WINDOW_SIZE`, negative-token validation, native DSpark, and native SM121 sparse-MLA behavior.
- Use `fp8_ds_mla`; do not add or select the misleading `nvfp4_ds_mla` compatibility alias.
- Keep the recipe cluster-only with TP2, 1,048,576 context, six sequences, 8,192 batch tokens, block size 256, and DSpark-6.
- Do not add runner-owned distributed executor, node-count, node-rank, rendezvous, or headless flags to the recipe command.
- Development validation is CPU/static only: do not build an image, download model weights, launch containers, or operate the cluster.
- Preserve the pre-existing user changes in the primary checkout; work only in `.worktrees/dsv4f-vision-exp`.

## File Structure

### New runtime mod

- `mods/dsv4f-vision-exp/run.sh` — locate the installed vLLM package and invoke the transactional patcher.
- `mods/dsv4f-vision-exp/patch_dsv4f_vision.py` — validate, transform, compile, and atomically install all existing-file changes and overlay modules.
- `mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/mm_preprocess.py` — exact multimodal processor from `a939984672/vllm@7996ea0`.
- `mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision.py` — exact Vision tower and aligner from `a939984672/vllm@7996ea0`.
- `mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision_model.py` — exact multimodal wrapper from `a939984672/vllm@7996ea0`.
- `mods/dsv4f-vision-exp/README.md` — compatibility, behavior, application, and rollback guidance.
- `mods/dsv4f-vision-exp/UPSTREAM.md` — source repositories, commits, file paths, and Apache-2.0 provenance.

### Tests and recipes

- `tests/test_dsv4f_vision_exp_mod.py` — synthetic JASL-v2 fixture tests for all patch sites, overlay installation, drift rejection, compilation, and idempotence.
- `recipes/deepseek-v4-flash-vision-exp.yaml` — qualified two-node Vision-Exp profile and exact local source-build arguments.
- `recipes/deepseek-v4-flash-0731-jasl.yaml` — update only the stale image tag from v1 to v2.
- `tests/test_recipes.sh` — focused dry-run assertions for the new Vision recipe and v2 tag correction.
- `README.md` — public changelog and launch example.

---

### Task 1: Transactional Vision Runtime Mod

**Files:**

- Create: `tests/test_dsv4f_vision_exp_mod.py`
- Create: `mods/dsv4f-vision-exp/patch_dsv4f_vision.py`
- Create: `mods/dsv4f-vision-exp/run.sh`
- Create: `mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/mm_preprocess.py`
- Create: `mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision.py`
- Create: `mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision_model.py`
- Create: `mods/dsv4f-vision-exp/UPSTREAM.md`

**Interfaces:**

- Consumes: an installed vLLM package tree matching JASL commit `9ad62027bc84ca0ccbcc40853179312de770220c`, selected with `--vllm-root PATH` or `VLLM_PACKAGE_ROOT`.
- Produces: `PatchError`, `patch_tree(vllm_root: Path, overlay_root: Path, *, check: bool = False) -> list[Path]`, and a no-argument `run.sh` mod entry point.
- Produces: registered architecture `DeepseekV4VForConditionalGeneration` and DSpark loading support for checkpoint weights ending in `.ffn.gate.bias_vl`.

- [ ] **Step 1: Write the failing patcher tests**

Create `tests/test_dsv4f_vision_exp_mod.py` with `unittest`. Import the future
patcher by file path, build a temporary `vllm/` tree containing the exact
unpatched JASL-v2 anchor extracts, and use small syntactically valid modules so
the patcher's compile checks remain meaningful.

The test class must cover these observable postconditions:

```python
class Dsv4fVisionExpModTests(unittest.TestCase):
    def test_patch_tree_installs_complete_port_idempotently(self):
        root, overlay = self.make_fixture()
        changed = PATCHER.patch_tree(root, overlay)
        self.assertEqual(
            {path.relative_to(root).as_posix() for path in changed},
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
        self.assertIn(".ffn.gate.e_score_correction_bias_vl", self.read("models/deepseek_v4/nvidia/dspark.py"))

        before = self.hash_tree(root)
        self.assertEqual(PATCHER.patch_tree(root, overlay), [])
        self.assertEqual(self.hash_tree(root), before)

    def test_check_mode_validates_without_writing(self):
        root, overlay = self.make_fixture()
        before = self.hash_tree(root)
        expected = PATCHER.patch_tree(root, overlay, check=True)
        self.assertEqual(len(expected), 9)
        self.assertEqual(self.hash_tree(root), before)

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
```

The fixtures must include both the ordinary `.ffn.gate.bias` loader block and
JASL's `_sync_fused_moe_metadata()` call so the test detects either upstream
implementation being lost.

- [ ] **Step 2: Run the patcher test to verify it fails**

Run:

```bash
python3 tests/test_dsv4f_vision_exp_mod.py
```

Expected: FAIL because `mods/dsv4f-vision-exp/patch_dsv4f_vision.py` does not
exist.

- [ ] **Step 3: Vendor the three exact Apache-2.0 Vision modules**

Read the files at the final selected community tree and add their contents
unchanged under the mod overlay:

```bash
git -C /tmp/a939-vllm-dsv4-vision show \
  7996ea0:vllm/models/deepseek_v4/mm_preprocess.py
git -C /tmp/a939-vllm-dsv4-vision show \
  7996ea0:vllm/models/deepseek_v4/vision.py
git -C /tmp/a939-vllm-dsv4-vision show \
  7996ea0:vllm/models/deepseek_v4/vision_model.py
```

Use `apply_patch` to create the three overlay files. Retain each file's
`SPDX-License-Identifier: Apache-2.0` header. Do not copy upstream docs, tests,
registry files, or the rest of its v0.28 tree.

Create `UPSTREAM.md` with this exact provenance table:

```markdown
| Content | Repository | Commit/tree | License |
| --- | --- | --- | --- |
| Vision modules and focused runtime behavior | `https://github.com/a939984672/vllm` | `7996ea0` (commits `f9ff02d`, `b706973`, `979e41f`, `ce9dd9c`, `7996ea0`) | Apache-2.0 |
| Target source layout | `https://github.com/jasl/vllm` | `9ad62027bc84ca0ccbcc40853179312de770220c` | Apache-2.0 |
| Larger runtime profile | `/home/arbusto/git/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` | `d97c808ec1c71b496badee6805dfd4818a8455d7` | MIT orchestration; Apache-2.0 vLLM-derived overlays |
```

- [ ] **Step 4: Implement exact, idempotent text transforms**

Create `patch_dsv4f_vision.py` with a marker per target file, exact anchor-count
checks, and one transformation function per responsibility:

```python
class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise PatchError(f"expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


PATCHERS: dict[str, Callable[[str], str]] = {
    "model_executor/layers/fused_moe/router/fused_topk_bias_router.py": patch_router,
    "model_executor/models/registry.py": patch_registry,
    "models/deepseek_v4/common/ops/cache_utils.py": patch_cache_utils,
    "models/deepseek_v4/nvidia/model.py": patch_model,
    "models/deepseek_v4/nvidia/dspark.py": patch_dspark,
    "v1/engine/input_processor.py": patch_input_processor,
}
```

Implement the transformations from the five selected community commits, with
these JASL-specific resolutions:

```python
# model.py: keep this before attaching the Vision router state.
self._sync_fused_moe_metadata()
if self.gate.bias_vl is not None:
    router = getattr(self.experts, "router", None)
    if router is not None:
        router.bias_vl = self.gate.bias_vl
        router.vl_vocab_size = self.vl_vocab_size
```

```python
# input_processor.py: preserve negative-id rejection, then widen only Vision.
allowed_max = max(tokenizer.max_token_id, model_vocab_size - 1)
hf_cfg = getattr(model_config, "hf_config", None)
if int(getattr(hf_cfg, "vision_n_layers", 0) or 0) > 0:
    allowed_max = max(allowed_max, model_vocab_size + 4)
if max_input_id > allowed_max:
    raise VLLMValidationError(f"Token id {max_input_id} is out of vocabulary")
```

```python
# dspark.py: load both text and modality-specific gate biases.
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
```

Append `compute_vision_visible_window` after JASL's existing cache utility
kernel without renaming or replacing `WINDOW_SIZE`. Port the complete
`_compute_routing_vision` method and its `_compute_routing` dispatch from
commit `979e41f`, and port the complete registry/model/hash-bias changes from
the selected commit range.

Every patch function must compile its result with
`compile(result, relative_path, "exec")` before returning it and must verify
its marker plus required semantic postconditions when called on an
already-patched file.

- [ ] **Step 5: Implement transactional overlay installation**

Implement `patch_tree` so it computes and validates every result before any
write, rejects a differing pre-existing overlay file, and writes changed files
through a sibling temporary file followed by `Path.replace()`:

```python
def patch_tree(
    vllm_root: Path,
    overlay_root: Path,
    *,
    check: bool = False,
) -> list[Path]:
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
```

The CLI must accept `--vllm-root`, `--overlay-root`, and `--check`. It must
print the relative paths that are compatible, changed, or already installed,
and return nonzero on `PatchError` or `SyntaxError`.

- [ ] **Step 6: Add the launch-time mod entry point**

Create executable `run.sh` using the repository's package-discovery pattern:

```bash
#!/bin/bash
set -euo pipefail

PREFIX="[dsv4f-vision-exp]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${VLLM_PACKAGE_ROOT:-}" ]]; then
    VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM is not installed for the active Python interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
    )
fi

python3 "$MOD_DIR/patch_dsv4f_vision.py" \
    --vllm-root "$VLLM_PACKAGE_ROOT" \
    --overlay-root "$MOD_DIR/overlay/vllm"
echo "$PREFIX Vision-Exp architecture and DSpark routing support are installed."
```

Do not import vLLM from the mod; package discovery must not initialize CUDA
while the launcher is preparing idle containers.

- [ ] **Step 7: Run the focused test and syntax checks**

Run:

```bash
python3 tests/test_dsv4f_vision_exp_mod.py
bash -n mods/dsv4f-vision-exp/run.sh
python3 -m py_compile mods/dsv4f-vision-exp/patch_dsv4f_vision.py \
  mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/mm_preprocess.py \
  mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision.py \
  mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision_model.py
```

Expected: all tests and compile checks pass.

- [ ] **Step 8: Commit the runtime mod**

```bash
git add mods/dsv4f-vision-exp tests/test_dsv4f_vision_exp_mod.py
git commit -m "feat: port DSv4 Vision support to DSpark v2"
```

---

### Task 2: Qualified Vision Recipe and v2 Tag Correction

**Files:**

- Create: `recipes/deepseek-v4-flash-vision-exp.yaml`
- Modify: `recipes/deepseek-v4-flash-0731-jasl.yaml:7`
- Modify: `tests/test_recipes.sh:1390-1495`
- Modify: `tests/test_recipes.sh:1820-1885`

**Interfaces:**

- Consumes: `mods/dsv4f-vision-exp/run.sh` from Task 1 and existing recipe fields consumed by `run-recipe.py`.
- Produces: recipe name `deepseek-v4-flash-vision-exp`, image tag `vllm-node-dsv4f:v2`, and a generated two-node vLLM command containing the qualified larger profile.

- [ ] **Step 1: Add the failing focused recipe assertions**

Add `test_dsv4f_vision_exp_profile()` beside the existing qualified Qwen and
GLM profile tests, and call it from `main()` immediately after those tests.
Generate a two-node dry-run with an empty config:

```bash
output=$("$PROJECT_DIR/run-recipe.py" deepseek-v4-flash-vision-exp \
    --config /dev/null --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
status=$?
vllm_cmd=$(extract_vllm_command "$output")
launch_cmd=$(extract_launch_cmd "$output")
```

Assert all of these exact strings:

```bash
required_vllm_args=(
    "vllm serve deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    "--served-model-name deepseek-v4-flash-vision-exp"
    "--tensor-parallel-size 2"
    "--gpu-memory-utilization 0.835"
    "--kv-cache-dtype fp8_ds_mla"
    "--block-size 256"
    "--max-model-len 1048576"
    "--max-num-seqs 6"
    "--max-num-batched-tokens 8192"
    "--long-prefill-token-threshold 1024"
    "--enable-prefix-caching"
    "--enable-chunked-prefill"
    "--async-scheduling"
    "--limit-mm-per-prompt '{\"image\":8}'"
    "--hf-overrides '{\"architectures\":[\"DeepseekV4VForConditionalGeneration\"],\"is_mm_prefix_lm\":true}'"
    "--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":6,\"draft_sample_method\":\"probabilistic\"}'"
    "--tokenizer-mode deepseek_v4"
    "--tool-call-parser deepseek_v4"
    "--reasoning-parser deepseek_v4"
    "--generation-config vllm"
)
```

Also assert:

```bash
grep -qF -- "-t vllm-node-dsv4f:v2" <<< "$launch_cmd"
grep -qF -- "--apply-mod mods/dsv4f-vision-exp" <<< "$launch_cmd"
grep -qF -- "Cluster only: Yes" <<< "$output"
grep -qF -- "Build args: --vllm-repo https://github.com/jasl/vllm.git" <<< "$output"
grep -qF -- "--vllm-ref 9ad62027bc84ca0ccbcc40853179312de770220c" <<< "$output"
grep -qF -- "--flashinfer-ref a0a6b019b9b27d49d209f85d028a1ae5a9b347d7" <<< "$output"
```

Reject `nvfp4_ds_mla`, `--exp-b12x`, published Anemll/MiaAI image names, and
the runner-owned distributed flags from the vLLM command. Read
`recipes/deepseek-v4-flash-0731-jasl.yaml` and assert its container is v2.

- [ ] **Step 2: Run the recipe suite to verify the new test fails**

Run:

```bash
./tests/test_recipes.sh -v
```

Expected: FAIL in `test_dsv4f_vision_exp_profile` because the new recipe does
not exist and the existing JASL recipe still names v1.

- [ ] **Step 3: Add the qualified recipe**

Create `recipes/deepseek-v4-flash-vision-exp.yaml` with the following complete
profile. Double literal braces because `run-recipe.py` performs Python format
substitution on the command:

```yaml
recipe_version: "1"
name: DeepSeek-V4-Flash-Vision-Exp
description: Locally built JASL v2 Vision-Exp runtime for a dual DGX Spark cluster

model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
container: vllm-node-dsv4f:v2

build_args:
  - --vllm-repo
  - https://github.com/jasl/vllm.git
  - --vllm-ref
  - 9ad62027bc84ca0ccbcc40853179312de770220c
  - --rebuild-flashinfer
  - --flashinfer-ref
  - a0a6b019b9b27d49d209f85d028a1ae5a9b347d7
  - --gpu-arch
  - 12.1a
  - --torch-version
  - 2.13.0
  - --torchvision-version
  - 0.28.0
  - --torchaudio-version
  - 2.11.0

cluster_only: true

mods:
  - mods/dsv4f-vision-exp

defaults:
  port: 54351
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.835
  max_model_len: 1048576
  block_size: 256
  max_num_seqs: 6
  max_num_batched_tokens: 8192
  long_prefill_token_threshold: 1024

env:
  TZ: Asia/Seoul
  VLLM_USE_BREAKABLE_CUDAGRAPH: "0"

command: |
  vllm serve deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
      --served-model-name deepseek-v4-flash-vision-exp \
      --host {host} \
      --port {port} \
      --trust-remote-code \
      --tensor-parallel-size {tensor_parallel} \
      --gpu-memory-utilization {gpu_memory_utilization} \
      --kv-cache-dtype fp8_ds_mla \
      --block-size {block_size} \
      --max-model-len {max_model_len} \
      --max-num-seqs {max_num_seqs} \
      --max-num-batched-tokens {max_num_batched_tokens} \
      --long-prefill-token-threshold {long_prefill_token_threshold} \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --async-scheduling \
      --limit-mm-per-prompt '{{"image":8}}' \
      --hf-overrides '{{"architectures":["DeepseekV4VForConditionalGeneration"],"is_mm_prefix_lm":true}}' \
      --speculative-config '{{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}}' \
      --tokenizer-mode deepseek_v4 \
      --enable-auto-tool-choice \
      --tool-call-parser deepseek_v4 \
      --reasoning-parser deepseek_v4 \
      --generation-config vllm
```

Do not add `--moe-backend flashinfer_b12x`: v2's build metadata explicitly
records B12X disabled.

- [ ] **Step 4: Correct the stale 0731 JASL tag**

Change only this line in `recipes/deepseek-v4-flash-0731-jasl.yaml`:

```diff
-container: vllm-node-dsv4f:v1
+container: vllm-node-dsv4f:v2
```

- [ ] **Step 5: Run recipe validation and explicit dry-runs**

Run:

```bash
./tests/test_recipes.sh -v
./run-recipe.sh deepseek-v4-flash-vision-exp \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
./run-recipe.sh deepseek-v4-flash-0731-jasl \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
```

Expected: the suite passes; both dry-runs select `vllm-node-dsv4f:v2`; the
Vision dry-run prints the pinned local source-build arguments and applies only
`mods/dsv4f-vision-exp`.

- [ ] **Step 6: Commit the recipes and focused assertions**

```bash
git add recipes/deepseek-v4-flash-vision-exp.yaml \
  recipes/deepseek-v4-flash-0731-jasl.yaml tests/test_recipes.sh
git commit -m "feat: add DSv4 Vision-Exp DSpark recipe"
```

---

### Task 3: Public Documentation and Complete Development Verification

**Files:**

- Create: `mods/dsv4f-vision-exp/README.md`
- Modify: `README.md:169`

**Interfaces:**

- Consumes: the mod behavior from Task 1 and recipe name/profile from Task 2.
- Produces: operator-facing setup, qualification boundary, limitations, and rollback instructions.

- [ ] **Step 1: Write the mod documentation**

Create `mods/dsv4f-vision-exp/README.md` documenting:

````markdown
# DeepSeek V4 Flash Vision-Exp on JASL DSpark v2

This mod ports only the DeepSeek V4 Vision Python runtime delta onto the locally
built `vllm-node-dsv4f:v2` image. It requires JASL vLLM commit
`9ad62027bc84ca0ccbcc40853179312de770220c`; unknown layouts fail before vLLM
starts.

The mod registers `DeepseekV4VForConditionalGeneration`, installs the Vision
tower and processor, enables image sentinels and `bias_vl` routing, and maps
the Vision gate bias into DSpark draft layers. It does not replace CUDA kernels
or install a different vLLM tree.

Use the qualified recipe:

```bash
./run-recipe.sh deepseek-v4-flash-vision-exp --setup
```

The recipe is two-node only. Development tests do not prove GPU correctness;
acceptance requires text, image, and DSpark smoke tests on the Spark pair.
Remove the recipe's `mods/dsv4f-vision-exp` entry to roll back the runtime
patch, and use the unmodified local v2 image for text-only serving.
````

Link to `UPSTREAM.md` for the complete provenance and license record. Explain
that `fp8_ds_mla` is deliberate because the reference profile's padded
`nvfp4_ds_mla` name routes the same FP8 layout and supplies no 4-bit capacity
gain.

- [ ] **Step 2: Add the public changelog entry**

Insert a `2026-09-02` entry immediately after `## CHANGELOG` in `README.md`:

````markdown
### 2026-09-02

#### DeepSeek V4 Flash Vision-Exp on locally built DSpark v2

Added the two-node `deepseek-v4-flash-vision-exp` recipe. It reproduces the
JASL v2 SM121 source build, applies the focused Vision runtime port on both
containers, and serves the larger 1M-context DSpark-6 profile with native
`fp8_ds_mla` KV cache. The recipe does not use a published final runtime image.

```bash
./run-recipe.sh deepseek-v4-flash-vision-exp --setup
```

Development validation is CPU-only; perform the first real build and launch
under `docs/AGENT_RUNBOOK.md`.
````

- [ ] **Step 3: Run documentation and diff checks**

Run:

```bash
rg -n "deepseek-v4-flash-vision-exp|vllm-node-dsv4f:v2|fp8_ds_mla" \
  README.md mods/dsv4f-vision-exp/README.md \
  recipes/deepseek-v4-flash-vision-exp.yaml
git diff --check
```

Expected: each public name and the canonical cache dtype agree; `git diff
--check` exits zero.

- [ ] **Step 4: Commit the public documentation**

```bash
git add README.md mods/dsv4f-vision-exp/README.md
git commit -m "docs: document DSv4 Vision-Exp profile"
```

- [ ] **Step 5: Run the complete development verification matrix**

Run all commands from the feature worktree root:

```bash
python3 --version
python3 -c 'import yaml; print(yaml.__version__)'
command -v bc

python3 tests/test_dsv4f_vision_exp_mod.py
bash -n mods/dsv4f-vision-exp/run.sh
python3 -m py_compile mods/dsv4f-vision-exp/patch_dsv4f_vision.py \
  mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/mm_preprocess.py \
  mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision.py \
  mods/dsv4f-vision-exp/overlay/vllm/models/deepseek_v4/vision_model.py

./tests/test_recipes.sh -v
./tests/test_launch_cluster_image_sync.sh
./tests/test_launch_cluster_vllm_pr.sh

./run-recipe.sh deepseek-v4-flash-vision-exp \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2
./run-recipe.sh deepseek-v4-flash-0731-jasl \
  --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2

git diff --check
git status --short --branch
```

Expected: every test and syntax check exits zero; both dry-runs select v2; the
final status contains only the intended committed feature history and no
uncommitted files.

- [ ] **Step 6: Inspect the final committed delta**

Run:

```bash
git log --oneline --decorate main..HEAD
git diff --stat main...HEAD
git diff --name-status main...HEAD
```

Expected commits, in addition to the design and plan commits:

```text
feat: port DSv4 Vision support to DSpark v2
feat: add DSv4 Vision-Exp DSpark recipe
docs: document DSv4 Vision-Exp profile
```

Confirm the primary checkout still retains its unrelated `AGENTS.md` and
`recipes/glm-5.3-flash-nvfp4.yaml` changes and that no operational state was
modified.
