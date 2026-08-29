# Qwen and GLM-5.3 GB10 Builds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Qwen and GLM-5.3 GB10 image profiles and recipes, then build and copy both images.

**Architecture:** The existing build wrapper owns profile selection and Docker build arguments. The Dockerfile conditionally installs the GLM-qualified FlashInfer nightly and applies vendored exact-match patches after wheel installation; recipe YAMLs select the resulting image tags and safe TP2 serve configuration.

**Tech Stack:** Bash, Docker BuildKit, Python patch scripts, YAML, mocked shell tests.

**Spec:** `docs/superpowers/specs/2026-08-29-qwen-glm53-gb10-build-design.md`

## Global Constraints

- Preserve unrelated work and operate directly in the user-approved clean `main` checkout.
- Pin vendored GLM source to `2815bcb63cd28daa3c52501da0c62dad0927e99b`.
- The GLM profile uses vLLM `06569a8696076eeae9558928b00f035ded8f8b60`, PR `53906`, and FlashInfer `0.6.18.dev20260819` from the FlashInfer nightly index.
- Qwen uses the same vLLM ref, PR `53896`, FlashInfer `v0.6.17`, and tag `vllm-node-qwen`.
- GLM uses tag `vllm-node-glm`; it does not install a DFlash2 draft model or automate privileged host cache eviction.
- Run only dry-runs during development; real builds and copies occur after source verification and use the user-supplied host `192.168.200.13`.

---

### Task 1: Vendor the GLM-5.3 GB10 patch set

**Files:**
- Create: `mods/glm53-gb10/apply.sh`
- Create: `mods/glm53-gb10/patch_v7.py`
- Create: `mods/glm53-gb10/patch_v8_fp8.py`
- Create: `mods/glm53-gb10/UPSTREAM.md`
- Test: `tests/test_build_and_copy.sh`

**Interfaces:**
- Consumes: `GLM53_GB10=1` in the runner Docker build.
- Produces: executable `/opt/spark-vllm/mods/glm53-gb10/apply.sh` that exits nonzero when a patch anchor does not match exactly.

- [ ] **Step 1: Write the failing mocked-build test**

Add `test_glm53_gb10_profile_forwards_runner_patch_mode` in `tests/test_build_and_copy.sh`. Run `run_build --glm53-gb10` and assert the mocked runner command includes `--build-arg GLM53_GB10=1` and tag `vllm-node-glm`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `bash tests/test_build_and_copy.sh`

Expected: FAIL because `--glm53-gb10` is not recognized or no runner build argument exists.

- [ ] **Step 3: Vendor the patch scripts and provenance**

Copy the NoPE-MLA/PDL exact-match patches from the pinned source Dockerfiles and the indexer/FP8 scripts into `mods/glm53-gb10/`. `apply.sh` runs every script in a deterministic order and verifies `flashinfer.__version__` begins with `0.6.18.dev20260819`. `UPSTREAM.md` records the source URL, commit, source paths, installed target paths, and the deliberate omission of DFlash2.

- [ ] **Step 4: Verify the vendored scripts are syntactically valid**

Run: `bash -n mods/glm53-gb10/apply.sh && python3 -m py_compile mods/glm53-gb10/patch_v7.py mods/glm53-gb10/patch_v8_fp8.py`

Expected: exit 0.

- [ ] **Step 5: Commit the vendor set**

```bash
git add mods/glm53-gb10 tests/test_build_and_copy.sh
git commit -m "feat: vendor GLM53 GB10 runtime patches"
```

### Task 2: Add the GLM-5.3 GB10 build profile

**Files:**
- Modify: `build-and-copy.sh`
- Modify: `Dockerfile`
- Modify: `tests/test_build_and_copy.sh`

**Interfaces:**
- Consumes: `--glm53-gb10` CLI option.
- Produces: `docker build -t vllm-node-glm ... --build-arg GLM53_GB10=1` and a runner image containing the qualified GLM patch set.

- [ ] **Step 1: Extend the failing test with exact profile inputs**

Assert that `run_build --glm53-gb10` builds the vLLM export with `VLLM_REF=06569a8696076eeae9558928b00f035ded8f8b60`, `VLLM_PRS=53906`, and builds the runner with `GLM53_GB10=1`. Assert it does not pull a prebuilt runner.

- [ ] **Step 2: Run the focused test and verify it fails for the missing flags**

Run: `bash tests/test_build_and_copy.sh`

Expected: the new profile assertions fail while the pre-existing tests continue to run.

- [ ] **Step 3: Implement profile selection and Docker integration**

Add `--glm53-gb10` parsing and validation in `build-and-copy.sh`. Set the GLM image tag by default, force vLLM and FlashInfer source/runner build paths, set the vLLM ref and PR, and pass `GLM53_GB10=1` to the runner build. In `Dockerfile`, copy the vendored directory, install `flashinfer-python` and `flashinfer-cubin` version `0.6.18.dev20260819` from `https://flashinfer.ai/whl/nightly/`, remove `flashinfer-jit-cache`, restore `nvidia-nccl-cu13==2.30.7` and `nvidia-cutlass-dsl==4.6.2`, then invoke `apply.sh` only when the argument is enabled.

- [ ] **Step 4: Run mocked profile and Dockerfile checks**

Run: `bash tests/test_build_and_copy.sh && bash -n build-and-copy.sh`

Expected: exit 0 and all build-wrapper tests pass.

- [ ] **Step 5: Commit the profile**

```bash
git add build-and-copy.sh Dockerfile tests/test_build_and_copy.sh
git commit -m "feat: add GLM53 GB10 build profile"
```

### Task 3: Add the Qwen and GLM recipes

**Files:**
- Create: `recipes/qwen3.8-flash-next-nvfp4.yaml`
- Create: `recipes/glm-5.3-flash-nvfp4.yaml`
- Modify: `tests/test_recipes.sh`

**Interfaces:**
- Qwen recipe consumes `vllm-node-qwen` and build args for PR `53896`.
- GLM recipe consumes `vllm-node-glm` and `--glm53-gb10`.
- Both produce a cluster-only TP2 vLLM command suitable for `run-recipe.py --dry-run -n HOST1,HOST2`.

- [ ] **Step 1: Write failing recipe dry-run assertions**

Add tests that invoke each new recipe with `--dry-run -n 10.0.0.1,10.0.0.2` and assert the intended image, model ID, `--tensor-parallel-size 2`, and profile-specific flags: Qwen parser/tool settings; GLM block size 2304, Marlin MoE, FP8 KV, and native MTP.

- [ ] **Step 2: Run the recipe tests and verify the new assertions fail**

Run: `bash tests/test_recipes.sh`

Expected: the new tests fail because the recipe files do not exist.

- [ ] **Step 3: Create the two recipe YAML files**

Encode Qwen source build inputs and a conservative TP2 serving command for `Inferact/Qwen3.8-Flash-Next-NVFP4`. Encode GLM `--glm53-gb10` build args and the forum-derived safe TP2/MTP command for `LibertAIDAI/GLM-5.3-Flash-NVFP4`. Add comments with the NVIDIA forum URLs and the host-cache-reclamation prerequisite; do not add privileged cache-clearing commands.

- [ ] **Step 4: Run full development verification**

Run: `bash tests/test_build_and_copy.sh && bash tests/test_recipes.sh && ./run-recipe.sh qwen3.8-flash-next-nvfp4 --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2 && ./run-recipe.sh glm-5.3-flash-nvfp4 --config /dev/null --dry-run -n 10.0.0.1,10.0.0.2`

Expected: exit 0 and generated commands retain TP2 and the correct custom image tag.

- [ ] **Step 5: Commit recipes and tests**

```bash
git add recipes/qwen3.8-flash-next-nvfp4.yaml recipes/glm-5.3-flash-nvfp4.yaml tests/test_recipes.sh
git commit -m "feat: add Qwen and GLM53 Spark recipes"
```

### Task 4: Build and distribute the images

**Files:**
- No source changes.

**Interfaces:**
- Consumes: verified source changes and target host `192.168.200.13`.
- Produces: locally built and remotely loaded `vllm-node-qwen` and `vllm-node-glm` images.

- [ ] **Step 1: Inspect operational prerequisites**

Run the runbook-required read-only checks: architecture, Docker status, NVIDIA GPU visibility, free disk, and existing local/remote image IDs. Do not print `.env` values.

- [ ] **Step 2: Build and copy Qwen**

Run the existing Qwen build with PR 53896, FlashInfer v0.6.17, tag `vllm-node-qwen`, and `-c <worker-host>`.

- [ ] **Step 3: Build and copy GLM**

Run `./build-and-copy.sh --glm53-gb10 -c <worker-host>`.

- [ ] **Step 4: Verify resulting images**

Inspect the local and remote Docker image IDs for both tags. Report the IDs only, never configuration secrets.
