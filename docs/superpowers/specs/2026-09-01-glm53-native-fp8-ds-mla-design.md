# GLM-5.3 Native FP8 DS-MLA Design

## Objective

Replace the qualified GLM-5.3 GB10 profile's custom BF16-only sparse-MLA
plugin with the native `fp8_ds_mla` path validated by
`/home/<user>/git/glm53-flash-cluster`, while retaining the existing
`LibertAIDAI/GLM-5.3-Flash-NVFP4` checkpoint and public repository interfaces.

The port is deliberately limited to the cluster reference's generic NoPE MLA
padding and native SM120/SM121 sparse-MLA compatibility chain. It does not
adopt the cluster reference's `local-inference-lab` checkpoint, 512K serving
profile, cluster orchestrator, or checkpoint-specific model and quantization
patches.

## Qualified Profile Boundary

The following public interfaces remain stable:

- build flag: `build-and-copy.sh --glm53-gb10`;
- image tag: `vllm-node-glm`;
- recipe: `recipes/glm-5.3-flash-nvfp4.yaml`;
- model: `LibertAIDAI/GLM-5.3-Flash-NVFP4`;
- served model name: `glm-5.3-flash`;
- two-node TP2 cluster requirement;
- API port default: `54351`.

The qualified runtime remains conservative:

- GPU memory utilization: `0.80`;
- maximum model length: `65,536`;
- maximum sequences: `2`;
- maximum batched tokens: `1,024`;
- block size: `256`;
- eager execution;
- model-native MTP with three speculative tokens;
- `glm47` tool parser and `deepseek_r1` reasoning parser;
- multimodal processing enabled.

No `--kv-cache-memory-bytes` value is added. The reference's 9 GiB pin is a
512K-context, vision-warmup memory setting and is not part of this 65K profile.

## Image Architecture

The GLM image will be rebuilt from the exact official GLM-5.3 base-image digest
validated by the cluster reference:

```text
vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
```

The custom sparse-MLA package, CUDA extension, general-plugin registration,
and build inputs under `mods/glm53-sparse-mla/` will be removed. The image will
instead apply a vendored copy of the cluster reference's `patch_mla.py` during
the Docker build. The vendored directory will record the source repository,
commit, source path, license, and file checksum.

The source is pinned to `glm53-flash-cluster` commit
`50e344433076efce702029e3f446c54a80916dc9`; the patch was introduced by commit
`ac425d3ae51e6e1472a9000f4e242e63b6510987`, is covered by Apache-2.0, and has
SHA-256 `12ed6565328c8b72edae62207b2640436355475d033729046170cdf5db96954f` at
the pinned tree.

The native-FP8 Dockerfile will:

1. copy the vendored patch into the build context;
2. execute it against the digest-pinned installed vLLM and FlashInfer trees;
3. rely on its exact anchor-count assertions to fail on incompatible source
   drift;
4. verify distinct markers for every applied patch section;
5. Python-compile each modified Python file;
6. remove temporary patch inputs from the final image layer.

`build-and-copy.sh --glm53-gb10` will select this Dockerfile while preserving
its existing qualified-profile override rejection and output image tag.

## Native FP8 Data Flow

The recipe will pass the existing checkpoint as a local Hugging Face snapshot
path. This preserves the separate startup fix for the checkpoint processor:
the installed GLM processor directly opens `processor_config.json` and cannot
use a repository identifier at that call site.

The recipe will set:

```text
VLLM_MLA_NOPE_PAD_ROPE=1
--kv-cache-dtype fp8_ds_mla
```

When the model has `qk_rope_head_dim == 0`, the environment-gated patch will:

1. expose a 64-element positional component to the inner MLA cache
   specification, producing the native DeepSeek `512+64` layout;
2. append zeros to the query positional portion immediately before attention;
3. construct an all-zero `k_pe` tensor of the same width;
4. route the resulting layout through vLLM's native `fp8_ds_mla` cache and
   FlashInfer SM120/SM121 sparse-MLA implementation.

The padding is mathematically neutral: the added query and key components are
zero, are unaffected by rotation, and contribute zero to every attention
logit.

The remainder of the imported patch chain will:

- pass the actual GLM k-pool top-k table width to the SM120 path;
- extend the FlashInfer sparse-decode dispatch allowlist for the GLM table
  geometry;
- compact the padded k-pool table to the native compiled width while retaining
  the always-selected tail entries.

The previous `VLLM_GLM53_CUDA_SPARSE_MLA` gate will be removed. No custom
attention backend will remain installed, so native vLLM/FlashInfer exclusively
owns the sparse-MLA execution path.

## Deliberately Excluded Cluster Changes

The following `glm53-flash-cluster` elements are not compatible with the
retained checkpoint or are outside this repository's orchestration boundary:

- `model.patch`, which adapts the `local-inference-lab` checkpoint layout;
- `modelopt.patch`, which fixes that checkpoint's MXFP8 MTP namespace mapping;
- the two CC12 indexer resource patches required by the 512K lab profile;
- the lab checkpoint and its pinned revision;
- 512K context, 9/10 GiB KV-pool pins, and lab-specific multimodal limits;
- the cluster reference's Compose files, launch scripts, watchdog, teardown,
  and host-management behavior;
- privileged filesystem-cache clearing.

The existing checkpoint's calibrated scale data remains authoritative. The
currently installed MoE scale plugin is inactive and will be removed with the
custom package.

## Failure Handling

The image build must stop if:

- the official base digest cannot be resolved;
- any patch anchor occurs an unexpected number of times;
- any expected post-patch marker is absent;
- a modified Python file fails compilation;
- an obsolete custom-plugin artifact remains in the qualified build inputs.

The recipe continues to require a complete local Hugging Face snapshot on both
nodes. A missing `refs/main` or snapshot file will fail before vLLM startup,
rather than falling back to the broken repository-ID processor path.

Runtime failures from unsupported backend selection, FP8 cache initialization,
or FlashInfer dispatch must remain visible. The recipe will not silently fall
back to the removed BF16 plugin.

The repository will continue to document operator-managed filesystem-cache
reclamation before launch. It will not add cache-clearing commands to recipes,
containers, tests, or launch orchestration.

## Testing Strategy

Implementation will follow red-green-refactor cycles.

A focused fixture test will execute the vendored patch against controlled
representations of all target files and verify the observable transformations:

- the NoPE environment gate and 64-wide pad are installed;
- the query and key positional tensors are zero-padded;
- SM120 top-k width handling is updated;
- the FlashInfer dispatch allowlist includes the GLM geometry;
- k-pool tail compaction is present;
- a second application is idempotent;
- a missing or duplicated anchor fails loudly.

Build-script tests will require the digest-pinned base, native-FP8 Dockerfile,
patch verification, and stable `vllm-node-glm` tag. They will reject the old
custom extension, package installation, plugin entry points, and BF16 build
description.

Recipe tests will require:

- local snapshot resolution for `LibertAIDAI/GLM-5.3-Flash-NVFP4`;
- `VLLM_MLA_NOPE_PAD_ROPE=1`;
- `--kv-cache-dtype fp8_ds_mla`;
- block size 256 and all retained conservative serving settings;
- absence of `VLLM_GLM53_CUDA_SPARSE_MLA`, `bfloat16` KV, and
  `--language-model-only`.

Development verification will run the focused patch and build tests, the full
recipe suite, image-consistency tests, runtime-vLLM-PR tests, shell/Python
syntax checks where applicable, an explicit two-node recipe dry-run, and
`git diff --check`. Development does not include a real image build, model
download, or cluster launch.

## Operational Acceptance

If operational verification is requested separately, success requires:

1. an image built from the pinned base and distributed with identical image IDs;
2. a complete retained checkpoint snapshot on both nodes;
3. operator-managed filesystem-cache reclamation before launch;
4. logs showing native `fp8_ds_mla` cache/backend initialization;
5. successful model and KV-cache initialization and API readiness;
6. coherent deterministic text generation;
7. one image-input smoke test proving multimodal processing remains enabled.

Operational verification must use the repository runbook and must not add
privileged cache-management behavior to source-controlled workflows.

## Rollback

The qualified image will contain only one sparse-MLA owner. It will not expose
a runtime switch between the native FP8 path and the custom BF16 backend.
Rollback is source-level: rebuild the qualified image from the preceding BF16
implementation commit and redistribute that image consistently across both
nodes.
