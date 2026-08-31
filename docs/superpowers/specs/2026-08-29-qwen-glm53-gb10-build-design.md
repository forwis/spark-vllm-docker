# Qwen and GLM-5.3 GB10 Build Design

> **GLM-5.3 note:** The GLM portions of this historical document are superseded
> by `docs/superpowers/specs/2026-08-31-glm53-sparse-mla-reimplementation-design.md`.
> Its Qwen portions remain historical records of the Qwen implementation.

## Goal

Provide reproducible local image builds for Qwen3.8 Flash Next and GLM-5.3
Flash on a two-node DGX Spark cluster, then distribute each image to the
explicitly selected worker host.

## Scope

- Preserve the existing Qwen source-build command as `vllm-node-qwen`.
- Add a `--glm53-gb10` build profile that creates `vllm-node-glm`.
- Vendor and pin the GLM-5.3 GB10 patch set rather than fetching mutable
  patches during the build.
- Add cluster-only Qwen and GLM recipe files that select those image tags.
- Do not launch containers, download model weights, alter networking, or
  automate host cache eviction.

## GLM-5.3 Profile

`--glm53-gb10` is an opt-in profile in `build-and-copy.sh`. It selects the
vLLM source ref `06569a8696076eeae9558928b00f035ded8f8b60`, applies PR 53906,
and builds FlashInfer from the vendored profile's pinned 0.6.18 nightly-era
revision rather than FlashInfer 0.6.17. The profile owns its Docker build
arguments and rejects incompatible overrides that could invalidate the
qualified patch set.

The runner image applies exact-match Python patch scripts after wheel
installation. A mismatch aborts the image build rather than silently applying
to an unknown upstream layout. The patches enable the FA2 NoPE MLA path on
SM121, disable unvalidated PDL on SM12x, correct sparse-indexer pool IDs, and
permit FP8 MLA KV cache without exceeding GB10 shared-memory limits. The
profile re-pins NCCL and Cutlass DSL after the FlashInfer nightly installation.

## Vendored Source

The repository will vendor only the runtime patch scripts and a provenance
record under `mods/glm53-gb10/`. The source is
`tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` at commit
`2815bcb63cd28daa3c52501da0c62dad0927e99b`. The provenance file lists each
upstream source file and the exact installed package files it modifies.

The DFlash2 overlay is intentionally out of scope: it requires a separate
draft-model checkout and bind mount. The included GLM recipe uses model-native
MTP instead.

## Recipes

`qwen3.8-flash-next-nvfp4.yaml` is a two-node, cluster-only recipe for
`Inferact/Qwen3.8-Flash-Next-NVFP4` and `vllm-node-qwen`. It encodes the
corresponding vLLM/FlashInfer build inputs and conservative TP2 serving
defaults.

`glm-5.3-flash-nvfp4.yaml` is a two-node, cluster-only recipe for
`LibertAIDAI/GLM-5.3-Flash-NVFP4` and `vllm-node-glm`. It uses TP2, FP8 KV,
block size 2304, Marlin MoE, GLM tool/reasoning parsers, and native MTP.
Its documentation states that each host needs its filesystem cache reclaimed
before launch; a container image cannot safely perform that privileged
host-level action.

## Verification and Operation

Mocked `build-and-copy.sh` tests will assert the GLM profile's pinned arguments
and runner patch mode. Recipe tests and explicit two-node dry-runs will verify
the generated commands without touching the cluster.

After development verification succeeds, the operator will build the Qwen
image and GLM profile image sequentially, copy each to `192.168.200.13`, and
verify the local and remote image IDs. No model download or server launch is
part of this request.
