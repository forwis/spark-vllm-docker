# Agent Instructions

These instructions apply to the entire repository.

## Repository

This project provides Bash and Python orchestration for running vLLM on one or
more NVIDIA DGX Spark systems. Work from the repository root and read `README.md`
for the public project overview.

## Choose One Guide

- **Use or operate the repository:** For host preparation, recipe selection,
  cluster discovery, image or model setup, recipe launches, and live-server
  verification, follow `docs/AGENT_RUNBOOK.md`.
- **Develop the repository:** For inspection, fixes, features, reviews, tests,
  or changes to scripts, recipes, mods, Dockerfiles, and documentation, follow
  `docs/AGENT_DEVELOPMENT.md`.
- **Both:** Follow the development guide first. Use the operational runbook
  afterward only when the user also requested a real build, download, or launch.

Read only the guide relevant to the task unless the work crosses that boundary.
A recipe `--dry-run` used to validate generated commands is development. A
non-dry recipe run, `--setup`, discovery, image preparation, model download, or
container launch is operation.

## Common Boundaries

- Inspect before changing repository, host, container, or cluster state.
- Preserve unrelated user changes and existing local configuration or artifacts.
- Do not expose credentials or `.env` contents in chat, logs, diffs, or commands.
- Operational tasks do not authorize source changes. Development tasks do not
  authorize real deployments. Perform both only when the user requests both.
- Do not prune, overwrite, stop, remove, or force-refresh existing resources
  unless the requested task requires it.

## Web Research

- Topic pages on `forums.developer.nvidia.com` lazy-load content while scrolling.
  Scroll down until no additional posts load before treating a topic as fully
  inspected or concluding that information is absent.

## Qwen and GLM-5.3 GB10 Profiles

- `vllm-node-qwen` is locally built from the official
  `vllm/vllm-openai:qwen38-flash-next` base. Preserve that qualified base and
  its SM121 architecture inputs unless a task explicitly changes them.
- **Qwen3.8 Flash Next implementation reference URLs:**
  Inspect the complete, current contents of these evolving sources before
  changing or qualifying the profile. Follow later corrections and withdrawal
  notices; do not treat an earlier successful launch or bounded benchmark as
  production-stability evidence.
  - `https://forums.developer.nvidia.com/t/qwen3-8-flash-next/381228/63`
  - `https://forums.developer.nvidia.com/t/qwen3-8-flash-next-nvfp4-on-2x-gb10-long-agent-service-crash-isolation-42-4-tok-s-qualified-tp2/381836`
  - `https://github.com/vllm-project/vllm/pull/53896`
  - `https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml`
- `--glm53-gb10` locally builds `vllm-node-glm` from the digest-pinned official
  GLM-5.3 vLLM base plus the vendored NoPE/native-`fp8_ds_mla` patch chain from
  `glm53-flash-cluster`. Its base digest, retained LibertAIDAI checkpoint,
  native FP8 cache, block size 256, SM121 TP2 topology, and serving flags are a
  single qualified profile; do not mix it with manual vLLM, FlashInfer, Torch,
  GPU-architecture, KV-dtype, or experimental-profile overrides.
- The GLM recipe is cluster-only and requires operator-managed filesystem-cache
  reclamation on each host before launch. Do not add privileged cache-clearing
  commands to recipes, containers, or ordinary development validation.
- **GLM-5.3 implementation reference URLs:**
  *(Note: `forums.developer.nvidia.com` pages require scrolling down to lazy-load all web contents)*
  - `https://forums.developer.nvidia.com/t/intel-glm-5-3-flash-w4a16-autoround/382041`
  - `https://forums.developer.nvidia.com/t/glm-5-3-flash-320b-total-parameters-18b-active/381350/149`
  - `https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-gb10-speculative-decoding-makes-long-prefill-ttft-alternate-2x-after-a-mixed-workload-plus-3-knobs-that-measurably-helped/382099`
  - `https://forums.developer.nvidia.com/t/glm-5-3-flash-nvfp4-on-2x-dgx-spark-vllm-tp-2-docker-compose/381541/4`
  - `https://github.com/Libertai/glm53-flash-vllm-gb10.git`
  - `https://github.com/local-inference-lab/vllm.git`

## DeepSeek V4 Vision-Exp References

- For future DeepSeek V4 Flash Vision-Exp investigation or improvement, inspect
  the complete current contents of these sources and reconcile them with the
  immutable revisions recorded by the relevant recipe or mod. Treat them as
  research inputs, not permission to mix unqualified runtime components:
  - `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
  - `https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
  - `https://github.com/a939984672/vllm`
  - `https://github.com/jasl/vllm/tree/codex/ds4-sm120-min-enable`
