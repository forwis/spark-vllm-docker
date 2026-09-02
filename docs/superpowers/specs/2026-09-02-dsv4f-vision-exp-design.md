# DeepSeek V4 Flash Vision-Exp on DSpark v2 Design

## Objective

Add a qualified two-DGX-Spark recipe for
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` by extending the locally built
`vllm-node-dsv4f:v2` runtime. The port combines the existing JASL SM121 and
DSpark implementation with the focused multimodal support from
`a939984672/vllm` and the larger serving profile validated by
`MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`.

The final runtime must use a locally built image. No published Anemll, vLLM,
MiaAI, or other prebuilt final image is an acceptable dependency.
This experimental port is intended to make the selected profile runnable; it
does not claim full production equivalence with the community Vision branch.

## Qualified Source Boundary

The v2 image's embedded build metadata is the authoritative base profile:

- image tag: `vllm-node-dsv4f:v2`;
- vLLM repository: `https://github.com/jasl/vllm.git`;
- vLLM commit: `9ad62027bc84ca0ccbcc40853179312de770220c`;
- vLLM branch of origin: `codex/ds4-sm120-min-enable`;
- FlashInfer commit: `a0a6b019b9b27d49d209f85d028a1ae5a9b347d7`;
- GPU architecture: `12.1a`;
- Torch stack: Torch 2.13, torchvision 0.28, torchaudio 2.11;
- CUTLASS DSL: 4.7;
- Transformers 5 compatibility enabled;
- B12X disabled.

The recipe's build arguments will reproduce those source inputs if the v2
image is absent or the operator explicitly requests a rebuild. If the local v2
tag already exists, the normal recipe behavior will reuse it. This design does
not pull or retag a published final runtime image.

The image itself already supplies the qualified SM121 native components:

- DeepSeek V4 model and sparse-MLA support;
- DSpark target/draft integration;
- the packed `fp8_ds_mla` cache writer;
- FlashInfer SM120/SM121 sparse-MLA decode support;
- the architecture-specific Torch and CUDA build.

## Vision Port

The port evaluates the focused changes represented by the following
commits from the `dsv4-vision-exp` branch of `a939984672/vllm`:

- `f9ff02d`: DeepSeek V4 Vision multimodal model support;
- `b706973`: skip the unused hash-gate routing bias;
- `979e41f`: apply the vision-specific `bias_vl` expert-routing bias;
- `ce9dd9c`: compute the vision-visible attention window (not selected);
- `7996ea0`: clamp multimodal-prefix attention metadata (not selected).

Only the Python runtime delta compatible with the pinned JASL sparse-MLA
backend will be ported. The branch's full v0.28 tree, build system,
documentation, unrelated source changes, and unsupported multimodal-prefix
attention path will not be copied over the JASL base.

The runtime mod will add the Vision model, tower, and multimodal preprocessing
modules and patch the existing JASL files that own:

- model registry registration for
  `DeepseekV4VForConditionalGeneration`;
- input token validation for the five out-of-vocabulary image sentinels;
- `bias_vl` creation and image-position routing;
- fused top-k routing with a modality-specific bias;
- DSpark draft-weight loading for `.ffn.gate.bias_vl`.

The port has two known source-overlap sites. Their resolution is fixed by
this design:

1. retain JASL's negative-token validation, but allow tokens through
   `vocab_size + 4` only when `vision_n_layers > 0`;
2. retain JASL's `_sync_fused_moe_metadata()` call, then attach `bias_vl` to
   the constructed router.

JASL's `DeepSeekV4DSparkLayer` constructs the same `DeepseekV4MoE` class used
by target layers. Both the checkpoint-facing community port and the
instantiated JASL parameter use `.ffn.gate.bias_vl`, so the draft loader must
retain that name rather than remap it to the nonexistent
`.ffn.gate.e_score_correction_bias_vl`. The existing text-bias mapping remains,
and unknown optional weights are skipped instead of indexed unconditionally.

The mod will be tied to the exact v2 source layout, be safe to apply on both
cluster nodes, recognize an already-applied state, and fail before server
startup if an expected anchor or installed semantic block has drifted. The
outer multimodal wrapper implements the pinned `SupportsEagle3` interface and
delegates auxiliary-hidden-state configuration to the inner language model.
Both text and Vision routes preserve the stock fused shared-expert append
behavior. The mod will not compile or replace any native CUDA component.

## Attention Qualification

The pinned `DeepseekV4SparseMLABackend` inherits
`supports_mm_prefix() == False`, and backend validation rejects
`use_mm_prefix`. The recipe therefore does not set `is_mm_prefix_lm`, the mod
does not install `compute_vision_visible_window`, and the wrapper does not set
`mm_prefix_clamp_sliding_window`.

The Vision tower itself retains its normal bidirectional image encoding, but
the resulting image tokens enter the language model through ordinary causal
decoder attention. This port does not implement bidirectional or full-visible
image-token attention. A sparse-MLA backend that supports that behavior is
separate future work, not part of this qualification.

## KV-Cache Choice

The recipe will use:

```text
--kv-cache-dtype fp8_ds_mla
--block-size 256
```

The larger MiaAI/Anemll profile spells its cache dtype `nvfp4_ds_mla`, but its
documented Stage-C DeepSeek V4 implementation uses the same padded FP8 sparse
MLA envelope and routes it through the FP8 kernel. It is not the abandoned
true-NVFP4 layout experiment. Adding that alias to v2 would not reduce memory
or increase context capacity.

Using v2's native `fp8_ds_mla` name preserves the reference profile's actual
data path while avoiding an unnecessary compatibility alias and the risk of
incorrect NVFP4 dispatch.

## Recipe Profile

The new recipe will be cluster-only and target exactly two DGX Spark nodes with
tensor parallelism 2. Its defaults will port the larger validated runtime
profile:

- model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`;
- served model name: `deepseek-v4-flash-vision-exp`;
- context ceiling: `1,048,576` tokens;
- maximum sequences: `6`;
- maximum batched tokens: `8,192`;
- long-prefill threshold: `1,024`;
- GPU memory utilization: `0.835`;
- prefix caching enabled;
- chunked prefill enabled;
- async scheduling enabled;
- DSpark speculative decoding with six tokens and probabilistic draft
  sampling;
- maximum eight images per prompt;
- DeepSeek V4 tokenizer, reasoning parser, and tool-call parser;
- automatic tool choice enabled;
- generation config source set to `vllm`.

Six speculative tokens are deliberate: the value is at least the checkpoint's
DSpark block size of five and is divisible by Vision-Exp's `n_predict` value of
three.

The command will include the Vision architecture override required by the
community port:

```text
--hf-overrides
{"architectures":["DeepseekV4VForConditionalGeneration"]}
```

The recipe will omit distributed-executor, node-count, node-rank, headless,
and rendezvous flags because `run-recipe.py` and `launch-cluster.sh` own those
arguments.

The model will track its Hugging Face `main` revision because the repository's
download orchestration does not currently expose a revision field and the user
did not request a checkpoint pin. The qualified vLLM and FlashInfer source
inputs remain pinned independently.

## Existing 0731 Recipe

`recipes/deepseek-v4-flash-0731-jasl.yaml` currently names
`vllm-node-dsv4f:v1`. Since v2 is the confirmed current local build, that
recipe will be updated to use `vllm-node-dsv4f:v2`, and its stale B12X comment
and description will be corrected to name JASL v2. Its model and serving
parameters remain unchanged; the separate B12X recipe is not modified.

No other DeepSeek, GLM, Qwen, or user-modified recipe will be changed.

## Failure Handling

The Vision mod must stop the launch if:

- the installed vLLM tree cannot be located;
- the expected JASL v2 anchors are missing or duplicated;
- a new Vision module would overwrite an unexpected existing file;
- the model registry cannot expose the Vision architecture;
- any modified or added Python file fails compilation;
- a second application cannot prove the installed state is complete.

The port will not silently fall back to a text-only architecture, disable
DSpark, change KV-cache layout, or copy whole source files from a mismatched
v0.28 base.

Runtime failures from model loading, FlashInfer dispatch, multimodal
preprocessing, or DSpark verification remain visible. No real image build,
model download, cluster launch, or cache reclamation is part of development
validation.

## Testing Strategy

Implementation will follow red-green-refactor cycles.

A focused mod test will apply the port to a controlled copy of the exact JASL
v2 source files and verify:

- all three new Vision modules are installed;
- the Vision architecture is registered;
- sentinel token validation is gated by Vision configuration;
- `bias_vl` is constructed and attached without removing JASL metadata sync;
- target routing receives the modality-specific bias;
- DSpark retains and loads `bias_vl` weights without a rejected remap;
- the outer wrapper exposes and delegates the pinned EAGLE3 interface;
- text and Vision routes both append stock fused shared-expert slots;
- mm-prefix configuration and visible-window patching are absent;
- all changed Python files compile;
- a second application is idempotent;
- missing, duplicated, foreign, or semantically corrupted installed blocks
  fail loudly without partial writes.

Recipe tests will require the exact local-source build pins and all larger
profile settings. They will reject:

- any published final image dependency;
- the obsolete v1 tag;
- `nvfp4_ds_mla` in the new Vision recipe;
- `is_mm_prefix_lm` in the new Vision recipe;
- unsupported single-node use;
- runner-owned distributed flags inside the recipe command.

Development verification will run the focused mod test, the full recipe suite,
the launch image-consistency and vLLM-PR suites, an explicit two-node dry-run
against `/dev/null`, syntax checks for changed shell and Python files, and
`git diff --check`.

## Operational Acceptance

If a real build and launch are requested separately, acceptance requires:

1. a fresh local image built from the pinned JASL and FlashInfer sources;
2. identical v2 image IDs on both nodes;
3. a complete Vision-Exp snapshot available on both nodes;
4. successful application of the Vision mod on both idle containers;
5. logs showing `fp8_ds_mla`, TP2, and DSpark-6 initialization;
6. API readiness and coherent deterministic text generation;
7. one image-input request proving tower, preprocessing, sentinel handling,
   and `bias_vl` routing initialize successfully;
8. one request showing DSpark draft acceptance without a draft-weight loader
   failure.

Operational work must follow `docs/AGENT_RUNBOOK.md`. It must not add
privileged cache clearing, overwrite local model state, or use a published
final runtime image.

## Rollback

Rollback is source-level and reversible:

- remove the new Vision recipe and mod;
- restore the 0731 JASL recipe's previous image tag if required;
- rebuild or redistribute the preceding local v2 image unchanged.

The design does not mutate the existing v2 image object, model snapshots, or
cluster state during development.
