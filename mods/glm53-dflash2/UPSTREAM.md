# GLM-5.3 DFlash2 runtime provenance

This runtime layer tracks
[`tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
at commit `3eef46632c45ffb6c397de0716c23b3d2d594798`.

- `Dockerfile.glm53-dflash2` starts from the reference's official
  `vllm/vllm-openai:glm53-flash-arm64-cu130` base and applies its v1 through v8
  patch chain locally.
- `patch_sm121_topk.py` applies the two later changes from upstream
  `docker/sparse_attn_indexer_kpool_sm121.py` relative to vLLM commit
  `487ecf187`: initialize partial top-k output to `-1`, and avoid
  `persistent_topk` on small-SM devices such as GB10.
- `qwen3_dflash2.py`, `dflash2/`, and the `patch_*drafter*`, registry, KV, and
  auxiliary-capture scripts come from upstream `docker/dflash2-overlay/`.
- `chat_template_mm.jinja` is upstream's root-level multimodal template.
- `build-and-copy.sh --glm53-gb10` always invokes the dedicated local
  Dockerfile; it never pulls the reference's final prebuilt image.

The recipe selects `RedHatAI/GLM-5.3-Flash-NVFP4` instead of a ModelOpt
checkpoint based on [vLLM issue #54150](https://github.com/vllm-project/vllm/issues/54150),
which reports invalid byte-token output from tested ModelOpt GLM-5.3 NVFP4
checkpoints while the compressed-tensors checkpoint remained clean.
