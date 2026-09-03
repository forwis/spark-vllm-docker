# Qwen3.8 Flash Next stability mod

This launch-time mod backports narrowly scoped guards to the digest-pinned
official `qwen38-flash-next` image. It validates every expected source anchor
before replacing any file and refuses unknown vLLM layouts.

The fixes reserve the MTP replay block during Mamba prefix lookup, seed and
schedule Mamba state on the Mamba group's actual block grid, bound both slot
mapping kernels to the request row, and disable FlashInfer CUTLASS's
nondeterministic fused NVFP4 final reduction.

Sources: vLLM [#48375](https://github.com/vllm-project/vllm/pull/48375),
[#53798](https://github.com/vllm-project/vllm/pull/53798),
[#54076](https://github.com/vllm-project/vllm/pull/54076),
[#54296](https://github.com/vllm-project/vllm/pull/54296), and
[#54948](https://github.com/vllm-project/vllm/pull/54948). The fine-grained
`max_length` form of #48375 follows the validated GB10 reproduction in the
Qwen3.8 community investigation rather than the coarser upstream block-count
form.

These patches address known correctness hazards. They do not resolve or hide
the separate high-context deep-prefill GPU wedge tracked in vLLM #54629.
