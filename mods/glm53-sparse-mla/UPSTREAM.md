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
