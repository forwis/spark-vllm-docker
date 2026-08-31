#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOD_DIR="$PROJECT_DIR/mods/glm53-sparse-mla"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

GLM53_ARCHS=121a python3 - "$MOD_DIR" <<'PY'
import os
import runpy
import sys
import types
from pathlib import Path

root = Path(sys.argv[1])
captured = {}

setuptools = types.ModuleType("setuptools")
setuptools.setup = lambda **kwargs: captured.update(kwargs)
cpp_extension = types.ModuleType("torch.utils.cpp_extension")
cpp_extension.BuildExtension = object
cpp_extension.CUDAExtension = lambda **kwargs: kwargs
torch_utils = types.ModuleType("torch.utils")
torch_utils.cpp_extension = cpp_extension
torch = types.ModuleType("torch")
torch.utils = torch_utils
sys.modules.update({
    "setuptools": setuptools,
    "torch": torch,
    "torch.utils": torch_utils,
    "torch.utils.cpp_extension": cpp_extension,
})

os.chdir(root)
runpy.run_path(str(root / "setup.py"), run_name="__main__")

assert captured["name"] == "glm53_sparse_mla"
assert captured["packages"] == ["glm53_sparse_mla"]
assert captured["entry_points"]["vllm.general_plugins"] == [
    "glm53_sparse_mla = glm53_sparse_mla.backend:register",
    "glm53_moe_fix = glm53_sparse_mla.moe_fix:register",
]
nvcc = captured["ext_modules"][0]["extra_compile_args"]["nvcc"]
assert "-gencode=arch=compute_121a,code=sm_121a" in nvcc
assert "-gencode=arch=compute_120,code=compute_120" in nvcc
assert "--use_fast_math" not in nvcc
PY

python3 -m py_compile "$MOD_DIR"/glm53_sparse_mla/*.py
echo "GLM53 sparse-MLA package test passed."
