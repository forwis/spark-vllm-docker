#!/bin/bash
set -euo pipefail

PREFIX="[qwen38-flash-next-stability]"
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

python3 "$MOD_DIR/patch_vllm.py" --vllm-root "$VLLM_PACKAGE_ROOT"
echo "$PREFIX Installed qualified prefix/MTP, Mamba, slot-mapping, and MoE guards."
