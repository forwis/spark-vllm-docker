#!/bin/bash
set -euo pipefail

PREFIX="[dsv4f-vision-exp]"
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

python3 "$MOD_DIR/patch_dsv4f_vision.py" \
    --vllm-root "$VLLM_PACKAGE_ROOT" \
    --overlay-root "$MOD_DIR/overlay/vllm"
echo "$PREFIX Vision-Exp architecture and DSpark routing support are installed."
