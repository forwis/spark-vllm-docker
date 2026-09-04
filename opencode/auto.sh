#!/usr/bin/env bash
# Select the OpenCode model-switch script from the first model served by vLLM.
#
# Required env:
#   VLLM_API_KEY   API key for the vLLM server
# Optional env:
#   VLLM_BASE_URL  base URL of the OpenAI-compatible API (default: http://192.168.1.31:54351/v1)
#   PYTHON_BIN     Python executable used to parse the response (default: python3)
set -euo pipefail

: "${VLLM_API_KEY:?VLLM_API_KEY is not set}"
base_url="${VLLM_BASE_URL:-http://192.168.1.31:54351/v1}"
python_bin="${PYTHON_BIN:-python3}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v curl >/dev/null 2>&1; then
  printf 'Error: curl is required\n' >&2
  exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Error: Python executable not found: %s\n' "$python_bin" >&2
  exit 1
fi

printf 'Fetching available models from %s/models\n' "$base_url"
models_json="$(curl -sf --connect-timeout 10 --max-time 60 \
  -H "Authorization: Bearer ${VLLM_API_KEY}" "${base_url}/models")" || {
  printf 'Failed to fetch %s/models\n' "$base_url" >&2
  exit 1
}

model="$("$python_bin" -c '
import json
import sys

data = json.load(sys.stdin).get("data")
if not isinstance(data, list) or not data:
    raise SystemExit("No models returned")
model = data[0].get("id") if isinstance(data[0], dict) else None
if not isinstance(model, str) or not model:
    raise SystemExit("First model has no valid id")
if model in {".", "..", "auto"} or "/" in model or "\\" in model:
    raise SystemExit("First model id is not a safe script name")
if any(ord(character) < 32 or ord(character) == 127 for character in model):
    raise SystemExit("First model id contains a control character")
sys.stdout.write(model)
' <<< "$models_json")" || {
  printf 'Failed to read the first model from %s/models\n' "$base_url" >&2
  exit 1
}

model_script="${script_dir}/${model}.sh"
if [[ -L "$model_script" ]]; then
  printf 'Refusing symbolic-link model script: %s\n' "$model_script" >&2
  exit 1
fi
if [[ "$model_script" -ef "${BASH_SOURCE[0]}" ]]; then
  printf 'Refusing recursive model script: %s\n' "$model_script" >&2
  exit 1
fi
if [[ ! -f "$model_script" || ! -x "$model_script" ]]; then
  printf 'No matching model script: %s\n' "$model_script" >&2
  exit 1
fi

printf 'Running model script: %s\n' "$model_script"
exec "$model_script"
