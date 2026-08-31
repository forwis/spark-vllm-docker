#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/mods/glm53-dflash2"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SITE_PACKAGES="$TMP_DIR/site-packages"
TARGET_DIR="$SITE_PACKAGES/vllm/model_executor/layers"
mkdir -p "$TARGET_DIR"
cat > "$TARGET_DIR/sparse_attn_indexer_kpool.py" <<'PY'
def prefill(logits, num_rows, select_k):
    if select_k:
        pool_topk = torch.empty(
            (num_rows, select_k), dtype=torch.int32, device=logits.device
        )

def decode(logits, num_rows, select_k, current_platform):
    if select_k:
        pool_topk = torch.empty(
            (num_rows, select_k), dtype=torch.int32, device=logits.device
        )
        if current_platform.is_cuda() and select_k in (512, 1024, 2048):
            return "persistent"
PY

python3 "$BUILD_DIR/patch_sm121_topk.py" \
    "$TARGET_DIR/sparse_attn_indexer_kpool.py"

test "$(grep -Fc 'pool_topk = torch.full(' "$TARGET_DIR/sparse_attn_indexer_kpool.py")" -eq 2
grep -Fq 'torch.cuda.get_device_properties(0).multi_processor_count >= 78' \
    "$TARGET_DIR/sparse_attn_indexer_kpool.py"
grep -q '^        if ($' "$TARGET_DIR/sparse_attn_indexer_kpool.py"
# The build patch is deliberately repeatable.
python3 "$BUILD_DIR/patch_sm121_topk.py" \
    "$TARGET_DIR/sparse_attn_indexer_kpool.py"

test "$(grep -Fc 'pool_topk = torch.full(' "$TARGET_DIR/sparse_attn_indexer_kpool.py")" -eq 2

for required in \
    Dockerfile.glm53-dflash2 \
    mods/glm53-dflash2/patch_nope_mla.py \
    mods/glm53-dflash2/patch_pdl.py \
    mods/glm53-dflash2/patch_v7.py \
    mods/glm53-dflash2/patch_v8_fp8.py \
    mods/glm53-dflash2/dflash2/speculator.py \
    mods/glm53-dflash2/patch_registry_and_select.py \
    mods/glm53-dflash2/patch_glm_aux_capture.py \
    mods/glm53-dflash2/patch_glm5_drafter_group.py \
    mods/glm53-dflash2/chat_template_mm.jinja; do
    test -f "$PROJECT_DIR/$required"
done

if grep -Fq 'ghcr.io/tonyd2wild/vllm-glm53-flash' \
    "$PROJECT_DIR/Dockerfile.glm53-dflash2"; then
    echo "GLM53 build must not use the reference's prebuilt image." >&2
    exit 1
fi

echo "GLM53 DFlash2 local-build test passed."
