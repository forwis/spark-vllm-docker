#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


path = Path(sys.argv[1])
text = path.read_text()
topk_pattern = re.compile(
    r"(?m)^(?P<indent>[ ]*)pool_topk = torch\.empty\(\n"
    r"(?P=indent)    \(num_rows, select_k\), dtype=torch\.int32, "
    r"device=logits\.device\n(?P=indent)\)"
)


def replace_topk(match: re.Match[str]) -> str:
    indent = match.group("indent")
    return (
        f"{indent}pool_topk = torch.full(\n"
        f"{indent}    (num_rows, select_k), -1, dtype=torch.int32, "
        f"device=logits.device\n{indent})"
    )


text, topk_count = topk_pattern.subn(replace_topk, text)
if topk_count == 0 and text.count("pool_topk = torch.full(") == 2:
    pass
elif topk_count != 2:
    raise SystemExit(
        "[glm53-dflash2] expected 2 top-k initialization anchor(s), "
        f"found {topk_count}"
    )
gate_pattern = re.compile(
    r"(?m)^(?P<indent>[ ]*)if current_platform\.is_cuda\(\) "
    r"and select_k in \(512, 1024, 2048\):$"
)


def replace_gate(match: re.Match[str]) -> str:
    indent = match.group("indent")
    lines = (
        "# SM121/GB10 (48 SMs, 99KB smem): persistent_topk oversubscribes past",
        "# ~24K ctx and its FilteredTopK fallback needs 128KB smem -> hard raise.",
        "# Route small-SM parts to top_k_per_row_decode instead.",
        "if (",
        "    current_platform.is_cuda()",
        "    and select_k in (512, 1024, 2048)",
        "    and torch.cuda.get_device_properties(0).multi_processor_count >= 78",
        "):",
    )
    return "\n".join(indent + line for line in lines)


text, gate_count = gate_pattern.subn(replace_gate, text)
if gate_count == 0 and text.count("persistent_topk oversubscribes past") == 1:
    pass
elif gate_count != 1:
    raise SystemExit(
        "[glm53-dflash2] expected 1 persistent top-k gate anchor, "
        f"found {gate_count}"
    )
compile(text, str(path), "exec")
path.write_text(text)
