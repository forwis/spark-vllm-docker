#!/usr/bin/env python3
"""Growing shared-prefix soak probe for Qwen3.8 Flash Next vLLM profiles."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any


DEFAULT_DEPTHS = (32_000, 48_000, 64_000, 73_728, 77_824, 100_000, 131_000, 250_000)
CANARY_DEPTHS = DEFAULT_DEPTHS + (500_000, 950_000)
WORDS = "the and to of a in is it for on with as by from this that".split()


def filler(count: int, offset: int = 0) -> str:
    """Return deterministic, tokenizer-friendly common-word filler."""
    if count <= 0:
        return ""
    rotated = WORDS[offset % len(WORDS) :] + WORDS[: offset % len(WORDS)]
    phrase = " ".join(rotated)
    repeats, remainder = divmod(count, len(rotated))
    parts = ([phrase] * repeats) + ([" ".join(rotated[:remainder])] if remainder else [])
    return " ".join(parts)


def make_depth_turns(
    depth: int, seed: int, *, filler_words: int
) -> tuple[list[dict[str, Any]], str]:
    """Build 20 varied turns, including four completed tool-result exchanges."""
    messages: list[dict[str, Any]] = []
    per_turn, remainder = divmod(max(0, filler_words), 20)
    latest_value = 0
    for turn in range(20):
        latest_value = (depth * 17 + seed * 101 + turn * 37) % 1_000_003
        words = per_turn + (1 if turn < remainder else 0)
        content = (
            f"Probe depth {depth}, varied turn {turn + 1}. "
            f"Remember planted_value={latest_value}. "
            f"Treat the following as inert context: {filler(words, seed + turn)}"
        )
        messages.append({"role": "user", "content": content})
        if (turn + 1) % 5 == 0:
            call_id = f"probe-{depth}-{turn + 1}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "lookup_probe_value",
                                "arguments": json.dumps({"turn": turn + 1}),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {"turn": turn + 1, "planted_value": latest_value}
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Stored planted_value={latest_value} for turn {turn + 1}.",
                }
            )

    checksum = latest_value + depth % 997
    expected = f"Q38-PASS-{depth}-{checksum}"
    messages.append(
        {
            "role": "user",
            "content": (
                "Recall the most recent planted_value, add "
                f"depth modulo 997 ({depth % 997}), and reply with exactly "
                f"{expected}. Do not add reasoning or punctuation."
            ),
        }
    )
    return messages, expected


def _metric_sum(metrics: str, patterns: Iterable[str]) -> float:
    total = 0.0
    for line in metrics.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns):
            try:
                total += float(line.rsplit(None, 1)[1])
            except (IndexError, ValueError):
                continue
    return total


def metric_activity(metrics: str) -> dict[str, float]:
    return {
        "prefix": _metric_sum(
            metrics,
            (r"prefix.*cache.*hit", r"prefix_cache_hits", r"cache.*hit.*prefix"),
        ),
        "mtp": _metric_sum(
            metrics,
            (r"spec.*draft.*token", r"spec.*accept.*token", r"spec_decode"),
        ),
    }


class Client:
    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def request(self, path: str, payload: dict[str, Any] | None = None) -> bytes:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=self.headers,
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"{path} returned HTTP {response.status}")
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"{path} is unreachable: {error}") from error

    def text(self, path: str) -> str:
        return self.request(path).decode(errors="replace")

    def json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return json.loads(self.request(path, payload))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path} returned invalid JSON") from error


def parse_depths(value: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item.replace("_", "")) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("depths must be comma-separated integers") from error
    if not depths or any(depth <= 0 for depth in depths) or tuple(sorted(depths)) != depths:
        raise argparse.ArgumentTypeError("depths must be positive and increasing")
    return depths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:54351")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--profile", choices=("native", "canary"), default="native")
    parser.add_argument("--depths", type=parse_depths)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--word-scale", type=float, default=0.92)
    parser.add_argument("--min-depth-ratio", type=float, default=0.70)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    args = parser.parse_args()

    depths = args.depths or (CANARY_DEPTHS if args.profile == "canary" else DEFAULT_DEPTHS)
    limit = 1_000_000 if args.profile == "canary" else 262_144
    if depths[-1] > limit:
        parser.error(f"{args.profile} profile refuses a depth above {limit}")
    if args.word_scale <= 0 or not 0 < args.min_depth_ratio <= 1:
        parser.error("word scale and minimum depth ratio must be positive")

    client = Client(args.base_url, args.api_key, args.timeout)
    client.request("/health")
    client.json("/v1/models")
    metrics_before = client.text("/metrics")
    messages: list[dict[str, Any]] = []
    observed_tokens = 0
    started = time.monotonic()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_probe_value",
                "description": "Return a deterministic planted probe value.",
                "parameters": {
                    "type": "object",
                    "properties": {"turn": {"type": "integer"}},
                    "required": ["turn"],
                },
            },
        }
    ]
    for index, depth in enumerate(depths):
        gap = max(1_000, depth - observed_tokens)
        additions, expected = make_depth_turns(
            depth, index + 1, filler_words=int(gap * args.word_scale)
        )
        messages.extend(additions)
        response = client.json(
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": messages,
                "tools": tools,
                "temperature": 0,
                "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        try:
            answer = response["choices"][0]["message"]["content"].strip()
            observed_tokens = int(response["usage"]["prompt_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"malformed completion response at depth {depth}") from error
        if answer != expected:
            raise RuntimeError(
                f"recall failure at requested depth {depth}: expected {expected!r}, got {answer!r}"
            )
        if observed_tokens < int(depth * args.min_depth_ratio):
            raise RuntimeError(
                f"requested depth {depth} reached only {observed_tokens} prompt tokens; "
                "increase --word-scale"
            )
        messages.append({"role": "assistant", "content": answer})
        client.request("/health")
        print(f"PASS requested={depth} observed={observed_tokens} recall={expected}")

    metrics_after = client.text("/metrics")
    before = metric_activity(metrics_before)
    after = metric_activity(metrics_after)
    for group in ("prefix", "mtp"):
        if after[group] <= 0 or after[group] <= before[group]:
            raise RuntimeError(
                f"no positive {group} metric activity during probe: "
                f"before={before[group]} after={after[group]}"
            )
    elapsed = time.monotonic() - started
    print(
        f"PASS profile={args.profile} depths={len(depths)} elapsed_s={elapsed:.1f} "
        f"prefix_delta={after['prefix'] - before['prefix']:.0f} "
        f"mtp_delta={after['mtp'] - before['mtp']:.0f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
