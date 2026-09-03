#!/usr/bin/env python3
"""Unit tests for the Qwen3.8 stability probe's deterministic workload."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "examples/qwen38-flash-next-stability-probe.py"


class StabilityProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("qwen38_probe", SCRIPT)
        assert spec and spec.loader
        cls.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.probe)

    def test_depth_profiles_keep_1m_only_in_canary(self) -> None:
        self.assertEqual(self.probe.DEFAULT_DEPTHS[-1], 250_000)
        self.assertEqual(self.probe.CANARY_DEPTHS[-2:], (500_000, 950_000))
        self.assertNotIn(500_000, self.probe.DEFAULT_DEPTHS)

    def test_twenty_turn_batch_contains_tool_results_and_planted_value(self) -> None:
        messages, expected = self.probe.make_depth_turns(48_000, 7, filler_words=100)
        self.assertGreaterEqual(sum(m["role"] == "user" for m in messages), 20)
        self.assertTrue(any(m["role"] == "tool" for m in messages))
        self.assertTrue(any("planted_value" in str(m.get("content")) for m in messages))
        self.assertRegex(expected, r"^Q38-PASS-48000-\d+$")

    def test_metric_groups_require_positive_activity(self) -> None:
        metrics = """
vllm:prefix_cache_queries_total 20
vllm:prefix_cache_hits_total 4
vllm:spec_decode_num_draft_tokens_total 30
vllm:spec_decode_num_accepted_tokens_total 10
"""
        activity = self.probe.metric_activity(metrics)
        self.assertGreater(activity["prefix"], 0)
        self.assertGreater(activity["mtp"], 0)


if __name__ == "__main__":
    unittest.main()
