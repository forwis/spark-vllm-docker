#!/usr/bin/env python3

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


CASES = {
    "glm-5.3-flash.sh": {
        "model_id": "glm-5.3-flash",
        "model": {
            "name": "GLM-5.3-Flash",
            "cost": {"input": 0.075, "output": 0.25},
            "attachment": True,
            "reasoning": True,
            "temperature": True,
            "tool_call": True,
            "interleaved": "reasoning_content",
            "limit": {"context": 1048576, "output": 163840},
            "modalities": {
                "input": ["text", "image", "video"],
                "output": ["text"],
            },
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "reasoningEffort": "max",
                "chat_template_kwargs": {
                    "reasoning_effort": "max",
                    "clear_thinking": True,
                },
            },
        },
        "profiles": {
            "general": (1.0, 0.95, "max"),
            "scout": (1.0, 0.95, "high"),
            "explore": (1.0, 0.95, "low"),
        },
    },
    "deepseek-v4-flash-vision-exp.sh": {
        "model_id": "deepseek-v4-flash-vision-exp",
        "model": {
            "name": "DeepSeek-V4-Flash-Vision-Exp",
            "cost": {"input": 0.22, "output": 0.66},
            "attachment": True,
            "reasoning": True,
            "temperature": True,
            "tool_call": True,
            "interleaved": "reasoning_content",
            "limit": {"context": 1048576, "output": 393216},
            "modalities": {
                "input": ["text", "image"],
                "output": ["text"],
            },
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "reasoningEffort": "max",
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": "max",
                },
            },
        },
        "profiles": {
            "general": (1.0, 0.95, "max", True),
            "scout": (1.0, 0.95, "high", True),
            "explore": (1.0, 1.0, "low", False),
        },
    },
}


class AdditionalModelScriptsTest(unittest.TestCase):
    def run_script(self, script_name: str, config_root: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["XDG_CONFIG_HOME"] = str(config_root)
        return subprocess.run(
            ["bash", str(ROOT / script_name)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def expected_agent_options(self, script_name: str, profile: tuple) -> dict:
        temperature, top_p, effort, *thinking = profile
        if script_name == "glm-5.3-flash.sh":
            return {
                "temperature": temperature,
                "top_p": top_p,
                "reasoningEffort": effort,
                "chat_template_kwargs": {
                    "reasoning_effort": effort,
                    "clear_thinking": True,
                },
            }
        return {
            "temperature": temperature,
            "top_p": top_p,
            "reasoningEffort": effort,
            "chat_template_kwargs": {
                "thinking": thinking[0],
                "reasoning_effort": effort,
            },
        }

    def test_scripts_install_exact_model_and_agent_profiles(self) -> None:
        for script_name, expected in CASES.items():
            with self.subTest(script=script_name), tempfile.TemporaryDirectory() as directory:
                config_root = Path(directory)
                config_dir = config_root / "opencode"
                config_dir.mkdir()
                config_path = config_dir / "opencode.json"
                config_path.write_text(
                    """
                    {
                      // Preserve provider connection and unrelated settings.
                      "$schema": "https://opencode.ai/config.json",
                      "permission": {"read": "allow"},
                      "provider": {
                        "vllm": {
                          "npm": "@ai-sdk/openai-compatible",
                          "options": {"baseURL": "http://localhost:8000/v1"},
                          "models": {"old": {"name": "Old"}},
                        },
                      },
                      "agent": {
                        "general": {"description": "General", "mode": "subagent", "stale": true},
                        "scout": {"description": "Scout", "mode": "subagent", "stale": true},
                        "explore": {"description": "Explore", "mode": "subagent", "stale": true},
                      },
                      "model": "vllm/old",
                    }
                    """,
                    encoding="utf-8",
                )
                config_path.chmod(0o640)

                result = self.run_script(script_name, config_root)

                self.assertEqual(result.returncode, 0, result.stderr)
                config = json.loads(config_path.read_text(encoding="utf-8"))
                model_id = expected["model_id"]
                self.assertEqual(config["model"], f"vllm/{model_id}")
                self.assertEqual(
                    config["provider"]["vllm"]["options"],
                    {"baseURL": "http://localhost:8000/v1"},
                )
                self.assertEqual(
                    config["provider"]["vllm"]["models"],
                    {model_id: expected["model"]},
                )
                self.assertEqual(config["permission"], {"read": "allow"})
                self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o640)
                backups = list(config_dir.glob("opencode.json.bak.*"))
                self.assertEqual(len(backups), 1)
                self.assertIn('"old"', backups[0].read_text(encoding="utf-8"))

                for name, profile in expected["profiles"].items():
                    temperature, top_p, _effort, *_thinking = profile
                    agent = config["agent"][name]
                    self.assertEqual(
                        set(agent),
                        {"description", "mode", "temperature", "top_p", "options"},
                    )
                    self.assertEqual(agent["description"], name.capitalize())
                    self.assertEqual(agent["mode"], "subagent")
                    self.assertEqual(agent["temperature"], temperature)
                    self.assertEqual(agent["top_p"], top_p)
                    self.assertEqual(agent["options"], self.expected_agent_options(script_name, profile))

                first_content = config_path.read_bytes()
                first_mtime = config_path.stat().st_mtime_ns
                second = self.run_script(script_name, config_root)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertEqual(config_path.read_bytes(), first_content)
                self.assertEqual(config_path.stat().st_mtime_ns, first_mtime)
                self.assertEqual(len(list(config_dir.glob("opencode.json.bak.*"))), 1)

    def test_scripts_create_all_required_agents_when_agent_config_is_absent(self) -> None:
        for script_name, expected in CASES.items():
            with self.subTest(script=script_name), tempfile.TemporaryDirectory() as directory:
                config_root = Path(directory)
                config_dir = config_root / "opencode"
                config_dir.mkdir()
                config_path = config_dir / "opencode.json"
                config_path.write_text(
                    json.dumps({"provider": {"vllm": {"models": {}}}}),
                    encoding="utf-8",
                )

                result = self.run_script(script_name, config_root)

                self.assertEqual(result.returncode, 0, result.stderr)
                agents = json.loads(config_path.read_text(encoding="utf-8"))["agent"]
                self.assertEqual(set(agents), {"general", "scout", "explore"})
                for name, profile in expected["profiles"].items():
                    self.assertEqual(agents[name]["mode"], "subagent")
                    self.assertTrue(agents[name]["description"])
                    self.assertEqual(agents[name]["options"], self.expected_agent_options(script_name, profile))

    def test_scripts_fail_safely_for_missing_invalid_and_symlinked_configs(self) -> None:
        for script_name in CASES:
            with self.subTest(script=script_name, case="missing"), tempfile.TemporaryDirectory() as directory:
                config_root = Path(directory)
                result = self.run_script(script_name, config_root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("not found", result.stderr.lower())
                self.assertFalse((config_root / "opencode").exists())

            with self.subTest(script=script_name, case="invalid"), tempfile.TemporaryDirectory() as directory:
                config_root = Path(directory)
                config_dir = config_root / "opencode"
                config_dir.mkdir()
                config_path = config_dir / "opencode.json"
                original = b'{"provider": {broken}\n'
                config_path.write_bytes(original)
                result = self.run_script(script_name, config_root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid jsonc", result.stderr.lower())
                self.assertEqual(config_path.read_bytes(), original)
                self.assertEqual(list(config_dir.glob("opencode.json.bak.*")), [])

            with self.subTest(script=script_name, case="symlink"), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_root = root / "config"
                config_dir = config_root / "opencode"
                config_dir.mkdir(parents=True)
                target = root / "managed-opencode.json"
                original = json.dumps({"provider": {"vllm": {"models": {}}}})
                target.write_text(original, encoding="utf-8")
                (config_dir / "opencode.json").symlink_to(target)
                result = self.run_script(script_name, config_root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("symbolic link", result.stderr.lower())
                self.assertTrue((config_dir / "opencode.json").is_symlink())
                self.assertEqual(target.read_text(encoding="utf-8"), original)
                self.assertEqual(list(config_dir.glob("opencode.json.bak.*")), [])


if __name__ == "__main__":
    unittest.main()
