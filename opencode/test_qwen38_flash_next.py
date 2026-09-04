#!/usr/bin/env python3

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "qwen3.8-flash-next.sh"


class Qwen38FlashNextScriptTest(unittest.TestCase):
    def run_script(self, config_root: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["XDG_CONFIG_HOME"] = str(config_root)
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_replaces_models_and_assigns_agent_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory)
            config_dir = config_root / "opencode"
            config_dir.mkdir()
            config_path = config_dir / "opencode.json"
            config_path.write_text(
                """
                {
                  // Keep this unrelated configuration.
                  "$schema": "https://opencode.ai/config.json",
                  "permission": {"read": "allow"},
                  "provider": {
                    "vllm": {
                      "npm": "@ai-sdk/openai-compatible",
                      "name": "vLLM",
                      "options": {"baseURL": "http://192.168.1.31:54351/v1"},
                      "models": {
                        "old-model": {"name": "Old"},
                        "another-model": {"name": "Another"},
                      },
                    },
                  },
                  "agent": {
                    "general": {
                      "description": "General work",
                      "mode": "subagent",
                      "reasoningEffort": "high",
                      "thinking": {"type": "enabled"},
                    },
                    "scout": {
                      "description": "External research",
                      "mode": "subagent",
                      "reasoningEffort": "low",
                    },
                    "explore": {
                      "description": "Repository exploration",
                      "mode": "subagent",
                      "reasoningEffort": "low",
                      "thinking": {"type": "enabled"},
                    },
                  },
                  "model": "vllm/old-model",
                  "plugin": ["superpowers@example"],
                }
                """,
                encoding="utf-8",
            )
            config_path.chmod(0o640)

            result = self.run_script(config_root)

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["permission"], {"read": "allow"})
            self.assertEqual(config["plugin"], ["superpowers@example"])
            self.assertEqual(config["model"], "vllm/qwen3.8-flash-next")
            self.assertEqual(
                config["provider"]["vllm"]["options"],
                {"baseURL": "http://192.168.1.31:54351/v1"},
            )

            models = config["provider"]["vllm"]["models"]
            self.assertEqual(list(models), ["qwen3.8-flash-next"])
            self.assertEqual(
                models["qwen3.8-flash-next"],
                {
                    "name": "Qwen3.8-Flash-Next",
                    "cost": {"input": 0.15, "output": 0.47},
                    "attachment": True,
                    "reasoning": True,
                    "temperature": True,
                    "tool_call": True,
                    "interleaved": "reasoning_content",
                    "limit": {"context": 262144, "output": 262144},
                    "modalities": {
                        "input": ["text", "image", "video"],
                        "output": ["text"],
                    },
                    "options": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": 0.0,
                        "repetition_penalty": 1.0,
                        "reasoningEffort": "xhigh",
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                            "preserve_thinking": True,
                        },
                    },
                },
            )

            expected_agents = {
                "general": ("General work", 1.0, 0.95, 0.0, "xhigh", True),
                "scout": ("External research", 1.0, 0.95, 0.0, "medium", True),
                "explore": ("Repository exploration", 0.7, 0.80, 1.5, "low", False),
            }
            for name, (description, temperature, top_p, presence, effort, thinking) in expected_agents.items():
                agent = config["agent"][name]
                self.assertEqual(
                    set(agent),
                    {"description", "mode", "temperature", "top_p", "options"},
                )
                self.assertEqual(agent["description"], description)
                self.assertEqual(agent["mode"], "subagent")
                self.assertEqual(agent["temperature"], temperature)
                self.assertEqual(agent["top_p"], top_p)
                self.assertEqual(
                    agent["options"],
                    {
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": 20,
                        "min_p": 0.0,
                        "presence_penalty": presence,
                        "repetition_penalty": 1.0,
                        "reasoningEffort": effort,
                        "chat_template_kwargs": {
                            "enable_thinking": thinking,
                            "preserve_thinking": True,
                        },
                    },
                )

            self.assertEqual(stat.S_IMODE(config_path.stat().st_mode), 0o640)
            backups = list(config_dir.glob("opencode.json.bak.*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("old-model", backups[0].read_text(encoding="utf-8"))

            first_content = config_path.read_bytes()
            first_mtime = config_path.stat().st_mtime_ns
            second = self.run_script(config_root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(config_path.read_bytes(), first_content)
            self.assertEqual(config_path.stat().st_mtime_ns, first_mtime)
            self.assertEqual(len(list(config_dir.glob("opencode.json.bak.*"))), 1)

    def test_missing_config_fails_without_creating_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory)

            result = self.run_script(config_root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not found", result.stderr.lower())
            self.assertFalse((config_root / "opencode").exists())

    def test_creates_missing_agent_profiles_and_preserves_partial_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory)
            config_dir = config_root / "opencode"
            config_dir.mkdir()
            config_path = config_dir / "opencode.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": {
                            "vllm": {
                                "npm": "@ai-sdk/openai-compatible",
                                "options": {"baseURL": "http://localhost:8000/v1"},
                                "models": {},
                            }
                        },
                        "agent": {
                            "general": {
                                "description": "Keep this description",
                                "reasoningEffort": "stale",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(config_root)

            self.assertEqual(result.returncode, 0, result.stderr)
            agents = json.loads(config_path.read_text(encoding="utf-8"))["agent"]
            self.assertEqual(set(agents), {"general", "scout", "explore"})
            self.assertEqual(agents["general"]["description"], "Keep this description")
            for name, effort, thinking in (
                ("general", "xhigh", True),
                ("scout", "medium", True),
                ("explore", "low", False),
            ):
                self.assertEqual(agents[name]["mode"], "subagent")
                self.assertTrue(agents[name]["description"])
                self.assertEqual(agents[name]["options"]["reasoningEffort"], effort)
                self.assertEqual(
                    agents[name]["options"]["chat_template_kwargs"]["enable_thinking"],
                    thinking,
                )

    def test_symlinked_config_is_rejected_without_modifying_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_root = root / "config"
            config_dir = config_root / "opencode"
            config_dir.mkdir(parents=True)
            target = root / "managed-opencode.json"
            original = json.dumps(
                {"provider": {"vllm": {"models": {"old": {"name": "Old"}}}}}
            )
            target.write_text(original, encoding="utf-8")
            (config_dir / "opencode.json").symlink_to(target)

            result = self.run_script(config_root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("symbolic link", result.stderr.lower())
            self.assertTrue((config_dir / "opencode.json").is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), original)
            self.assertEqual(list(config_dir.glob("opencode.json.bak.*")), [])

    def test_invalid_jsonc_does_not_modify_or_back_up_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory)
            config_dir = config_root / "opencode"
            config_dir.mkdir()
            config_path = config_dir / "opencode.json"
            original = b'{"provider": {broken}\n'
            config_path.write_bytes(original)

            result = self.run_script(config_root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid jsonc", result.stderr.lower())
            self.assertEqual(config_path.read_bytes(), original)
            self.assertEqual(list(config_dir.glob("opencode.json.bak.*")), [])


if __name__ == "__main__":
    unittest.main()
