#!/usr/bin/env python3

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "auto.sh"


@contextmanager
def model_server(payload, status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                }
            )
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            body = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class AutoScriptTest(unittest.TestCase):
    def make_config(self, config_root: Path) -> Path:
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
                            "models": {"old": {"name": "Old"}},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def run_auto(self, config_root: Path, base_url: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "VLLM_API_KEY": "test-key",
                "VLLM_BASE_URL": base_url,
                "XDG_CONFIG_HOME": str(config_root),
            }
        )
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_runs_exact_script_for_first_returned_model_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory)
            config_path = self.make_config(config_root)
            payload = json.dumps(
                {
                    "data": [
                        {"id": "glm-5.3-flash"},
                        {"id": "deepseek-v4-flash-vision-exp"},
                    ]
                }
            )
            with model_server(payload) as (base_url, requests):
                result = self.run_auto(config_root, base_url)

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["model"], "vllm/glm-5.3-flash")
            self.assertEqual(
                list(config["provider"]["vllm"]["models"]),
                ["glm-5.3-flash"],
            )
            self.assertIn("glm-5.3-flash.sh", result.stdout)
            self.assertEqual(
                requests,
                [{"path": "/v1/models", "authorization": "Bearer test-key"}],
            )

    def test_does_not_normalize_first_model_or_fall_back_to_later_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory)
            config_path = self.make_config(config_root)
            original = config_path.read_bytes()
            payload = json.dumps(
                {
                    "data": [
                        {"id": "zai-org/GLM-5.3-Flash"},
                        {"id": "deepseek-v4-flash-vision-exp"},
                    ]
                }
            )
            with model_server(payload) as (base_url, _requests):
                result = self.run_auto(config_root, base_url)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("First model id is not a safe script name", result.stderr)
            self.assertEqual(config_path.read_bytes(), original)
            self.assertEqual(list(config_path.parent.glob("opencode.json.bak.*")), [])

    def test_fails_without_modifying_config_for_empty_or_failed_query(self) -> None:
        for payload, status, expected_error in (
            ('{"data": []}', 200, "Failed to read the first model"),
            ('{"data": [}', 200, "Failed to read the first model"),
            ('{"error": "unavailable"}', 503, "Failed to fetch"),
        ):
            with self.subTest(status=status, payload=payload), tempfile.TemporaryDirectory() as directory:
                config_root = Path(directory)
                config_path = self.make_config(config_root)
                original = config_path.read_bytes()
                with model_server(payload, status=status) as (base_url, _requests):
                    result = self.run_auto(config_root, base_url)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self.assertEqual(config_path.read_bytes(), original)
                self.assertEqual(list(config_path.parent.glob("opencode.json.bak.*")), [])

    def test_rejects_unsafe_or_non_exact_first_model_ids(self) -> None:
        cases = (
            "GLM-5.3-Flash",
            "missing-model",
            "../glm-5.3-flash",
            "auto",
            "glm-5.3-flash\n",
            None,
        )
        for model_id in cases:
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as directory:
                config_root = Path(directory)
                config_path = self.make_config(config_root)
                original = config_path.read_bytes()
                first = {} if model_id is None else {"id": model_id}
                payload = json.dumps(
                    {"data": [first, {"id": "deepseek-v4-flash-vision-exp"}]}
                )
                with model_server(payload) as (base_url, _requests):
                    result = self.run_auto(config_root, base_url)

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(config_path.read_bytes(), original)
                self.assertEqual(list(config_path.parent.glob("opencode.json.bak.*")), [])

    def test_rejects_symlinked_model_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner_dir = root / "runner"
            runner_dir.mkdir()
            copied_auto = runner_dir / "auto.sh"
            shutil.copy2(SCRIPT, copied_auto)
            outside_script = root / "outside.sh"
            marker = root / "executed"
            outside_script.write_text(
                f"#!/usr/bin/env bash\ntouch {marker}\n",
                encoding="utf-8",
            )
            outside_script.chmod(0o755)
            (runner_dir / "linked.sh").symlink_to(outside_script)
            payload = json.dumps({"data": [{"id": "linked"}]})
            with model_server(payload) as (base_url, _requests):
                env = os.environ.copy()
                env.update(
                    {
                        "VLLM_API_KEY": "test-key",
                        "VLLM_BASE_URL": base_url,
                        "XDG_CONFIG_HOME": str(root / "config"),
                    }
                )
                result = subprocess.run(
                    ["bash", str(copied_auto)],
                    cwd=runner_dir,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(marker.exists())

    def test_requires_api_key_before_querying(self) -> None:
        env = os.environ.copy()
        env.pop("VLLM_API_KEY", None)
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VLLM_API_KEY is not set", result.stderr)


if __name__ == "__main__":
    unittest.main()
