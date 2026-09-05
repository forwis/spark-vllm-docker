#!/usr/bin/env python3
"""Focused tests for run-recipe container environment resolution."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "run_recipe", PROJECT_DIR / "run-recipe.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_RECIPE)


class DetectSystemTimezoneTests(unittest.TestCase):
    def test_prefers_tz_environment_variable(self) -> None:
        self.assertEqual(
            RUN_RECIPE.detect_system_timezone(
                {"TZ": "Pacific/Honolulu"},
                Path("/missing/timezone"),
                Path("/missing/localtime"),
            ),
            "Pacific/Honolulu",
        )

    def test_reads_etc_timezone_when_environment_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            timezone_file = Path(temp_dir) / "timezone"
            timezone_file.write_text("Europe/Paris\n", encoding="utf-8")

            self.assertEqual(
                RUN_RECIPE.detect_system_timezone(
                    {}, timezone_file, Path(temp_dir) / "missing-localtime"
                ),
                "Europe/Paris",
            )

    def test_derives_timezone_from_localtime_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zoneinfo = Path(temp_dir) / "zoneinfo" / "America" / "Los_Angeles"
            zoneinfo.parent.mkdir(parents=True)
            zoneinfo.write_bytes(b"")
            localtime = Path(temp_dir) / "localtime"
            localtime.symlink_to(zoneinfo)

            self.assertEqual(
                RUN_RECIPE.detect_system_timezone(
                    {}, Path(temp_dir) / "missing-timezone", localtime
                ),
                "America/Los_Angeles",
            )

    def test_ignores_non_utf8_etc_timezone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            timezone_file = Path(temp_dir) / "timezone"
            timezone_file.write_bytes(b"\xff\xfe")

            self.assertEqual(
                RUN_RECIPE.detect_system_timezone(
                    {}, timezone_file, Path(temp_dir) / "missing-localtime"
                ),
                "Asia/Seoul",
            )

    def test_falls_back_to_asia_seoul(self) -> None:
        self.assertEqual(
            RUN_RECIPE.detect_system_timezone(
                {}, Path("/missing/timezone"), Path("/missing/localtime")
            ),
            "Asia/Seoul",
        )


if __name__ == "__main__":
    unittest.main()
