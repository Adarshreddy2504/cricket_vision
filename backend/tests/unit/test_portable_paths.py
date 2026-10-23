"""Environment paths resolve the same way from any working directory.

A relative path in `.env` is written relative to the repository, not to
whatever directory the server happened to be started from. These tests pin that
rule for the three places that accept a path from the environment.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.api.config import PROJECT_DIR, Settings
from app.core.config import _env_file as tracking_env_file
from app.pitch.config import _env_file as pitch_env_file


class TestRelativeModelPaths(unittest.TestCase):
    def test_ball_model_relative_path_is_project_relative(self):
        with patch.dict(
            os.environ,
            {"CRICKET_BALL_MODEL": "backend/models/custom.pt"},
            clear=False,
        ):
            path = tracking_env_file(
                "CRICKET_BALL_MODEL",
                Path("default.pt"),
            )
        self.assertEqual(path, (PROJECT_DIR / "backend/models/custom.pt").resolve())

    def test_pitch_model_relative_path_is_project_relative(self):
        with patch.dict(
            os.environ,
            {"CRICKET_PITCH_BOX_MODEL": "weights/pitch.pt"},
            clear=False,
        ):
            path = pitch_env_file(
                "CRICKET_PITCH_BOX_MODEL",
                Path("default.pt"),
            )
        self.assertEqual(path, (PROJECT_DIR / "weights/pitch.pt").resolve())

    def test_absolute_model_path_is_left_alone(self):
        absolute = (PROJECT_DIR / "somewhere" / "ball.pt").resolve()
        with patch.dict(
            os.environ,
            {"CRICKET_BALL_MODEL": str(absolute)},
            clear=False,
        ):
            path = tracking_env_file(
                "CRICKET_BALL_MODEL",
                Path("default.pt"),
            )
        self.assertEqual(path, absolute)


class TestRelativeDataDir(unittest.TestCase):
    def test_data_dir_from_environment_is_project_relative(self):
        with patch.dict(
            os.environ,
            {"CRICKET_DATA_DIR": "var/cricketvision"},
            clear=False,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.data_dir, (PROJECT_DIR / "var/cricketvision").resolve())

    def test_absolute_data_dir_is_left_alone(self):
        absolute = (PROJECT_DIR / "runtime").resolve()
        with patch.dict(
            os.environ,
            {"CRICKET_DATA_DIR": str(absolute)},
            clear=False,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.data_dir, absolute)


if __name__ == "__main__":
    unittest.main()
