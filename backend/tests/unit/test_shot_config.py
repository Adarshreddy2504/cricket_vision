"""
test_shot_config.py — Where the checkpoint is looked for, and how the loader is
configured.

Every test here runs against a temporary directory or a patched project root, so
nothing depends on whether the real 68 MB checkpoint happens to be installed.
The paths in the assertions are the paths the resolver reports, which is the
behaviour that matters.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from app.shots import config
from app.shots.config import (
    ENV_MODELS_DIR,
    ENV_SHOT_DEVICE,
    ENV_SHOT_MODEL,
    ENV_SHOT_MODEL_MD5,
    ShotConfig,
    describe_weights,
    resolve_models_dir,
    resolve_shot_model_path,
)
from app.shots.constants import SHOT_MODEL_FILENAME


@contextmanager
def clean_env(**values: str):
    """Override every shot-related variable, treating "" as unset."""
    overrides = {
        ENV_MODELS_DIR: "",
        ENV_SHOT_MODEL: "",
        ENV_SHOT_MODEL_MD5: "",
        ENV_SHOT_DEVICE: "",
    }
    overrides.update(values)
    with patch.dict(os.environ, overrides, clear=False):
        yield


class TemporaryDirectoryCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def make_checkpoint(self, name: str = SHOT_MODEL_FILENAME) -> Path:
        path = self.directory / name
        path.write_bytes(b"checkpoint fixture")
        return path


class TestResolveModelsDir(TemporaryDirectoryCase):
    def test_argument_wins_over_environment(self) -> None:
        other = self.directory / "other"
        with clean_env(**{ENV_MODELS_DIR: str(other)}):
            self.assertEqual(resolve_models_dir(self.directory), self.directory)

    def test_environment_directory_is_used(self) -> None:
        with clean_env(**{ENV_MODELS_DIR: str(self.directory)}):
            self.assertEqual(resolve_models_dir(), self.directory)

    def test_default_directory_when_nothing_is_set(self) -> None:
        with clean_env():
            self.assertEqual(resolve_models_dir(), config.DEFAULT_MODELS_DIR)

    def test_relative_environment_directory_uses_project_root(self) -> None:
        with clean_env(**{ENV_MODELS_DIR: "weights"}):
            with patch.object(config, "PROJECT_DIR", self.directory):
                self.assertEqual(resolve_models_dir(), self.directory / "weights")

    def test_blank_environment_is_ignored(self) -> None:
        with clean_env(**{ENV_MODELS_DIR: "   "}):
            self.assertEqual(resolve_models_dir(), config.DEFAULT_MODELS_DIR)


class TestResolveShotModelPath(TemporaryDirectoryCase):
    def test_default_file_is_found_in_models_dir(self) -> None:
        checkpoint = self.make_checkpoint()
        with clean_env():
            self.assertEqual(
                resolve_shot_model_path(models_dir=self.directory),
                checkpoint,
            )

    def test_explicit_path_wins(self) -> None:
        explicit = self.make_checkpoint("chosen.ckpt")
        self.make_checkpoint()
        with clean_env():
            self.assertEqual(
                resolve_shot_model_path(explicit, models_dir=self.directory),
                explicit,
            )

    def test_environment_path_is_used(self) -> None:
        env_checkpoint = self.make_checkpoint("from_env.ckpt")
        with clean_env(**{ENV_SHOT_MODEL: str(env_checkpoint)}):
            self.assertEqual(resolve_shot_model_path(), env_checkpoint)

    def test_relative_environment_path_uses_project_root(self) -> None:
        relative = self.directory / "sub" / "model.ckpt"
        relative.parent.mkdir()
        relative.write_bytes(b"bytes")
        with clean_env(**{ENV_SHOT_MODEL: "sub/model.ckpt"}):
            with patch.object(config, "PROJECT_DIR", self.directory):
                self.assertEqual(resolve_shot_model_path(), relative)

    def test_missing_explicit_path_names_it_and_raises(self) -> None:
        missing = self.directory / "absent.ckpt"
        with clean_env():
            with self.assertRaises(FileNotFoundError) as caught:
                resolve_shot_model_path(missing)
        message = str(caught.exception)
        self.assertIn(str(missing), message)
        self.assertIn("CRICKET_SHOT_MODEL", message)

    def test_missing_environment_path_reports_env_and_default(self) -> None:
        missing = self.directory / "absent.ckpt"
        with clean_env(**{ENV_SHOT_MODEL: str(missing)}):
            with self.assertRaises(FileNotFoundError) as caught:
                resolve_shot_model_path(models_dir=self.directory)
        message = str(caught.exception)
        self.assertIn(str(missing), message)
        self.assertIn(str(self.directory / SHOT_MODEL_FILENAME), message)

    def test_no_file_anywhere_raises(self) -> None:
        with clean_env():
            with self.assertRaises(FileNotFoundError):
                resolve_shot_model_path(models_dir=self.directory)


class TestShotConfig(TemporaryDirectoryCase):
    def test_from_env_reads_all_four_variables(self) -> None:
        with clean_env(
            **{
                ENV_SHOT_MODEL: "models/custom.ckpt",
                ENV_SHOT_DEVICE: "cpu",
                ENV_SHOT_MODEL_MD5: "ABCDEF",
            }
        ):
            settings = ShotConfig.from_env()
        self.assertEqual(settings.model_path, Path("models/custom.ckpt"))
        self.assertEqual(settings.device, "cpu")
        self.assertEqual(settings.expected_md5, "ABCDEF")

    def test_from_env_defaults(self) -> None:
        with clean_env():
            settings = ShotConfig.from_env()
        self.assertIsNone(settings.model_path)
        self.assertEqual(settings.device, "auto")
        self.assertIsNone(settings.expected_md5)

    def test_from_env_is_not_frozen_at_import_time(self) -> None:
        with clean_env():
            first = ShotConfig.from_env()
        with clean_env(**{ENV_SHOT_DEVICE: "cpu"}):
            second = ShotConfig.from_env()
        self.assertEqual(first.device, "auto")
        self.assertEqual(second.device, "cpu")

    def test_metadata_reports_a_missing_checkpoint_without_raising(self) -> None:
        methods = ShotConfig(model_path=self.directory / "missing.ckpt", device="cpu")
        metadata = methods.as_metadata()
        self.assertFalse(metadata["model_available"])
        self.assertIn("model_error", metadata)
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["model_filename"], SHOT_MODEL_FILENAME)

    def test_metadata_reports_size_for_a_present_checkpoint(self) -> None:
        checkpoint = self.make_checkpoint()
        metadata = ShotConfig(model_path=checkpoint, device="cpu").as_metadata()
        self.assertTrue(metadata["model_available"])
        self.assertEqual(metadata["model_path"], str(checkpoint))
        self.assertEqual(metadata["model_size_bytes"], checkpoint.stat().st_size)

    def test_resolve_model_path_honours_models_dir(self) -> None:
        checkpoint = self.make_checkpoint()
        with clean_env():
            resolved = ShotConfig().resolve_model_path(models_dir=self.directory)
        self.assertEqual(resolved, checkpoint)


class TestDescribeWeights(TemporaryDirectoryCase):
    def test_missing_file_says_not_found(self) -> None:
        description = describe_weights(self.directory / "missing.ckpt")
        self.assertIn("NOT FOUND", description)

    def test_present_file_reports_size(self) -> None:
        checkpoint = self.make_checkpoint()
        description = describe_weights(checkpoint)
        self.assertIn(str(checkpoint), description)
        self.assertIn("MiB", description)


if __name__ == "__main__":
    unittest.main()
