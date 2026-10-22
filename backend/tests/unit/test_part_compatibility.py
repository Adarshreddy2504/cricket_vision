"""
test_part_compatibility.py — Does Part 3 speak Part 1's and Part 2's languages?

The analytics pass deliberately imports neither `app.tracking` nor `app.pitch`.
It accepts their records structurally: anything with `.points`, and points with
the documented field names, is enough. That is what lets this branch work
standalone and merge cleanly.

The structural stubs below always run. The classes at the end import the REAL
Part 1 and Part 2 schemas and are skipped on a branch that does not have them,
so they start proving the contract as soon as the branches are merged rather
than being deleted once and forgotten. Part 2's tests used the same pattern for
Part 1.

The subprocess test locks in one structural decision: importing the analytics
package must not drag torch into the process.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from app.analytics import (
    AnalyticsInputError,
    summarise_pitch_trajectory,
    summarise_trajectory,
)
from tests.shots_helpers import (
    PitchPoint,
    PitchTrajectory,
    Point,
    Trajectory,
)

try:  # pragma: no cover - depends on the merge state, not on the code
    from app.schemas.tracking import Trajectory as Part1Trajectory
    from app.schemas.tracking import TrajectoryPoint as Part1Point

    HAVE_PART1 = True
except ImportError:  # pragma: no cover
    Part1Trajectory = None  # type: ignore[assignment,misc]
    Part1Point = None  # type: ignore[assignment,misc]
    HAVE_PART1 = False

try:  # pragma: no cover - depends on the merge state, not on the code
    from app.schemas.pitch import PitchPoint as Part2Point
    from app.schemas.pitch import PitchTrajectory as Part2Trajectory

    HAVE_PART2 = True
except ImportError:  # pragma: no cover
    Part2Point = None  # type: ignore[assignment,misc]
    Part2Trajectory = None  # type: ignore[assignment,misc]
    HAVE_PART2 = False


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class TestStructuralContract(unittest.TestCase):
    """Runs everywhere: proves the protocols, not the other branches' files."""

    def test_part1_shaped_trajectory_is_summarised(self) -> None:
        trajectory = Trajectory(
            [
                Point(1, 0, 0, "detected"),
                Point(2, 3, 4, "detected"),
                Point(3, 6, 8, "detected"),
            ]
        )
        summary = summarise_trajectory(trajectory, fps=10)
        self.assertAlmostEqual(summary.path_length_px or 0.0, 10.0)
        self.assertAlmostEqual(summary.mean_speed_px_s or 0.0, 50.0)

    def test_part2_shaped_trajectory_is_summarised(self) -> None:
        trajectory = PitchTrajectory(
            [PitchPoint(1, 0, 0), PitchPoint(2, 0, 5)],
            units="metre",
        )
        summary = summarise_pitch_trajectory(trajectory, fps=10)
        self.assertAlmostEqual(summary.path_length_m or 0.0, 5.0)
        self.assertAlmostEqual(summary.mean_speed_kmh or 0.0, 180.0)

    def test_a_part2_fraction_document_is_refused(self) -> None:
        trajectory = PitchTrajectory(
            [PitchPoint(1, 0, 0), PitchPoint(2, 0.1, 0.2)],
            units="pitch_fraction",
        )
        with self.assertRaises(AnalyticsInputError):
            summarise_pitch_trajectory(trajectory, fps=10)

    def test_analytics_import_does_not_import_torch(self) -> None:
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(BACKEND_ROOT)!r}); "
            "import app.analytics; "
            "print('torch' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False", completed.stderr)


@unittest.skipUnless(HAVE_PART1, "Part 1 tracking schema is not on this branch")
class TestRealPart1Schema(unittest.TestCase):
    def test_real_trajectory_is_accepted(self) -> None:
        trajectory = Part1Trajectory(
            points=[
                Part1Point(1, 0, 0, "detected"),
                Part1Point(2, 3, 4, "detected"),
                Part1Point(3, 6, 8, "detected"),
            ]
        )
        summary = summarise_trajectory(trajectory, fps=10)

        self.assertEqual(summary.point_count, 3)
        self.assertAlmostEqual(summary.path_length_px or 0.0, 10.0)
        self.assertAlmostEqual(summary.mean_speed_px_s or 0.0, 50.0)
        self.assertEqual(summary.gaps, ())
        self.assertTrue(summary.continuous)

    def test_real_trajectory_gaps_are_preserved(self) -> None:
        trajectory = Part1Trajectory(
            points=[
                Part1Point(1, 0, 0, "detected"),
                Part1Point(2, 3, 4, "detected"),
                Part1Point(5, 3, 4, "predicted"),
            ]
        )
        summary = summarise_trajectory(trajectory, fps=10)

        self.assertEqual(summary.gaps, ((3, 4),))
        self.assertEqual(summary.unobserved_segments, 1)
        self.assertAlmostEqual(summary.path_length_px or 0.0, 5.0)
        self.assertEqual(summary.predicted_count, 1)


@unittest.skipUnless(HAVE_PART2, "Part 2 pitch schema is not on this branch")
class TestRealPart2Schema(unittest.TestCase):
    def _point(self, frame: int, along: float, **kwargs: object) -> object:
        fields = {
            "frame": frame,
            "x": 100 + frame,
            "y": 200,
            "across": 0.0,
            "along": along,
            "space": "metres",
            "units": "metre",
            "source": "detected",
        }
        fields.update(kwargs)
        return Part2Point(**fields)

    def test_real_metre_trajectory_is_accepted(self) -> None:
        trajectory = Part2Trajectory(
            points=[self._point(1, 0.0), self._point(2, 10.0)],
            units="metre",
            space="metres",
        )
        summary = summarise_pitch_trajectory(trajectory, fps=10)

        self.assertEqual(summary.units, "metre")
        self.assertAlmostEqual(summary.path_length_m or 0.0, 10.0)
        self.assertAlmostEqual(summary.mean_speed_mps or 0.0, 100.0)
        self.assertTrue(summary.speed_is_estimate)
        self.assertFalse(summary.speed_is_calibrated)

    def test_real_fraction_trajectory_is_refused(self) -> None:
        trajectory = Part2Trajectory(
            points=[
                self._point(1, 0.0, units="pitch_fraction"),
                self._point(2, 0.4, units="pitch_fraction"),
            ],
            units="pitch_fraction",
        )
        with self.assertRaises(AnalyticsInputError):
            summarise_pitch_trajectory(trajectory, fps=10)

    def test_real_outside_pitch_flag_is_counted(self) -> None:
        trajectory = Part2Trajectory(
            points=[
                self._point(1, 0.0, inside_pitch=True),
                self._point(2, 1.0, inside_pitch=False),
            ],
            units="metre",
            space="metres",
        )
        summary = summarise_pitch_trajectory(trajectory, fps=10)
        self.assertEqual(summary.outside_pitch_count, 1)


if __name__ == "__main__":
    unittest.main()
