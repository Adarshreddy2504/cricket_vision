"""
test_pitch_adapter.py — Part 1's ball trajectory, re-expressed in pitch
coordinates.

Part 1 lives on its own branch, so most of these tests run against a
stand-in whose fields are exactly Part 1's. The last class runs against the
REAL `app.schemas.tracking` dataclasses whenever they are importable, which
is the evidence that matters after the two branches are merged; it skips on
a tree where Part 1 is absent.
"""

from __future__ import annotations

import unittest

from app.pitch.adapter import (
    TrajectoryLike,
    TrajectoryPointLike,
    project_points,
    project_trajectory,
)
from app.pitch.calibration import CornerMethod, detected_calibration, manual_calibration
from app.pitch.corners import order_corners
from app.pitch.errors import CalibrationError
from app.pitch.homography import PitchSpace, build_projection
from tests.pitch_helpers import (
    FRAME_SIZE,
    StubTrajectory,
    StubTrajectoryPoint,
    rectangle,
    sample_trajectory,
    trapezoid,
)

try:  # Part 1 is on its own branch; this import succeeds once it is merged in.
    from app.schemas.tracking import Trajectory, TrajectoryPoint

    HAVE_PART1 = True
except ImportError:  # pragma: no cover - the branch without Part 1
    Trajectory = None
    TrajectoryPoint = None
    HAVE_PART1 = False


def manual_rectangle():
    return manual_calibration(rectangle(), image_size=FRAME_SIZE)


def normalized_projection():
    projection, reason = build_projection(manual_rectangle(), space=PitchSpace.NORMALIZED)
    if projection is None:
        raise AssertionError(f"no projection: {reason}")
    return projection


class TestStructuralCompatibility(unittest.TestCase):
    def test_a_part1_shaped_point_satisfies_the_protocol(self):
        point = StubTrajectoryPoint(1, 120, 120, "detected", 0.9, 7)
        self.assertIsInstance(point, TrajectoryPointLike)

    def test_a_part1_shaped_trajectory_satisfies_the_protocol(self):
        self.assertIsInstance(sample_trajectory(), TrajectoryLike)

    def test_an_arbitrary_object_does_not(self):
        self.assertNotIsInstance(object(), TrajectoryPointLike)
        self.assertNotIsInstance(object(), TrajectoryLike)

    def test_the_adapter_imports_nothing_from_tracking(self):
        # A structural adapter is what lets Part 2 run on its own branch and
        # merge without an import-order puzzle; importing Part 1 here would
        # undo that, so the absence is checked rather than trusted.
        import ast
        import inspect

        import app.pitch.adapter as adapter

        imported: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(adapter))):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        suspects = [
            name
            for name in imported
            if name.startswith(("app.tracking", "app.schemas.tracking"))
        ]
        self.assertEqual(suspects, [])


class TestProjection(unittest.TestCase):
    def test_a_trajectory_object_projects(self):
        result = project_trajectory(sample_trajectory(), manual_rectangle())
        self.assertEqual(result.space, "normalized")
        self.assertEqual(result.units, "pitch_fraction")
        self.assertEqual(result.point_count, 4)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.skipped_frames, [])

    def test_a_bare_sequence_of_points_projects_identically(self):
        trajectory = sample_trajectory()
        from_object = project_trajectory(trajectory, manual_rectangle())
        from_sequence = project_trajectory(trajectory.points, manual_rectangle())
        self.assertEqual(from_object.points, from_sequence.points)

    def test_known_pixels_map_to_known_pitch_coordinates(self):
        # The rectangle's homography is affine, so this is hand arithmetic.
        trajectory = StubTrajectory(
            [
                StubTrajectoryPoint(1, 200, 200, "detected", 0.9),
                StubTrajectoryPoint(2, 250, 150, "detected", 0.8),
            ]
        )
        result = project_trajectory(trajectory, manual_rectangle())
        self.assertAlmostEqual(result.points[0].across, 0.5, places=5)
        self.assertAlmostEqual(result.points[0].along, 0.5, places=5)
        self.assertAlmostEqual(result.points[1].across, 0.75, places=5)
        self.assertAlmostEqual(result.points[1].along, 0.25, places=5)

    def test_metres_are_available_when_the_corners_were_measured(self):
        result = project_trajectory(
            sample_trajectory(), manual_rectangle(), space=PitchSpace.METRES
        )
        self.assertEqual(result.space, "metres")
        self.assertEqual(result.units, "metre")
        first = result.points[0]
        # (120, 120) is 10% of the way across and 10% of the way along.
        self.assertAlmostEqual(first.across, -1.056, places=4)
        self.assertAlmostEqual(first.along, 2.012, places=4)

    def test_the_calibration_travels_with_the_points(self):
        result = project_trajectory(sample_trajectory(), manual_rectangle())
        self.assertEqual(result.calibration["status"], "valid")
        self.assertEqual(result.calibration["provenance"], "manually_supplied")
        self.assertEqual(result.calibration["image_size"], [640, 360])


class TestFieldPreservation(unittest.TestCase):
    def test_every_part1_field_survives_the_conversion(self):
        source = sample_trajectory()
        result = project_trajectory(source, manual_rectangle())
        self.assertEqual(len(result.points), len(source.points))
        for original, mapped in zip(source.points, result.points, strict=True):
            with self.subTest(frame=original.frame):
                self.assertEqual(mapped.frame, original.frame)
                self.assertEqual(mapped.x, original.x)
                self.assertEqual(mapped.y, original.y)
                self.assertEqual(mapped.source, original.source)
                self.assertEqual(mapped.confidence, original.confidence)
                self.assertEqual(mapped.track_id, original.track_id)

    def test_a_predicted_point_stays_predicted(self):
        result = project_trajectory(sample_trajectory(), manual_rectangle())
        predicted = [p for p in result.points if p.frame == 3]
        self.assertEqual(len(predicted), 1)
        self.assertEqual(predicted[0].source, "predicted")
        self.assertIsNone(predicted[0].confidence)
        self.assertEqual(result.predicted_count, 1)
        self.assertEqual(result.detected_count, 3)

    def test_a_ball_wide_of_the_strip_is_reported_outside_it(self):
        result = project_trajectory(sample_trajectory(), manual_rectangle())
        wide = [p for p in result.points if p.frame == 4]
        self.assertEqual(len(wide), 1)
        self.assertFalse(wide[0].inside_pitch)
        self.assertGreater(wide[0].across, 1.0)
        self.assertEqual(result.inside_count, 3)


class TestNoInvention(unittest.TestCase):
    def test_an_unprojectable_point_is_counted_and_named(self):
        # y = -900 is the trapezoid's horizon: the pixel has no finite image.
        trajectory = StubTrajectory(
            [
                StubTrajectoryPoint(1, 200, 200, "detected", 0.9),
                StubTrajectoryPoint(2, 0, -900, "detected", 0.9),
                StubTrajectoryPoint(3, 240, 240, "detected", 0.9),
            ]
        )
        result = project_trajectory(
            trajectory, manual_calibration(trapezoid(), image_size=FRAME_SIZE)
        )
        self.assertEqual(result.point_count, 2)
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.skipped_frames, [2])
        self.assertEqual([p.frame for p in result.points], [1, 3])

    def test_a_gap_is_named_rather_than_bridged(self):
        trajectory = StubTrajectory(
            [
                StubTrajectoryPoint(1, 200, 200, "detected", 0.9),
                StubTrajectoryPoint(4, 240, 240, "detected", 0.9),
            ]
        )
        result = project_trajectory(trajectory, manual_rectangle())
        self.assertEqual(result.point_count, 2)
        self.assertEqual(result.gaps, [[2, 3]])
        self.assertEqual(result.continuity, "gapped")

    def test_no_point_is_added_for_a_frame_that_had_none(self):
        result = project_trajectory(sample_trajectory(), manual_rectangle())
        self.assertEqual([p.frame for p in result.points], [1, 2, 3, 4])
        self.assertEqual(result.gaps, [])
        self.assertEqual(result.continuity, "continuous")


class TestRefusals(unittest.TestCase):
    def test_metres_are_refused_when_the_calibration_cannot_justify_them(self):
        detected = detected_calibration(
            rectangle(),
            image_size=FRAME_SIZE,
            method=CornerMethod.SEGMENTATION,
            confidence=0.95,
        )
        with self.assertRaises(CalibrationError) as ctx:
            project_trajectory(sample_trajectory(), detected, space=PitchSpace.METRES)
        self.assertIn("metric_geometry_not_justified", str(ctx.exception))

    def test_an_invalid_calibration_projects_nothing(self):
        off_picture = order_corners(
            [(-300.0, 100.0), (-100.0, 100.0), (-80.0, 300.0), (-320.0, 300.0)]
        )
        with self.assertRaises(CalibrationError) as ctx:
            project_trajectory(
                sample_trajectory(), manual_calibration(off_picture, image_size=FRAME_SIZE)
            )
        self.assertIn("invalid_calibration", str(ctx.exception))

    def test_an_object_without_points_is_refused(self):
        with self.assertRaises(TypeError) as ctx:
            project_trajectory(object(), manual_rectangle())
        self.assertIn("points", str(ctx.exception))

    def test_a_point_without_a_source_is_refused(self):
        class NoSource:
            frame = 1
            x = 100
            y = 100

        with self.assertRaises(TypeError) as ctx:
            project_points([NoSource()], normalized_projection())
        self.assertIn("source", str(ctx.exception))

    def test_a_foreign_source_vocabulary_is_refused(self):
        point = StubTrajectoryPoint(1, 100, 100, "guessed")
        with self.assertRaises(TypeError) as ctx:
            project_points([point], normalized_projection())
        self.assertIn("detected", str(ctx.exception))

    def test_a_missing_coordinate_is_refused(self):
        class NoY:
            frame = 1
            x = 100
            source = "detected"

        with self.assertRaises(TypeError) as ctx:
            project_points([NoY()], normalized_projection())
        self.assertIn("y", str(ctx.exception))

    def test_a_non_numeric_coordinate_is_refused(self):
        point = StubTrajectoryPoint(1, 100, 100, "detected")
        point.x = "left"
        with self.assertRaises(TypeError) as ctx:
            project_points([point], normalized_projection())
        self.assertIn("not an integer", str(ctx.exception))


class TestPersistedForm(unittest.TestCase):
    def test_the_document_states_its_space_and_units(self):
        document = project_trajectory(sample_trajectory(), manual_rectangle()).as_dict()
        self.assertEqual(document["space"], "normalized")
        self.assertEqual(document["units"], "pitch_fraction")
        self.assertEqual(document["point_count"], 4)
        self.assertEqual(document["detected_count"], 3)
        self.assertEqual(document["predicted_count"], 1)
        self.assertEqual(document["inside_count"], 3)
        self.assertEqual(document["continuity"], "continuous")
        self.assertEqual(document["gaps"], [])
        self.assertEqual(document["skipped"], 0)
        self.assertEqual(document["calibration"]["status"], "valid")

    def test_a_point_document_is_self_describing(self):
        document = project_trajectory(sample_trajectory(), manual_rectangle()).as_dict()
        first = document["points"][0]
        self.assertEqual(
            set(first),
            {
                "frame",
                "x",
                "y",
                "across",
                "along",
                "space",
                "units",
                "source",
                "confidence",
                "track_id",
                "inside_pitch",
            },
        )
        self.assertEqual(first["space"], "normalized")
        self.assertEqual(first["units"], "pitch_fraction")
        self.assertEqual(first["source"], "detected")
        self.assertEqual(first["confidence"], 0.91)
        self.assertEqual(first["track_id"], 7)


@unittest.skipUnless(HAVE_PART1, "Part 1's tracking schema is not on this branch")
class TestRealPart1Schema(unittest.TestCase):
    def test_the_real_trajectory_and_points_project(self):
        source = Trajectory(
            points=[
                TrajectoryPoint(
                    frame=1, x=200, y=200, source="detected", confidence=0.9, track_id=3
                ),
                TrajectoryPoint(frame=2, x=250, y=150, source="detected", track_id=3),
                TrajectoryPoint(frame=3, x=260, y=160, source="predicted", track_id=3),
            ]
        )
        result = project_trajectory(source, manual_rectangle())
        self.assertEqual(result.point_count, 3)
        self.assertAlmostEqual(result.points[0].across, 0.5, places=5)
        self.assertAlmostEqual(result.points[0].along, 0.5, places=5)
        self.assertEqual(result.points[2].source, "predicted")
        self.assertIsNone(result.points[1].confidence)
        self.assertEqual(result.gaps, [])

    def test_the_real_dataclasses_satisfy_the_protocols(self):
        point = TrajectoryPoint(frame=1, x=1, y=2, source="detected")
        self.assertIsInstance(point, TrajectoryPointLike)
        self.assertIsInstance(Trajectory(points=[point]), TrajectoryLike)


if __name__ == "__main__":
    unittest.main()
