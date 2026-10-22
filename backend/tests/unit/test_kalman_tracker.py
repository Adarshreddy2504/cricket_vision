"""
Kalman ball tracker: smoothing, gating, coasting and recovery.

The tracker answers one question — "where is the ball right now" — and it must
answer it truthfully. Every test here is about a case where a tracker could
plausibly lie:

  * A frame with no detection must be reported as an EXTRAPOLATION, never as a
    measurement, so nothing downstream fits a physics curve to an estimate.
  * A detection that lands far from the prediction is almost always a false
    positive; it must be refused rather than snapped to.
  * A refused measurement is evidence AGAINST the current track. It must count
    towards the track dying, otherwise a stream of detections belonging to the
    NEXT ball keeps the previous one alive forever.
  * Once the track is lost, a new detection must be able to re-acquire the
    ball. Refusing it against a prediction the filter has already disowned is
    a livelock, not a rejection policy.
"""

from __future__ import annotations

import unittest

from app.core.constants import GATE_RADIUS, MAX_MISSED, MAX_TRAIL
from app.tracking.kalman_tracker import BallTracker


def drive_misses(tracker: BallTracker, start_frame: int, count: int):
    """Feed `count` empty frames, returning the last result."""
    result = None
    for i in range(count):
        result = tracker.update(None, start_frame + i)
    return result


class TestTrackerProvenance(unittest.TestCase):
    """What the tracker says about the origin of its answer."""

    def test_detection_yields_detected_source(self):
        tracker = BallTracker()
        result = tracker.update((100, 100), 1)
        self.assertEqual(result.source, "detected")
        self.assertEqual(result.position, (100, 100))

    def test_missed_frame_yields_predicted_source(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        result = tracker.update(None, 2)
        self.assertEqual(result.source, "predicted")
        self.assertIsNotNone(result.position)

    def test_a_lost_track_reports_no_position_and_no_source(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        result = drive_misses(tracker, 2, MAX_MISSED + 1)
        self.assertTrue(tracker.is_lost)
        self.assertIsNone(result.position)
        self.assertIsNone(result.source)

    def test_the_tracker_still_predicts_inside_the_coast_window(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        result = drive_misses(tracker, 2, MAX_MISSED)
        self.assertFalse(tracker.is_lost)
        self.assertEqual(result.source, "predicted")


class TestAssociationGate(unittest.TestCase):
    """A detection far from the prediction is refused."""

    def test_a_distant_detection_is_refused(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        result = tracker.update((1000, 1000), 2)
        self.assertEqual(result.source, "predicted")
        self.assertTrue(result.gated_out)

    def test_a_refused_detection_does_not_move_the_ball(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        result = tracker.update((1000, 1000), 2)
        self.assertLess(abs(result.position[0] - 100), GATE_RADIUS)

    def test_a_detection_inside_the_gate_is_associated(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        result = tracker.update((110, 104), 2)
        self.assertEqual(result.source, "detected")
        self.assertFalse(result.gated_out)

    def test_a_refusal_counts_towards_the_track_dying(self):
        """A refused measurement is evidence against the track, not for it."""
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        for i in range(2, 2 + MAX_MISSED + 1):
            result = tracker.update((100 + GATE_RADIUS + 200, 100), i)
            self.assertTrue(result.gated_out)
        self.assertTrue(tracker.is_lost)

    def test_the_gate_radius_is_not_relaxed(self):
        """Guard against the gate being 'fixed' by widening it."""
        from app.tracking import kalman_tracker as tracker_mod

        self.assertEqual(tracker_mod.GATE_RADIUS_PX, GATE_RADIUS)
        self.assertEqual(tracker_mod.GATE_RADIUS_PX, 120)

    def test_the_gate_is_not_applied_while_the_track_is_lost(self):
        """
        Refusing a detection against a prediction the filter has disowned is a
        livelock: the refusal advances the miss counter, so the track can never
        come back.
        """
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        drive_misses(tracker, 2, MAX_MISSED + 2)
        self.assertTrue(tracker.is_lost)

        result = tracker.update((900, 600), 100)
        self.assertEqual(result.source, "detected")
        self.assertFalse(result.gated_out)
        self.assertEqual(result.position, (900, 600))


class TestRecovery(unittest.TestCase):
    """A lost track must be able to re-acquire the ball."""

    def test_a_lost_track_reinitialises_on_a_new_detection(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        drive_misses(tracker, 2, MAX_MISSED + 2)
        result = tracker.update((900, 600), 100)
        self.assertEqual(result.source, "detected")
        self.assertEqual(result.position, (900, 600))
        self.assertEqual(tracker.missed_frames, 0)
        self.assertFalse(tracker.is_lost)

    def test_recovery_is_immediate_not_deferred(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        drive_misses(tracker, 2, MAX_MISSED + 1)
        tracker.update((900, 600), 100)
        second = tracker.update((904, 602), 101)
        self.assertEqual(second.source, "detected")
        self.assertEqual(tracker.missed_frames, 0)

    def test_recovery_does_not_resume_the_stale_ghost(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        drive_misses(tracker, 2, MAX_MISSED + 2)
        tracker.update((900, 600), 100)
        result = tracker.update((905, 603), 101)
        self.assertEqual(result.source, "detected")
        self.assertGreater(abs(result.position[0] - 100), GATE_RADIUS)

    def test_a_spurious_seed_cannot_become_a_track_on_its_own(self):
        """One false positive after a loss must not become a confident track."""
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        drive_misses(tracker, 2, MAX_MISSED + 2)
        tracker.update((900, 600), 100)
        drive_misses(tracker, 101, MAX_MISSED + 2)
        self.assertTrue(tracker.is_lost)

    def test_repeated_loss_and_recovery_cycles(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        for cycle, x in enumerate((900, 400, 1200, 250)):
            drive_misses(tracker, 2 + cycle * 50, MAX_MISSED + 2)
            result = tracker.update((x, 300), 3 + cycle * 50)
            self.assertEqual(result.source, "detected")
            self.assertEqual(tracker.missed_frames, 0)


class TestTrackerState(unittest.TestCase):
    """State hygiene."""

    def test_pure_silence_invents_no_track(self):
        tracker = BallTracker()
        drive_misses(tracker, 1, 30)
        self.assertTrue(tracker.is_lost)
        self.assertEqual(list(tracker.traj_points), [])

    def test_the_trail_is_bounded(self):
        tracker = BallTracker()
        for i in range(1, MAX_TRAIL + 50):
            tracker.update((100 + i, 100), i)
        self.assertLessEqual(len(tracker.trajectory), MAX_TRAIL)

    def test_reset_clears_the_filter_and_the_trail(self):
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        tracker.reset()
        self.assertFalse(tracker.kf.initialized)
        self.assertEqual(tracker.missed_frames, 0)
        self.assertEqual(list(tracker.traj_points), [])

    def test_reseed_reacquires_the_ball_without_a_gap(self):
        """`reseed` restarts the filter on a ball that is actually visible."""
        tracker = BallTracker()
        tracker.update((100, 100), 1)
        drive_misses(tracker, 2, MAX_MISSED + 2)
        self.assertTrue(tracker.is_lost)

        result = tracker.reseed((500, 500), 100)
        self.assertEqual(result.source, "detected")
        self.assertEqual(result.position, (500, 500))
        self.assertEqual(tracker.missed_frames, 0)
        self.assertFalse(tracker.is_lost)


class TestTrackerConfiguration(unittest.TestCase):
    """The coast window and the gate radius are configurable."""

    def test_a_custom_coast_window_is_respected(self):
        tracker = BallTracker(max_missed_frames=2)
        tracker.update((100, 100), 1)
        result = drive_misses(tracker, 2, 3)
        self.assertTrue(tracker.is_lost)
        self.assertIsNone(result.position)

    def test_a_custom_gate_radius_is_respected(self):
        tracker = BallTracker(gate_radius_px=10)
        tracker.update((100, 100), 1)
        result = tracker.update((140, 100), 2)
        self.assertTrue(result.gated_out)
        self.assertEqual(result.source, "predicted")


if __name__ == "__main__":
    unittest.main()
