"""
errors.py — The failure vocabulary of the shot module.

WHY THERE IS MORE THAN ONE EXCEPTION
------------------------------------
A missing checkpoint and a malformed frame are different facts about the world,
and a caller that cannot tell them apart will report the wrong one. The four
types below keep that distinction:

    ShotError                 base class; catch this to catch any shot failure
    ShotModelUnavailable      the weights are absent, unreadable, or do not
                              match the architecture / expected checksum
    ShotInputError            the pixels cannot form the model's input tensor
                              (also a ValueError, for ordinary argument errors)
    ShotInferenceError        the model ran and produced something unusable
                              (also a RuntimeError)

`ShotClassifier.classify()` deliberately converts all of them into a recorded
status on the returned `ShotClassification` instead of raising, because one
unclassifiable delivery must not fail an otherwise valid analysis. The types
still matter: they are what the record names, and what a caller loading a
model sees when it asks for one explicitly.
"""

from __future__ import annotations


class ShotError(Exception):
    """Base class for every failure raised by the shot module."""


class ShotModelUnavailable(ShotError):
    """The checkpoint is missing, unreadable, or incompatible with the model."""


class ShotInputError(ShotError, ValueError):
    """The available pixels cannot be turned into the tensor the model expects."""


class ShotInferenceError(ShotError, RuntimeError):
    """The model ran but its output could not be used as a classification."""


__all__ = [
    "ShotError",
    "ShotInferenceError",
    "ShotInputError",
    "ShotModelUnavailable",
]
