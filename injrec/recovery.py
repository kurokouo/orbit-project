"""deciding whether an injected signal came back or not
"""

from __future__ import annotations

from enum import Enum


class Match(Enum):
    """Outcome of comparing a recovered period to an injected one."""

    EXACT = "perfect"
    HARMONIC = "harmonic"
    MISSED = "fail"


def classify_recovery(
    recovered_period: float,
    true_period: float,
    tolerance: float = 0.01,
    max_harmonic: int = 3,
) -> Match:
    """Classify a BLS detection against the period that was injected."""
    if _agrees(recovered_period, true_period, tolerance):
        return Match.EXACT

    for order in range(2, max_harmonic + 1):
        if _agrees(recovered_period, true_period * order, tolerance):
            return Match.HARMONIC
        if _agrees(recovered_period, true_period / order, tolerance):
            return Match.HARMONIC

    return Match.MISSED


def _agrees(value: float, reference: float, tolerance: float) -> bool:
    return abs(value - reference) / reference < tolerance
