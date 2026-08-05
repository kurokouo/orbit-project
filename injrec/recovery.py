"""Deciding whether an injected signal was recovered.

Keptsmall and dependency-free: this is the module whose bugs
would be invisible in the final completeness surface.
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
    """Classify a BLS detection against theperiod.
    """
    if recovered_period <= 0:
        raise ValueError(f"recovered_period must be positive, got {recovered_period}")
    if true_period <= 0:
        raise ValueError(f"true_period must be positive, got {true_period}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if max_harmonic < 1:
        raise ValueError(f"max_harmonic must be >= 1, got {max_harmonic}")

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
