"""taking the real planets out before putting fake ones in.

Kepler-10 already has planets in the data. leave them there and BLS will
happily hand back the real period for an injection that was never
detectable — which scores as a recovery that never happened.

the ephemerides come from the data itself, by iterative pre-whitening:
search, mask the winner, search again. no literature values, so the mask
always matches whatever curve is actually in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from injrec.search import SearchConfig, search

MASK_DURATION_FACTOR = 1.5

MASK_MAX_PERIOD = 100.0

@dataclass(frozen=True)
class Ephemeris:
    """A periodic transit signal to mask out."""

    period: float
    epoch: float
    duration: float

    def in_transit(self, time_days: np.ndarray) -> np.ndarray:
        """Boolean mask, True for points inside the masked window."""
        phase = (np.asarray(time_days, dtype=float) - self.epoch) % self.period
        phase = np.where(phase > self.period / 2, phase - self.period, phase)
        return np.abs(phase) < MASK_DURATION_FACTOR * self.duration / 2


def mask_transits(light_curve, ephemerides: list[Ephemeris]):
    """Return a copy of ``light_curve`` with in-transit points removed."""
    time = np.asarray(light_curve.time.value, dtype=float)
    keep = np.ones(time.size, dtype=bool)
    for eph in ephemerides:
        keep &= ~eph.in_transit(time)
    return light_curve[keep]


def find_and_mask(
    light_curve,
    n_signals: int = 2,
    config: SearchConfig | None = None,
) -> tuple[object, list[Ephemeris]]:
    """Iteratively detect and mask the strongest signals.

    Returns the cleaned curve and the ephemerides that were removed, so the
    caller can confirm they correspond to the known planets rather than to
    systematics.
    """
    if n_signals < 0:
        raise ValueError(f"n_signals must be non-negative, got {n_signals}")

    if config is None:
        config = SearchConfig(max_period=MASK_MAX_PERIOD)
    cleaned = light_curve
    found: list[Ephemeris] = []

    for _ in range(n_signals):
        detection = search(cleaned, config)
        eph = Ephemeris(
            period=detection.period,
            epoch=detection.epoch,
            duration=detection.duration,
        )
        found.append(eph)
        cleaned = mask_transits(cleaned, [eph])

    return cleaned, found
