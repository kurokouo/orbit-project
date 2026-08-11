"""detrend + BLS. this is the pipeline being measured.

takes a *raw* curve and detrends internally, so the inject-before-detrend
ordering can't be got wrong from outside.

BLS is Kovacs, Zucker & Mazeh (2002), via astropy through lightkurve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass()
class SearchConfig:
    """Detrending and BLS settings."""

    window_length: int = 101
    min_period: float = 0.5
    max_period: float = 20.0
    min_duration: float = 0.03
    max_duration: float = 0.30
    n_durations: int = 12

    # df = frequency_factor * min_duration / baseline^2, so higher is coarser
    frequency_factor: float = 500.0

    def __post_init__(self) -> None:
        if self.window_length % 2 == 0:
            raise ValueError(f"window_length must be odd, got {self.window_length}")
        if self.min_period <= 0 or self.max_period <= self.min_period:
            raise ValueError("period range must be positive and increasing")
        if self.min_duration <= 0 or self.max_duration <= self.min_duration:
            raise ValueError("duration range must be positive and increasing")

    @property
    def durations(self) -> np.ndarray:
        return np.linspace(self.min_duration, self.max_duration, self.n_durations)


@dataclass()
class Detection:
    """Strongest periodic box signal found."""

    period: float
    epoch: float
    duration: float
    depth: float
    power: float


def detrend(light_curve, config: SearchConfig):
    """Window is in cadences, not days."""
    return light_curve.flatten(window_length=config.window_length)


def _periodogram(flat_curve, config: SearchConfig):
    return flat_curve.to_periodogram(
        method="bls",
        minimum_period=config.min_period,
        maximum_period=config.max_period,
        duration=config.durations,
        frequency_factor=config.frequency_factor,
    )


def null_peak_power(light_curve, config: SearchConfig, rng: np.random.Generator) -> float:
    """Peak BLS power from one signal-free copy of the same noise."""
    flat = detrend(light_curve, config)
    order = rng.permutation(flat.flux.size)
    shuffled = flat.copy()
    shuffled.flux = flat.flux[order]  # index, don't permute: keeps the units
    return float(np.nanmax(_periodogram(shuffled, config).power.value))


def search(light_curve, config: SearchConfig) -> Detection:
    """Detrend then run BLS, returning the peak."""
    periodogram = _periodogram(detrend(light_curve, config), config)
    return Detection(
        period=float(periodogram.period_at_max_power.value),
        epoch=float(periodogram.transit_time_at_max_power.value),
        duration=float(periodogram.duration_at_max_power.value),
        depth=float(periodogram.depth_at_max_power),
        power=float(np.nanmax(periodogram.power.value)),
    )
