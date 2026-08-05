"""picking which (period, radius) pairs to inject.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from injrec.injection import InjectedSignal

MAX_IMPACT = 1.0


@dataclass(frozen=True)
class GridSpec:
    """Extent and resolution of the completeness grid."""

    period_range: tuple[float, float]
    radius_range: tuple[float, float]
    n_periods: int
    n_radii: int
    repeats: int

    def __post_init__(self) -> None:
        for name, (lo, hi) in (
            ("period_range", self.period_range),
            ("radius_range", self.radius_range),
        ):
            if lo <= 0:
                raise ValueError(f"{name} lower bound must be positive, got {lo}")
            if hi <= lo:
                raise ValueError(f"{name} must be increasing, got ({lo}, {hi})")
        for name, value in (
            ("n_periods", self.n_periods),
            ("n_radii", self.n_radii),
            ("repeats", self.repeats),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

    @property
    def period_edges(self) -> np.ndarray:
        return np.geomspace(*self.period_range, self.n_periods + 1)

    @property
    def radius_edges(self) -> np.ndarray:
        return np.geomspace(*self.radius_range, self.n_radii + 1)

    @property
    def total_injections(self) -> int:
        return self.n_periods * self.n_radii * self.repeats


def sample_signals(
    spec: GridSpec,
    baseline_days: float,
    rng: np.random.Generator,
) -> list[InjectedSignal]:
    """Draw ``spec.repeats`` signals uniformly within each grid cell.

    Sampling within cells rather than at cell centres avoids a pathology
    where every injection in a row shares one period and therefore one
    alias structure, which makes recovery look artificially bimodal.
    """
    if baseline_days <= 0:
        raise ValueError(f"baseline_days must be positive, got {baseline_days}")
    if spec.period_range[1] > baseline_days:
        raise ValueError(
            f"period_range upper bound {spec.period_range[1]} d exceeds the "
            f"{baseline_days} d baseline; such signals cannot be recovered"
        )

    p_edges = spec.period_edges
    r_edges = spec.radius_edges

    signals: list[InjectedSignal] = []
    for i in range(spec.n_periods):
        for j in range(spec.n_radii):
            for _ in range(spec.repeats):
                period = _loguniform(p_edges[i], p_edges[i + 1], rng)
                radius = _loguniform(r_edges[j], r_edges[j + 1], rng)
                signals.append(
                    InjectedSignal(
                        period=period,
                        rp_earth=radius,
                        epoch=float(rng.uniform(0.0, period)),
                        impact=float(rng.uniform(0.0, MAX_IMPACT)),
                    )
                )
    return signals


def _loguniform(low: float, high: float, rng: np.random.Generator) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))
