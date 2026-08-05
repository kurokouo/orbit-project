"""actually running the thing.

every trial is independent, so the grid parallelises for free. each worker
keeps the base light curve in a module-level slot filled once when the pool
starts
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from injrec.geometry import StellarParams, duration_in_cadences
from injrec.injection import InjectedSignal, inject_into
from injrec.recovery import Match, classify_recovery
from injrec.search import SearchConfig, null_peak_power, search

_WORKER_STATE: dict = {}


@dataclass(frozen=True)
class Trial:
    """One injection and its outcome."""

    period: float
    rp_earth: float
    epoch: float
    impact: float
    recovered_period: float
    recovered_power: float
    outcome: str
    duration_cadences: float

    def as_row(self) -> dict:
        return asdict(self)


def _init_null_worker(time: np.ndarray, flux: np.ndarray, config: dict) -> None:
    import lightkurve as lk

    _WORKER_STATE["lc"] = lk.LightCurve(time=time, flux=flux)
    _WORKER_STATE["config"] = SearchConfig(**config)


def _init_worker(
    time: np.ndarray, flux: np.ndarray, star: dict, config: dict, min_power: float
) -> None:
    _init_null_worker(time, flux, config)
    _WORKER_STATE["star"] = StellarParams(**star)
    _WORKER_STATE["min_power"] = min_power


def _run_trial(signal: InjectedSignal) -> Trial:
    return run_single(
        _WORKER_STATE["lc"],
        signal,
        _WORKER_STATE["star"],
        _WORKER_STATE["config"],
        _WORKER_STATE["min_power"],
    )


def _run_null(seed: int) -> float:
    return null_peak_power(
        _WORKER_STATE["lc"],
        _WORKER_STATE["config"],
        np.random.default_rng(seed),
    )


def run_single(
    base_curve,
    signal: InjectedSignal,
    star: StellarParams,
    config: SearchConfig,
    min_power: float = 0.0,
    cadence_minutes: float = 29.4244,
) -> Trial:
    """Inject one signal, re-run the pipeline, and classify the result.

    Injection happens on the raw curve; ``search`` detrends internally.

    A peak at or below ``min_power`` is a miss whatever period it sits at.
    Without that gate an undetectable injection counts as recovered whenever
    the noise peak lands near the injected period. See ``detection_threshold``
    for where the number comes from.
    """
    injected = inject_into(base_curve, signal, star)
    detection = search(injected, config)
    if detection.power <= min_power:
        outcome = Match.MISSED
    else:
        outcome = classify_recovery(detection.period, signal.period)
    _, _, duration = signal.geometry(star)
    return Trial(
        period=signal.period,
        rp_earth=signal.rp_earth,
        epoch=signal.epoch,
        impact=signal.impact,
        recovered_period=detection.period,
        recovered_power=detection.power,
        outcome=outcome.value,
        duration_cadences=duration_in_cadences(duration, cadence_minutes),
    )


def run_grid(
    base_curve,
    signals: list[InjectedSignal],
    star: StellarParams,
    config: SearchConfig,
    min_power: float = 0.0,
    max_workers: int | None = None,
) -> list[Trial]:
    """Run every signal, in parallel across processes."""
    if not signals:
        raise ValueError("signals must not be empty")

    time, flux = _arrays(base_curve)
    with ProcessPoolExecutor(
        max_workers=_worker_count(max_workers),
        initializer=_init_worker,
        initargs=(time, flux, asdict(star), asdict(config), min_power),
    ) as pool:
        return list(pool.map(_run_trial, signals, chunksize=1))


def detection_threshold(
    base_curve,
    config: SearchConfig,
    n_trials: int = 32,
    percentile: float = 99.0,
    seed: int = 0,
    max_workers: int | None = None,
) -> float:
    """Peak BLS power a signal-free curve can reach by chance.

    Runs ``n_trials`` shuffled copies of the curve and returns the requested
    percentile of their peak powers. A detection at or below this is
    indistinguishable from noise, whatever period it sits at.

    At the default ``n_trials`` the 99th percentile is essentially the largest
    null peak seen, so an injection has to beat every shuffled run. Raise
    ``n_trials`` into the hundreds for the percentile to be literal.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")

    time, flux = _arrays(base_curve)
    seeds = [seed + i for i in range(n_trials)]
    with ProcessPoolExecutor(
        max_workers=_worker_count(max_workers),
        initializer=_init_null_worker,
        initargs=(time, flux, asdict(config)),
    ) as pool:
        peaks = list(pool.map(_run_null, seeds, chunksize=1))
    return float(np.percentile(peaks, percentile))


def _arrays(light_curve) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(light_curve.time.value, dtype=float),
        np.asarray(light_curve.flux.value, dtype=float),
    )


def _worker_count(max_workers: int | None) -> int:
    if max_workers is None:
        return max(1, (os.cpu_count() or 2) - 1)
    return max_workers


@dataclass(frozen=True)
class Recovery:
    """The headline number: what fraction of injections came back.

    Carries an interval because the fraction alone invites over-reading. A few
    hundred injections is a normal budget, and at that size the third digit is
    noise.
    """

    recovered: int
    total: int
    low: float
    high: float

    @property
    def fraction(self) -> float:
        return self.recovered / self.total

    def __str__(self) -> str:
        return (
            f"{self.fraction:.1%}  (95% CI {self.low:.1%}-{self.high:.1%}, "
            f"{self.recovered}/{self.total})"
        )


def recovery_fraction(trials: list[Trial], count_harmonics: bool = False) -> Recovery:
    """Fraction recovered. Harmonics are excluded unless asked for."""
    if not trials:
        raise ValueError("trials must not be empty")
    accepted = {Match.EXACT.value}
    if count_harmonics:
        accepted.add(Match.HARMONIC.value)
    hits = sum(t.outcome in accepted for t in trials)
    low, high = _wilson_interval(hits, len(trials))
    return Recovery(recovered=hits, total=len(trials), low=low, high=high)


def _wilson_interval(hits: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval on a binomial proportion, 95% by default.

    Wilson rather than the textbook normal interval because completeness
    saturates: the big-planet corner recovers everything and the small-planet
    corner recovers nothing. At exactly 0 or 1 the normal interval has zero
    width, claiming certainty where the estimate is weakest. Wilson stays
    inside [0, 1] and keeps a sensible width there.
    """
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, centre - half), min(1.0, centre + half)
