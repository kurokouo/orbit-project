"""building and injecting the fake transits.

the ordering matters more than anything else in this file. `inject_into`
works on the raw stitched curve, and detrending happens *after*, so the
filter chews on the fake signal exactly like it would a real one.

get that backwards and completeness comes out too high with nothing to show
for it — no crash, no error, just a wrong number that looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import batman
import numpy as np

from injrec.geometry import (
    StellarParams,
    inclination_from_impact,
    radius_ratio_from_earth_radii,
    scaled_semi_major_axis,
    transit_duration,
)

#: Kepler long-cadence integration time, in days.
LONG_CADENCE_DAYS = 29.4244 / (24 * 60)

#: Sub-samples per exposure when modelling finite integration time.
DEFAULT_SUPERSAMPLE = 7


@dataclass(frozen=True)
class InjectedSignal:
    """A synthetic transit to inject. Circular orbit, quadratic limb darkening."""

    period: float
    rp_earth: float
    epoch: float = 0.0
    impact: float = 0.0

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.rp_earth <= 0:
            raise ValueError(f"rp_earth must be positive, got {self.rp_earth}")
        if self.impact < 0:
            raise ValueError(f"impact must be non-negative, got {self.impact}")

    def geometry(self, star: StellarParams) -> tuple[float, float, float]:
        """Return (radius_ratio, scaled_axis, duration_days)."""
        k = radius_ratio_from_earth_radii(self.rp_earth, star)
        a_rs = scaled_semi_major_axis(self.period, star)
        dur = transit_duration(self.period, a_rs, k, self.impact)
        return k, a_rs, dur


def build_transit_model(
    time_days: np.ndarray,
    signal: InjectedSignal,
    star: StellarParams,
    exp_time_days: float = LONG_CADENCE_DAYS,
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> np.ndarray:
    """Return normalized flux (1.0 out of transit) sampled at ``time_days``.

    ``exp_time_days`` drives batman's supersampling and defaults to Kepler
    long cadence. Set it to the exposure time of whatever mission the curve
    came from. It must be positive: an instantaneous model gives a sharper
    transit than the data could ever show. Kepler-10b's transit spans only
    ~3.5 long-cadence points, so the correction is not optional at short
    periods, and getting it wrong inflates completeness instead of raising.
    """
    time_days = np.asarray(time_days, dtype=float)
    if time_days.size == 0:
        raise ValueError("time_days must not be empty")
    if exp_time_days <= 0:
        raise ValueError(f"exp_time_days must be positive, got {exp_time_days}")
    if supersample < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}")

    k, a_rs, duration = signal.geometry(star)
    if duration == 0.0:
        return np.ones_like(time_days)

    params = batman.TransitParams()
    params.t0 = signal.epoch
    params.per = signal.period
    params.rp = k
    params.a = a_rs
    params.inc = inclination_from_impact(signal.impact, a_rs)
    params.ecc = 0.0
    params.w = 90.0
    params.u = [star.limb_dark_u1, star.limb_dark_u2]
    params.limb_dark = "quadratic"

    model = batman.TransitModel(
        params,
        time_days,
        supersample_factor=supersample,
        exp_time=exp_time_days,
    )
    return np.asarray(model.light_curve(params), dtype=float)


def inject_into(light_curve, signal: InjectedSignal, star: StellarParams):
    """Return a copy of ``light_curve`` with ``signal`` multiplied in.

    Call this on the raw stitched curve, never on a flattened one.
    """
    model = build_transit_model(
        np.asarray(light_curve.time.value, dtype=float), signal, star
    )
    injected = light_curve.copy()
    injected.flux = light_curve.flux * model
    return injected
