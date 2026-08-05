"""transit geometry — the pure math, no i/o.

turns stellar and orbital parameters into the things the rest of the harness
needs: a/R*, inclination, depth, duration.

units: angles are degrees on the way in and out, radians inside. times are
days. radii are whatever each field says they are — read the names.

if something here is wrong the whole completeness surface shifts and nothing
raises, so check it against published Kepler-10b values after touching it
(see injrec/README.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants, SI unless noted.
_G = 6.67430e-11
_R_SUN_M = 6.957e8
_M_SUN_KG = 1.98892e30
_SECONDS_PER_DAY = 86400.0

#: Earth radii per solar radius, for converting planet sizes to Rp/R*
EARTH_RADII_PER_SOLAR = _R_SUN_M / 6.371e6


@dataclass(frozen=True)
class StellarParams:
    """Host star properties.

    Defaults describe Kepler-10. Limb darkening is quadratic in the Kepler
    bandpass; the defaults are appropriate for a ~5600 K dwarf and should be
    replaced when moving to a different star.
    """

    radius_sun: float = 1.065
    mass_sun: float = 0.910
    limb_dark_u1: float = 0.40
    limb_dark_u2: float = 0.26

    def __post_init__(self) -> None:
        if self.radius_sun <= 0:
            raise ValueError(f"radius_sun must be positive, got {self.radius_sun}")
        if self.mass_sun <= 0:
            raise ValueError(f"mass_sun must be positive, got {self.mass_sun}")

    @property
    def density_kg_m3(self) -> float:
        mass = self.mass_sun * _M_SUN_KG
        radius = self.radius_sun * _R_SUN_M
        return mass / ((4.0 / 3.0) * math.pi * radius**3)


def scaled_semi_major_axis(period_days: float, star: StellarParams) -> float:
    """Return a/R* for a circular orbit.

    Derived from Kepler's third law expressed through the stellar density:
        (a/R*)^3 = G * rho_star * P^2 / (3 * pi)
    """
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")
    period_s = period_days * _SECONDS_PER_DAY
    cubed = _G * star.density_kg_m3 * period_s**2 / (3.0 * math.pi)
    return float(cubed ** (1.0 / 3.0))


def inclination_from_impact(impact: float, scaled_axis: float) -> float:
    """Return orbital inclination in degrees, for a circular orbit."""
    if scaled_axis <= 0:
        raise ValueError(f"scaled_axis must be positive, got {scaled_axis}")
    cos_i = min(impact / scaled_axis, 1.0)
    return float(math.degrees(math.acos(cos_i)))


def transit_duration(
    period_days: float,
    scaled_axis: float,
    radius_ratio: float,
    impact: float,
) -> float:
    """Return total (first-to-fourth contact) transit duration in days.

    Returns 0.0 when the geometry does not produce a transit, i.e. when the
    impact parameter exceeds 1 + Rp/R*.
    """
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")

    limit = 1.0 + radius_ratio
    if impact >= limit:
        return 0.0

    inc = math.radians(inclination_from_impact(impact, scaled_axis))
    numerator = math.sqrt(limit**2 - impact**2)
    denominator = scaled_axis * math.sin(inc)
    ratio = numerator / denominator
    if ratio >= 1.0:  # planet never leaves the disc; degenerate geometry
        return float(period_days / 2.0)
    return float((period_days / math.pi) * math.asin(ratio))


def duration_in_cadences(duration_days: float, cadence_minutes: float) -> float:
    """Points spanning the transit. Below ~3 the transit is unresolved."""
    if cadence_minutes <= 0:
        raise ValueError(f"cadence_minutes must be positive, got {cadence_minutes}")
    return float(duration_days * 24 * 60 / cadence_minutes)


def radius_ratio_from_earth_radii(rp_earth: float, star: StellarParams) -> float:
    """Convert a planet radius in Earth radii to Rp/R*."""
    if rp_earth <= 0:
        raise ValueError(f"rp_earth must be positive, got {rp_earth}")
    return float(rp_earth / (star.radius_sun * EARTH_RADII_PER_SOLAR))
