"""transit geometry. pure math, nothing here touches a light curve.

units: angles are degrees on the way in and out, radians inside. times in
days. radii are whatever the field name says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#gravitational newtons constant
_G = 6.67430e-11

#solar radius
_R_SUN_M = 6.957e8

#solar mass
_M_SUN_KG = 1.98892e30

#seconds/day
_SECONDS_PER_DAY = 86400.0

# R_sun expressed in Earth radii, for getting to Rp/R*
EARTH_RADII_PER_SOLAR = _R_SUN_M / 6.371e6


@dataclass(frozen=True)
class StellarParams:
    """Host star. Defaults are Kepler-10, limb darkening for the Kepler band."""

    radius_sun: float = 1.065
    mass_sun: float = 0.910
    limb_dark_u1: float = 0.40
    limb_dark_u2: float = 0.26

    def __post_init__(self) -> None:
        if self.radius_sun <= 0 or self.mass_sun <= 0:
            raise ValueError("stellar radius and mass must both be positive")

    @property
    def density_kg_m3(self) -> float:
        mass = self.mass_sun * _M_SUN_KG
        radius = self.radius_sun * _R_SUN_M
        return mass / ((4.0 / 3.0) * math.pi * radius**3)


def scaled_semi_major_axis(period_days: float, star: StellarParams) -> float:
    """a/R* for a circular orbit, Kepler III written through the density:

        (a/R*)^3 = G * rho_star * P^2 / (3 * pi)
    """
    period_s = period_days * _SECONDS_PER_DAY
    cubed = _G * star.density_kg_m3 * period_s**2 / (3.0 * math.pi)
    return float(cubed ** (1.0 / 3.0))


def inclination_from_impact(impact: float, scaled_axis: float) -> float:
    """Inclination in degrees. Inverts b = (a/R*) cos i."""
    cos_i = min(impact / scaled_axis, 1.0)
    return float(math.degrees(math.acos(cos_i)))


def transit_duration(
    period_days: float,
    scaled_axis: float,
    radius_ratio: float,
    impact: float,
) -> float:
    """First-to-fourth contact duration, in days.

        T = (P/pi) * arcsin( sqrt((1+k)^2 - b^2) / ((a/R*) sin i) )

    0.0 if it doesn't transit at all.
    """
    limit = 1.0 + radius_ratio
    if impact >= limit:
        return 0.0

    inc = math.radians(inclination_from_impact(impact, scaled_axis))
    numerator = math.sqrt(limit**2 - impact**2)
    denominator = scaled_axis * math.sin(inc)
    ratio = numerator / denominator
    if ratio >= 1.0:  # never leaves the disc, degenerate
        return float(period_days / 2.0)
    return float((period_days / math.pi) * math.asin(ratio))


def duration_in_cadences(duration_days: float, cadence_minutes: float) -> float:
    """Points spanning the transit. Below ~3 it's unresolved."""
    return float(duration_days * 24 * 60 / cadence_minutes)


def radius_ratio_from_earth_radii(rp_earth: float, star: StellarParams) -> float:
    return float(rp_earth / (star.radius_sun * EARTH_RADII_PER_SOLAR))
