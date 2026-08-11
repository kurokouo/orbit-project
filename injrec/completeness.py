"""turning a pile of trials into the completeness surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from injrec.grid import GridSpec
from injrec.recovery import Match
from injrec.runner import Trial


@dataclass(frozen=True)
class CompletenessSurface:
    """Recovery fraction on the (period, radius) grid.

    fraction is indexed [radius, period], so radius plots up the vertical axis.
    """

    period_edges: np.ndarray
    radius_edges: np.ndarray
    fraction: np.ndarray
    counts: np.ndarray


def build_surface(
    trials: list[Trial],
    spec: GridSpec,
    count_harmonics: bool = False,
) -> CompletenessSurface:
    """Bin trials into grid cells and get the recovery fraction of each."""
    accepted = {Match.EXACT.value}
    if count_harmonics:
        accepted.add(Match.HARMONIC.value)

    periods = np.array([t.period for t in trials])
    radii = np.array([t.rp_earth for t in trials])
    hits = np.array([t.outcome in accepted for t in trials], dtype=float)

    p_edges = spec.period_edges
    r_edges = spec.radius_edges

    counts, _, _ = np.histogram2d(radii, periods, bins=[r_edges, p_edges])
    successes, _, _ = np.histogram2d(
        radii, periods, bins=[r_edges, p_edges], weights=hits
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(counts > 0, successes / counts, np.nan)

    return CompletenessSurface(p_edges, r_edges, fraction, counts)


def plot_surface(surface: CompletenessSurface, ax=None, cmap: str = "viridis"):
    """Render the completeness surface as a pcolormesh."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    mesh = ax.pcolormesh(
        surface.period_edges,
        surface.radius_edges,
        surface.fraction,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Injected period [d]")
    ax.set_ylabel("Injected radius [$R_\\oplus$]")
    ax.figure.colorbar(mesh, ax=ax, label="Recovery fraction")

    for i in range(surface.fraction.shape[0]):
        for j in range(surface.fraction.shape[1]):
            value = surface.fraction[i, j]
            if np.isnan(value):
                continue
            x = np.sqrt(surface.period_edges[j] * surface.period_edges[j + 1])
            y = np.sqrt(surface.radius_edges[i] * surface.radius_edges[i + 1])
            ax.text(
                x, y, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if value < 0.6 else "black",
            )
    return ax
