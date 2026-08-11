"""the head of the process. runs the whole experiment on real Kepler data.

downloads the light curve, takes the known planets out, injects a grid of fake
ones, counts what comes back.

writes a CSV of every trial plus a PNG of the surface. re-analyse from the CSV
rather than re-running, the injections are the expensive part.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd

from injrec.completeness import build_surface, plot_surface
from injrec.geometry import StellarParams
from injrec.grid import GridSpec, sample_signals
from injrec.masking import MASK_MAX_PERIOD, find_and_mask
from injrec.runner import detection_threshold, recovery_fraction, run_grid
from injrec.search import SearchConfig, window_cadences

TARGET = "Kepler-10"
OUTDIR = Path("results")

# log g of the Sun
LOGG_SUN = 4.438


def load_base_curve(target: str = TARGET, mission="Kepler", author="Kepler",
                    exptime=1800):
    """grab the data, stitch it, drop the NaNs.

    author matters. without it the Kepler search also picks up a KBONUS-BKG
    product that is a completely different star, and TESS gives you QLP and
    TESS-SPOC alongside SPOC.
    """
    search = lk.search_lightcurve(
        target, mission=mission, author=author, exptime=exptime
    )
    if len(search) == 0:
        raise RuntimeError(f"no {author} {exptime}s products found for {target!r}")
    return search.download_all().stitch().remove_nans()


def star_from_header(light_curve) -> StellarParams:
    """the curve already knows its star. RADIUS and LOGG ride along with it.
    """
    radius = light_curve.meta["RADIUS"]
    mass = 10.0 ** (light_curve.meta["LOGG"] - LOGG_SUN) * radius**2
    return StellarParams(radius_sun=radius, mass_sun=mass)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=TARGET)
    p.add_argument("--mission", default="Kepler")
    p.add_argument("--author", default="Kepler", help="SPOC for TESS")
    p.add_argument("--exptime", type=int, default=1800, help="120 for TESS 2-min")
    p.add_argument("--window-days", type=float, default=2.064,
                   help="detrending window, converted to cadences")
    p.add_argument("--periods", type=int, default=4, help="period bins")
    p.add_argument("--radii", type=int, default=4, help="radius bins")
    p.add_argument("--repeats", type=int, default=8, help="injections per cell")
    p.add_argument("--min-period", type=float, default=1.0)
    p.add_argument("--max-period", type=float, default=40.0)
    p.add_argument("--min-radius", type=float, default=0.2)
    p.add_argument("--max-radius", type=float, default=4.0)
    p.add_argument("--frequency-factor", type=float, default=3000.0)
    p.add_argument("--mask-planets", type=int, default=2,
                   help="known signals to remove before injecting")
    p.add_argument("--null-trials", type=int, default=32,
                   help="shuffled runs used to calibrate the detection threshold")
    p.add_argument("--null-percentile", type=float, default=99.0,
                   help="percentile of the null peaks to require injections to beat")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--outdir", type=Path, default=OUTDIR)
    p.add_argument("--seed", type=int, default=20260804)
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    base = load_base_curve(args.target, args.mission, args.author, args.exptime)
    baseline = float(base.time.value.max() - base.time.value.min())
    cadence = float(np.median(np.diff(base.time.value))) * 24 * 60
    window = window_cadences(args.window_days, cadence)
    star = star_from_header(base)
    print(f"  {len(base)} points over {baseline:.0f} d at {cadence:.2f} min")
    print(f"  {star.radius_sun:.3f} Rsun, {star.mass_sun:.3f} Msun")
    print(f"  detrending window {window} cadences = {window * cadence / 1440:.2f} d")

    if args.mask_planets:
        print(f"masking {args.mask_planets} known signals")
        mask_config = SearchConfig(
            max_period=min(MASK_MAX_PERIOD, baseline / 2), window_length=window
        )
        base, found = find_and_mask(base, args.mask_planets, mask_config)
        for i, eph in enumerate(found, 1):
            print(f"  signal {i}: P = {eph.period:.5f} d, "
                  f"duration = {eph.duration * 24:.2f} h")
        print(f"  {len(base)} points remain")

    spec = GridSpec(
        period_range=(args.min_period, args.max_period),
        radius_range=(args.min_radius, args.max_radius),
        n_periods=args.periods,
        n_radii=args.radii,
        repeats=args.repeats,
    )
    signals = sample_signals(
        spec, baseline_days=baseline, rng=np.random.default_rng(args.seed)
    )
    config = SearchConfig(
        min_period=args.min_period,
        max_period=args.max_period,
        frequency_factor=args.frequency_factor,
        window_length=window,
    )

    # calibrated on the masked curve so the threshold matches the noise the
    # injections actually compete against. costs one BLS per shuffle
    print(f" ({args.null_trials} shuffled runs) ...")
    started = time.time()
    threshold = detection_threshold(
        base, config, args.null_trials, args.null_percentile, args.seed, args.workers
    )
    elapsed = time.time() - started
    print(f"  {args.null_percentile:g}th pct = {threshold:.3g} "
          f"({elapsed:.0f}s, {elapsed / args.null_trials:.2f}s per run)")

    print(f"running {spec.total_injections} injections ...")
    started = time.time()
    trials = run_grid(base, signals, star, config, cadence, threshold, args.workers)
    elapsed = time.time() - started
    print(f"  done in {elapsed:.0f}s ({elapsed / len(trials):.2f}s per injection)")

    frame = pd.DataFrame([t.as_row() for t in trials])
    csv_path = args.outdir / "trials.csv"
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    surface = build_surface(trials, spec)
    strict = recovery_fraction(trials)
    loose = recovery_fraction(trials, count_harmonics=True)

    print()
    print(f"recovered (exact)           : {strict}")
    print(f"recovered (incl. harmonics) : {loose}")
    print(frame["outcome"].value_counts().to_string())
    print()
    print("completeness by cell (radius rows, period columns):")
    print(_format_surface(surface))

    ax = plot_surface(surface)
    ax.set_title(f"{args.target}: BLS transit recovery, {len(trials)} injections")
    png_path = args.outdir / "completeness.png"
    ax.figure.savefig(png_path, dpi=120, bbox_inches="tight")
    print(f"\nwrote {csv_path} and {png_path}")


def _format_surface(surface) -> str:
    centres_p = np.sqrt(surface.period_edges[:-1] * surface.period_edges[1:])
    centres_r = np.sqrt(surface.radius_edges[:-1] * surface.radius_edges[1:])
    header = "  R\\P  " + "".join(f"{p:>8.1f}" for p in centres_p)
    rows = [header]
    for i, r in reversed(list(enumerate(centres_r))):
        cells = "".join(
            "     ---" if np.isnan(v) else f"{v:>8.2f}"
            for v in surface.fraction[i]
        )
        rows.append(f"{r:>6.1f} {cells}")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
