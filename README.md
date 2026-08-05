# orbit-project

this is a passion project i am doing by myself, it was a image processing task for a discipline but it ended becoming this, before i was a computer science student i was also a kid and dreamt about space, so this is something i am proud of!

i also based some stuff from these articles:
<https://arxiv.org/abs/1906.03575>

<https://iopscience.iop.org/article/10.3847/1538-3881/abab0b>

### files

1. **`planets.ipynb`** — an exploratory notebook that finds Kepler-10b from
   scratch: download, stitch, detrend, BLS period search, phase-fold.
2. **`injrec/`** — an injection–recovery harness that measures how good that
   pipeline actually is, by injecting synthetic transits of known size and
   period and counting how many come back.

---

## Quick start

```bash
uv sync                                    # install deps + injrec
uv run python scripts/run_completeness.py  # experiment -> csv + png
```

---

## File map

### ./

| File | What it is |
| --- | --- |
| `planets.ipynb` | exploratory analysis |
| `pyproject.toml` | build config |
| `uv.lock` | uv stuff |
| `results/` | output of the last experiment run |

### `./injrec/` — the harness

Read `./injrec/README.md` reasoning behind the
configuration choices. What each module owns:

| Module | Responsibility |
| --- | --- |
| `geometry.py` | a/R\*, inclination, depth, duration from stellar + orbital parameters. Math |
| `injection.py` | builds batman transit models and multiplies them into a light curve. Handles long-cadence sme aring via sampling. |
| `masking.py` | removes the *real* planets before injecting fake ones |
| `search.py` | **The pipeline under test.** Savitzky-Golay detrend + BLS. Change this and you change what's being measured. |
| `recovery.py` | classifies a detection as `perfect` / `harmonic` / `fail`. |
| `grid.py` | samples the log-spaced (period, radius) parameter grid. |
| `runner.py` | runs trials in parallel, one per injection. |
| `completeness.py` | plots stuff |

### `scripts/`

| `run_completeness.py` | the head of the process, combines everything and writes CSV + PNG |

## How to run the experiment

```bash
uv run python scripts/run_completeness.py --help
```

Bare defaults give a 4×4 grid with 8 repeats — 128 injections.

Useful flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--periods`, `--radii` | 4, 4 | grid resolution. Cells = periods × radii. |
| `--repeats` | 8 | injections per cell. Drives the statistical error: 12 gives |
| `--frequency-factor` | 3000 | BLS grid coarseness. Validated against 500 on 47–166 ppm signals — identical recovery, and much faster! |
| `--mask-planets` | 2 | known signals to remove first. For Kepler-10 that's b and c. |
| `--workers` | cpus−1 | Parallelism. |
| `--seed` | 20260804 | Makes the sampling reproducible. |
| `--outdir` | `results/` | |

## example of a run with flags

```bash
uv run python scripts/run_completeness.py \
  --periods 5 --radii 5 --repeats 12 \
  --min-radius 0.2 --max-radius 4.0 \
  --min-period 1.0 --max-period 40.0
```

### Outputs

- `results/trials.csv` — one row per injection, columns
  `period, rp_earth, epoch, impact, recovered_period, recovered_power, outcome, duration_cadences`.
  **Re-analyse from this rather than re-running.**
- `results/completeness.png` — the recovery-fraction surface.

---

## The one rule that matters

**Injection happens on the raw light curve, before detrending.**

pipeline:

```
stitch → mask known planets → INJECT → flatten → BLS → classify
```

---
