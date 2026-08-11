# orbit-project

this is a passion project i am doing by myself, it was a image processing task
for a discipline but it ended becoming this, before i was a computer science
student i was also a kid and dreamt about space, so this is something i am
proud of!

it's in two halves:

1. **`planets.ipynb`** — finds Kepler-10b from scratch: download, stitch,
   detrend, BLS period search, phase-fold.
2. **`injrec/`** — measures how good that pipeline actually is, by injecting
   synthetic transits of known size and period and counting how many come back.

the second half is really the point. "i found a planet" means nothing on its
own, you also need to know what fraction of planets you *would* have found.
that fraction is the completeness surface, and it's the number you divide by
before you can say anything about how many planets are out there.

---

## quick start

```bash
uv sync
uv run python scripts/run_completeness.py
```

defaults give a 4×4 grid with 8 repeats, so 128 injections. it writes
`results/trials.csv` and `results/completeness.png`.

## the pipeline

```
stitch → mask known planets → INJECT → flatten → BLS → classify
```

injection happens on the **raw** curve, before detrending. that ordering is
the entire experiment. [`injrec/README.md`](injrec/README.md) has the reasoning
and what each module does.

## files

| | |
| --- | --- |
| `planets.ipynb` | the exploratory analysis |
| `injrec/` | the injection–recovery harness |
| `scripts/run_completeness.py` | runs the experiment, writes CSV + PNG |
| `tests/` | `uv run pytest` |
| `results/` | output of the last run |

## flags

`--help` lists them all. the ones worth knowing:

| Flag | Default | Notes |
| --- | --- | --- |
| `--periods`, `--radii` | 4, 4 | grid resolution. cells = periods × radii |
| `--repeats` | 8 | injections per cell. this is what sets the error bars |
| `--frequency-factor` | 3000 | BLS grid coarseness. higher is coarser and faster |
| `--mask-planets` | 2 | known signals to remove first. for Kepler-10 that's b and c |
| `--null-trials` | 32 | shuffled runs used to calibrate the detection threshold |
| `--workers` | cpus−1 | |
| `--seed` | 20260804 | makes the sampling reproducible |
| `--outdir` | `results/` | |

a bigger run:

```bash
uv run python scripts/run_completeness.py \
  --periods 5 --radii 5 --repeats 12 \
  --min-radius 0.2 --max-radius 4.0 \
  --min-period 1.0 --max-period 40.0
```

## outputs

- `results/trials.csv` — one row per injection, with columns `period,
  rp_earth, epoch, impact, recovered_period, recovered_power, outcome,
  duration_cadences`. **re-analyse from this rather than re-running**, the
  injections are the expensive part.
- `results/completeness.png` — the recovery-fraction surface.

---

## papers i read for this

the two that got me started, and the ones this is basically a tiny version of:

- Christiansen et al. 2020, *Measuring Transit Signal Recovery in the Kepler
  Pipeline. IV*, AJ 160, 159 —
  [doi:10.3847/1538-3881/abab0b](https://iopscience.iop.org/article/10.3847/1538-3881/abab0b).
  same idea as `injrec/`, at mission scale: inject synthetic transits into real
  data, re-run the actual pipeline.

- Bryson et al. 2020, *A Probabilistic Approach to Kepler Completeness and
  Reliability for Exoplanet Occurrence Rates*, AJ 159, 279 —
  [arXiv:1906.03575](https://arxiv.org/abs/1906.03575).

- shoutout petfisicaUFRN :<https://github.com/PETfisicaUFRN/PET.py/blob/main/Notebooks/Identificando%20Tr%C3%A2nsito%20Planet%C3%A1rio.ipynb> this is the place where i learned how the
idea of transit detection works and i think it is important to show it as it is where i learned
how to do it!

- Kreidberg 2015, *batman: BAsic Transit Model cAlculatioN in Python*, PASP 127,
1161 — [doi:10.1086/683602](https://iopscience.iop.org/article/10.1086/683602).
the transit model `injrec/injection.py` builds every fake planet with, pretty interesting
actually.
