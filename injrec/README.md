# injrec

the harness that measures how good the pipeline actually is. it injects fake
transits of known size and period into the real light curve, runs the same
search, and counts how many come back.

the output is a completeness surface — the number you divide by to turn "i
found 3 planets" into "there are N planets out there". without it a detection
count means nothing.

---

## modules

| Module | Responsibility |
| --- | --- |
| `geometry.py` | a/R\*, inclination, depth, duration from stellar + orbital parameters. pure math, no i/o |
| `injection.py` | builds batman transit models and multiplies them into a curve. supersamples for the 29.4 min integration |
| `masking.py` | removes the *real* planets first, by iterative pre-whitening |
| `search.py` | **the pipeline under test.** Savitzky-Golay detrend + BLS |
| `recovery.py` | classifies a detection as `perfect` / `harmonic` / `fail` |
| `grid.py` | samples the log-spaced (period, radius) grid |
| `runner.py` | runs trials in parallel, one process per injection. calibrates the detection threshold |
| `completeness.py` | bins trials into a surface and plots it |

---

## the one rule

**inject on the raw curve, before detrending.**

```
stitch → mask known planets → INJECT → flatten → BLS → classify
```

injecting after `flatten()` skips the exact step whose signal loss the whole
experiment exists to measure. nothing crashes, nothing errors, and
completeness comes out too high, the worst kind of bug, because the number
still looks reasonable.

---

## measured cost

BLS dominates.

| Stage | Time |
| --- | --- |
| load + stitch (once) | 9.4 s |
| inject | <0.1 s |
| detrend | 0.6 s |
| BLS `ff=500` | 40.3 s |
| BLS `ff=1500` | 13.1 s |
| BLS `ff=3000` | 7.0 s |
| null threshold, 32 shuffles | 7.0 s × 32, parallel |

`frequency_factor=3000` is the default because it was checked against 500 on
166 / 74 / 47 ppm injections and recovered the period exactly every time — not
because it's fast. frequency resolution scales as 1/baseline², so re-check if
you move to a much longer or shorter dataset.

masking deliberately keeps the finer default instead. the 3000 setting was
only validated out to ~12 d, and Kepler-10c sits at 45.3 d where a coarse grid
is least trustworthy. it runs twice per experiment, so the extra ~60 s is
cheap insurance.

---

## usage

```bash
uv run python scripts/run_completeness.py
uv run python scripts/run_completeness.py --periods 5 --radii 5 --repeats 12
```

writes `results/trials.csv` and `results/completeness.png`. re-analyse from
the CSV if you only want to check that

---

## validation

there's no test suite in the repo, so check the geometry by hand after
touching `geometry.py`. it should reproduce published Kepler-10b values:

```python
from injrec.geometry import StellarParams, scaled_semi_major_axis, transit_duration

star = StellarParams()                                        # Kepler-10
a_rs = scaled_semi_major_axis(0.837491, star)
print(a_rs)                                                   # expect ~3.4-3.5
print(transit_duration(0.837491, a_rs, 0.01264, 0.30) * 24)   # expect ~1.8 h
```

a unit-conversion slip here shifts the entire completeness surface without
raising anything, so it's worth the 10 seconds.

---
