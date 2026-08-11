# injrec

the harness that measures how good the pipeline actually is. it injects fake
transits of known size and period into the real light curve, runs the same
search, and counts how many come back.

usage and flags are in the ../README.md, this file is the
readme for the pipeline

---

## pipeline

**inject on the raw curve, before detrending.**

```
stitch → mask known planets → INJECT → flatten → BLS → classify
```

injecting after `flatten()` skips the exact step whose signal loss the whole
experiment exists to measure.

`search()` takes a raw curve and detrends internally.

---

## modules

| Module | Responsibility |
| --- | --- |
| `geometry.py` | a/R\*, inclination, depth, duration from stellar + orbital parameters. pure math, no i/o |
| `injection.py` | builds batman transit models and multiplies them into a curve. supersamples over the exposure time it is handed |
| `masking.py` | removes the *real* planets first, by iterative pre-whitening |
| `search.py` | **the pipeline under test.** Savitzky-Golay detrend + BLS |
| `recovery.py` | classifies a detection as `perfect` / `harmonic` / `fail` |
| `grid.py` | samples the log-spaced (period, radius) grid |
| `runner.py` | runs trials in parallel, one process per injection. calibrates the detection threshold |
| `completeness.py` | bins trials into a surface and plots it |

---

## why the detection threshold exists

BLS always returns *something*. on a curve with no planet in it, the peak
lands wherever the noise happens to pile up, an undetectable injection scores
as a recovery every time that noise peak lands nearby, and completeness comes
out too high at exactly the small-planet end you care about.

so `runner.detection_threshold` shuffles the flux of the masked curve 32 times
and takes the 99th percentile of the peak powers. anything at or below that is
indistinguishable from noise, whatever period it sits at.

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
166 / 74 / 47 ppm injections and recovered the period exactly every time, not
because it's fast. frequency resolution scales as 1/baseline², so re-check if
you move to a much longer or shorter dataset.

masking deliberately keeps the finer default instead. the 3000 setting was
only validated out to ~12 d, and Kepler-10c sits at 45.3 d where a coarse grid
is least trustworthy. it runs twice per experiment, so the extra ~60 s is
cheap insurance.
