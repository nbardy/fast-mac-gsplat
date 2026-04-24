# v7.2 validation notes

This environment cannot run the Metal/MPS extension, so the validation here is limited to the Python reference path and algorithmic checks.

## 1) Tiled capture exactness versus dense capture

Ran:

- `tests/reference_tiled_exactness.py`

Observed:

```text
case B=1 G=8 H=7 W=9 K=2 tile=4 dense_work=504 tiled_work=490 ratio=1.03x
  max |g_means - ref|   = 3.5762786865234375e-07
  max |g_conics - ref|  = 1.6540288925170898e-06
  max |g_colors - ref|  = 3.5762786865234375e-07
  max |g_opacity - ref| = 6.556510925292969e-07
  max |g_depths|        = 0.0
case B=1 G=12 H=8 W=8 K=2 tile=4 dense_work=768 tiled_work=768 ratio=1.00x
  max |g_means - ref|   = 4.76837158203125e-07
  max |g_conics - ref|  = 1.430511474609375e-06
  max |g_colors - ref|  = 7.152557373046875e-07
  max |g_opacity - ref| = 8.344650268554688e-07
  max |g_depths|        = 0.0
case B=2 G=10 H=6 W=7 K=4 tile=4 dense_work=840 tiled_work=812 ratio=1.03x
  max |g_means - ref|   = 5.960464477539062e-07
  max |g_conics - ref|  = 1.3709068298339844e-06
  max |g_colors - ref|  = 4.76837158203125e-07
  max |g_opacity - ref| = 1.430511474609375e-06
  max |g_depths|        = 0.0
reference_tiled_exactness: all checks passed
```

Interpretation:

- The tiled candidate lists reproduce the same front-K state as the dense scan on the tested cases.
- The saved-state backward matches dense autograd closely on the tested cases.
- The remaining validation gap is **Metal runtime validation on Apple hardware**.

## 2) Synthetic scan-work model

Ran:

- `tests/synthetic_scan_model.py`

Observed:

```text
clustered_512_6k:
  dense pixel*splat work    = 1,572,864,000
  tiled pixel*bin work      = 1,625,600
  reduction                 = 967.56x
  total tile refs           = 6,350
  mean contributors / tile  = 6.20
  p95 contributors / tile   = 1.00
  max contributors / tile   = 516
uniform_512_6k:
  dense pixel*splat work    = 1,572,864,000
  tiled pixel*bin work      = 1,627,904
  reduction                 = 966.19x
  total tile refs           = 6,359
  mean contributors / tile  = 6.21
  p95 contributors / tile   = 10.00
  max contributors / tile   = 16
```

Interpretation:

- These are **synthetic** scenes, not the user's real benchmark scenes.
- The numbers are meant to show the shape change, not to claim real end-to-end runtime wins.
- They confirm the intended effect: tiled capture scales with local candidate count, not global splat count.

## Honest status

This handoff fixes the clearly wrong `pixels × splats` capture design in source form and validates the sparse-candidate math in the CPU reference path.

It does **not** prove that the full Metal branch is now fast or build-clean on a real Mac. That still needs Apple-hardware measurement.
