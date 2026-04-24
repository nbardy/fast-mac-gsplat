# v7.1 reference validation report

These numbers were produced in this environment with the pure-Python reference implementation in `torch_gsplat_bridge_v71/reference.py`, compared against a dense autograd reference.

## Random case

- shape: `B=1, G=4, H=6, W=5`
- front_k = 1
  - image max error: `0.0`
  - means grad max error: `4.07e-10`
  - conics grad max error: `9.31e-10`
  - colors grad max error: `4.66e-10`
  - opacity grad max error: `7.45e-09`
  - overflow ratio: `0.80`
- front_k = 2
  - image max error: `0.0`
  - means grad max error: `4.07e-10`
  - conics grad max error: `9.31e-10`
  - colors grad max error: `4.66e-10`
  - opacity grad max error: `7.45e-09`
  - overflow ratio: `0.7333`
- front_k = 4
  - image max error: `0.0`
  - means grad max error: `4.07e-10`
  - conics grad max error: `9.31e-10`
  - colors grad max error: `4.66e-10`
  - opacity grad max error: `7.45e-09`
  - overflow ratio: `0.0`

## Clamp / overflow-heavy case

- shape: `B=1, G=5, H=4, W=4`
- front_k = 1
  - image max error: `0.0`
  - means grad max error: `3.73e-08`
  - conics grad max error: `5.59e-08`
  - colors grad max error: `2.24e-08`
  - opacity grad max error: `4.47e-08`
  - overflow ratio: `1.0`
- front_k = 3
  - image max error: `0.0`
  - means grad max error: `3.73e-08`
  - conics grad max error: `5.59e-08`
  - colors grad max error: `2.24e-08`
  - opacity grad max error: `4.47e-08`
  - overflow ratio: `1.0`

## Takeaway

The math in the new front-K / overflow split is internally consistent. What remains unvalidated here is the Apple-specific Objective-C++ / Metal bringup and the real performance profile on your target scenes.
