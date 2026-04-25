# V9 HW Eval Parity vs V8 Forward

Comparison target: v9 fixed eval RGBA premultiplied RGB against v8 forward-eval RGB.
Validation readback happens after each native op returns; the native v9 render op itself does not read GPU data on CPU.

## Current v9 limitations

- eval-only; no backward path
- batch size 1 only
- expects already projected pixel-space means2d and conics
- no depth sort; multi-splat hardware blend order is not v8-equivalent
- no tile/imageblock path
- no transmittance early termination
- black transparent clear only; compare RGB against v8 black background
- direct path requires width * 16 bytes to be 256-byte aligned

## Rows

| status | case | size | G | comparable | <=1e-5 | max err | mean err | v8 median ms | v9 median ms | v8/v9 | notes |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ok | tiny_single | 16x16 | 1 | True | True | 1.49012e-08 | 3.37726e-10 | 4.952 | 0.521 | 9.507 | single Gaussian; v8 depth sort is a no-op |
| ok | tiny_single | 64x64 | 1 | True | True | 1.49012e-08 | 2.11079e-11 | 2.893 | 0.358 | 8.071 | single Gaussian; v8 depth sort is a no-op |
| ok | grid_ordered | 16x16 | 1 | True | True | 1.86265e-09 | 7.1168e-11 | 1.971 | 0.298 | 6.616 | depths are monotonic with input order; black background |
| ok | grid_ordered | 16x16 | 16 | False | False | 0.0967607 | 0.0255689 | 4.043 | 0.519 | 7.784 | depths are monotonic with input order; black background; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | grid_ordered | 64x64 | 1 | True | True | 1.86265e-09 | 6.74068e-12 | 4.553 | 0.425 | 10.719 | depths are monotonic with input order; black background |
| ok | grid_ordered | 64x64 | 16 | False | False | 0.00247034 | 4.08272e-05 | 4.123 | 0.427 | 9.657 | depths are monotonic with input order; black background; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | overlap_ordered | 16x16 | 1 | True | True | 1.49012e-08 | 8.42192e-10 | 4.448 | 0.671 | 6.630 | overlapping splats with depths monotonic in input order |
| ok | overlap_ordered | 16x16 | 16 | False | False | 0.123525 | 0.0278566 | 15.151 | 0.427 | 35.452 | overlapping splats with depths monotonic in input order; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | overlap_ordered | 64x64 | 1 | True | True | 1.49012e-08 | 1.16652e-10 | 4.322 | 0.372 | 11.632 | overlapping splats with depths monotonic in input order |
| ok | overlap_ordered | 64x64 | 16 | False | False | 0.0579901 | 0.00166997 | 4.341 | 0.389 | 11.173 | overlapping splats with depths monotonic in input order; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | depth_mismatch | 16x16 | 16 | False | True | 1.19209e-07 | 2.34383e-08 | 3.937 | 0.424 | 9.283 | intentional order diagnostic: v8 sorts by depth while v9 has no v8 order contract |
| ok | depth_mismatch | 64x64 | 16 | False | True | 8.9407e-08 | 2.12441e-09 | 6.229 | 0.498 | 12.515 | intentional order diagnostic: v8 sorts by depth while v9 has no v8 order contract |
