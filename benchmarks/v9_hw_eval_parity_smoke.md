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
| ok | tiny_single | 16x16 | 1 | True | True | 1.49012e-08 | 3.37726e-10 | 2.744 | 0.633 | 4.336 | single Gaussian; v8 depth sort is a no-op |
| ok | tiny_single | 64x64 | 1 | True | True | 1.49012e-08 | 2.11079e-11 | 1.995 | 0.430 | 4.635 | single Gaussian; v8 depth sort is a no-op |
| ok | grid_ordered | 16x16 | 1 | True | True | 1.86265e-09 | 7.1168e-11 | 1.774 | 0.376 | 4.717 | depths are monotonic with input order; black background |
| ok | grid_ordered | 16x16 | 16 | False | False | 0.0967607 | 0.0255689 | 1.660 | 0.287 | 5.778 | depths are monotonic with input order; black background; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | grid_ordered | 64x64 | 1 | True | True | 1.86265e-09 | 6.74068e-12 | 2.344 | 0.426 | 5.501 | depths are monotonic with input order; black background |
| ok | grid_ordered | 64x64 | 16 | False | False | 0.00247034 | 4.08272e-05 | 2.107 | 0.363 | 5.804 | depths are monotonic with input order; black background; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | overlap_ordered | 16x16 | 1 | True | True | 1.49012e-08 | 8.42192e-10 | 2.447 | 0.395 | 6.198 | overlapping splats with depths monotonic in input order |
| ok | overlap_ordered | 16x16 | 16 | False | False | 0.123525 | 0.0278566 | 2.163 | 0.369 | 5.869 | overlapping splats with depths monotonic in input order; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | overlap_ordered | 64x64 | 1 | True | True | 1.49012e-08 | 1.16652e-10 | 2.300 | 0.336 | 6.852 | overlapping splats with depths monotonic in input order |
| ok | overlap_ordered | 64x64 | 16 | False | False | 0.0579901 | 0.00166997 | 2.321 | 0.418 | 5.552 | overlapping splats with depths monotonic in input order; multi-splat diagnostic; current v9 blend order is not a v8 order guarantee |
| ok | depth_mismatch | 16x16 | 16 | False | True | 1.19209e-07 | 2.34383e-08 | 2.201 | 0.365 | 6.026 | intentional order diagnostic: v8 sorts by depth while v9 has no v8 order contract |
| ok | depth_mismatch | 64x64 | 16 | False | True | 8.9407e-08 | 2.12441e-09 | 2.530 | 0.428 | 5.912 | intentional order diagnostic: v8 sorts by depth while v9 has no v8 order contract |
