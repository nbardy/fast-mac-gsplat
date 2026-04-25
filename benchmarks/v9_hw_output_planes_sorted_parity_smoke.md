# V9 Output-Planes Sorted Parity Diagnostic

This compares output-planes fixed eval against v8 forward eval on black-background overlap stacks.
Reverse/depth-descending order is the only fixed-blend candidate expected to match v8 color.
Even a color match does not produce `final_T`, `stop_count`, or backward replay state.

| status | size | G | format | order | <=1e-5 | max err | mean err | v8 ms | v9 ms | v8/v9 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ok | 16x32 | 2 | rgba32f | input | False | 0.25 | 0.00102274 | 22.883 | 1.490 | 15.362 |
| ok | 16x32 | 2 | rgba32f | ascending | False | 0.25 | 0.00102274 | 10.468 | 5.598 | 1.870 |
| ok | 16x32 | 2 | rgba32f | descending | True | 9.31323e-10 | 9.70128e-12 | 9.430 | 2.167 | 4.351 |
| ok | 16x32 | 2 | rgba16f | input | False | 0.25 | 0.00102332 | 6.560 | 5.671 | 1.157 |
| ok | 16x32 | 2 | rgba16f | ascending | False | 0.25 | 0.00102332 | 22.938 | 9.006 | 2.547 |
| ok | 16x32 | 2 | rgba16f | descending | False | 0.000203565 | 1.84308e-06 | 8.177 | 4.012 | 2.038 |
| ok | 16x32 | 16 | rgba32f | input | False | 0.385184 | 0.0419339 | 9.730 | 0.684 | 14.221 |
| ok | 16x32 | 16 | rgba32f | ascending | False | 0.385184 | 0.0419339 | 12.323 | 20.702 | 0.595 |
| ok | 16x32 | 16 | rgba32f | descending | True | 1.19209e-07 | 9.77646e-09 | 6.134 | 1.492 | 4.111 |
| ok | 16x32 | 16 | rgba16f | input | False | 0.38403 | 0.0420611 | 7.540 | 1.136 | 6.638 |
| ok | 16x32 | 16 | rgba16f | ascending | False | 0.38403 | 0.0420611 | 6.660 | 2.419 | 2.754 |
| ok | 16x32 | 16 | rgba16f | descending | False | 0.0012944 | 0.000360685 | 5.862 | 1.473 | 3.980 |
| ok | 64x64 | 2 | rgba32f | input | False | 0.25 | 0.000127843 | 4.985 | 0.565 | 8.823 |
| ok | 64x64 | 2 | rgba32f | ascending | False | 0.25 | 0.000127843 | 5.392 | 8.526 | 0.632 |
| ok | 64x64 | 2 | rgba32f | descending | True | 9.31323e-10 | 1.21266e-12 | 24.600 | 9.843 | 2.499 |
| ok | 64x64 | 2 | rgba16f | input | False | 0.25 | 0.000127916 | 13.358 | 0.919 | 14.533 |
| ok | 64x64 | 2 | rgba16f | ascending | False | 0.25 | 0.000127916 | 11.451 | 9.291 | 1.232 |
| ok | 64x64 | 2 | rgba16f | descending | False | 0.000203565 | 2.30385e-07 | 46.329 | 4.089 | 11.330 |
| ok | 64x64 | 16 | rgba32f | input | False | 0.385184 | 0.00528164 | 17.113 | 0.863 | 19.819 |
| ok | 64x64 | 16 | rgba32f | ascending | False | 0.385184 | 0.00528164 | 23.103 | 3.273 | 7.059 |
| ok | 64x64 | 16 | rgba32f | descending | True | 1.19209e-07 | 1.32301e-09 | 6.326 | 2.053 | 3.081 |
| ok | 64x64 | 16 | rgba16f | input | False | 0.38403 | 0.00529879 | 5.920 | 0.915 | 6.471 |
| ok | 64x64 | 16 | rgba16f | ascending | False | 0.38403 | 0.00529879 | 19.158 | 1.963 | 9.761 |
| ok | 64x64 | 16 | rgba16f | descending | False | 0.0012944 | 4.80052e-05 | 19.937 | 4.267 | 4.672 |
