# V9 Output-Planes Sorted Parity Diagnostic

This compares output-planes fixed eval against v8 forward eval on black-background overlap stacks.
Reverse/depth-descending order is the only fixed-blend candidate expected to match v8 color.
Even a color match does not produce `final_T`, `stop_count`, or backward replay state.

| status | size | G | format | order | <=1e-5 | max err | mean err | v8 ms | v9 ms | v8/v9 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ok | 16x32 | 2 | rgba32f | input | False | 0.25 | 0.00102274 | 17.930 | 4.468 | 4.013 |
| ok | 16x32 | 2 | rgba32f | ascending | False | 0.25 | 0.00102274 | 10.309 | 1.985 | 5.194 |
| ok | 16x32 | 2 | rgba32f | descending | True | 9.31323e-10 | 9.70128e-12 | 17.291 | 14.885 | 1.162 |
| ok | 16x32 | 2 | rgba16f | input | False | 0.25 | 0.00102332 | 36.204 | 1.391 | 26.032 |
| ok | 16x32 | 2 | rgba16f | ascending | False | 0.25 | 0.00102332 | 128.114 | 8.549 | 14.985 |
| ok | 16x32 | 2 | rgba16f | descending | False | 0.000203565 | 1.84308e-06 | 50.033 | 15.662 | 3.195 |
| ok | 16x32 | 16 | rgba32f | input | False | 0.385184 | 0.0419339 | 8.514 | 1.023 | 8.323 |
| ok | 16x32 | 16 | rgba32f | ascending | False | 0.385184 | 0.0419339 | 22.610 | 3.667 | 6.166 |
| ok | 16x32 | 16 | rgba32f | descending | True | 1.19209e-07 | 9.77646e-09 | 35.721 | 21.060 | 1.696 |
| ok | 16x32 | 16 | rgba16f | input | False | 0.38403 | 0.0420611 | 38.599 | 7.857 | 4.913 |
| ok | 16x32 | 16 | rgba16f | ascending | False | 0.38403 | 0.0420611 | 39.347 | 20.399 | 1.929 |
| ok | 16x32 | 16 | rgba16f | descending | False | 0.0012944 | 0.000360685 | 42.221 | 10.928 | 3.863 |
