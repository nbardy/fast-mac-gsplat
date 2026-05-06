# Suggested benchmark plan for engineering

## Primary matrix
Run each case at:
- batch sizes: 1 / 2 / 4 / 8
- batch strategies: auto / flatten / serial
- tile sizes: 8 / 16 / 32
- chunk sizes: 32 / 64 / 128
- fast cap: 1024 / 2048
- eval forward and forward+backward

## Cases
1. `sparse_sigma_1_5`
2. `medium_sigma_3_8`
3. `heavy_sigma_8_24`
4. `center_hotspot`
5. `quadrant_cluster`
6. `overflow_stress`

## What to record
- mean / median / p95 wall time
- forward ms and backward ms separately
- total tile pairs
- p95 / max pairs per tile
- overflow tile count
- p95 / max stop count
- p95 stop ratio
- chosen batch chunk size

## Interpretation
- if eval is much faster than train, sorted-ID writeback is the right split
- if `flatten` beats `serial` at B=2/4 but loses at B=8, launch pressure is the limiting factor
- if p95 stop ratio is low but backward is still expensive, atomics / parameter bandwidth dominate
- if overflow count is nontrivial, fast cap or tile size needs adjustment
