Benchmarking is not claimed from this environment because Apple Metal hardware is unavailable here.

The intended benchmark split for this handoff is:

- render pass time
- tile-bin build time
- capture pass time
- common backward time
- overflow replay time
- aux tensor movement time

The critical comparison versus old v7.1 is:

- old capture: `pixels × splats`
- new capture: `pixels × average_tile_bin_size`
