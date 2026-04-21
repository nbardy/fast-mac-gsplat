# Full Rasterizer Benchmark

Generated: 2026-04-21 23:51:44

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` is time delta, so negative means faster than v6 direct.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `120.0`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 512x512 | 1 | 512 | clustered_hot_tiles | forward | v6_direct | 6.648 |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_direct | 12.119 |
| 512x512 | 1 | 512 | layered_depth | forward | v6_direct | 7.069 |
| 512x512 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_direct | 11.098 |
| 512x512 | 1 | 512 | microbench_uniform_random | forward | v6_direct | 6.168 |
| 512x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_direct | 10.330 |
| 512x512 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | 11.828 |
| 512x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | 15.370 |
| 512x512 | 1 | 512 | sparse_screen | forward | v6_direct | 6.741 |
| 512x512 | 1 | 512 | sparse_screen | forward_backward | v6_direct | 10.506 |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_direct | 8.192 |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | 13.677 |
| 512x512 | 1 | 2048 | layered_depth | forward | v6_direct | 7.230 |
| 512x512 | 1 | 2048 | layered_depth | forward_backward | v6_direct | 11.673 |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward | v6_direct | 6.814 |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_direct | 12.367 |
| 512x512 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | 10.025 |
| 512x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | 23.550 |
| 512x512 | 1 | 2048 | sparse_screen | forward | v6_direct | 6.674 |
| 512x512 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | 10.708 |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | 40.827 |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 85.346 |
| 512x512 | 1 | 65536 | layered_depth | forward | v6_direct | 165.182 |
| 512x512 | 1 | 65536 | layered_depth | forward_backward | v6_auto | 135.544 |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | 11.833 |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_auto | 19.485 |
| 512x512 | 1 | 65536 | overflow_adversarial | forward | v6_direct | 787.045 |
| 512x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_direct | 926.269 |
| 512x512 | 1 | 65536 | sparse_screen | forward | v6_direct | 52.583 |
| 512x512 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | 76.442 |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | 7.995 |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 17.455 |
| 512x512 | 4 | 512 | layered_depth | forward | v6_auto | 8.209 |
| 512x512 | 4 | 512 | layered_depth | forward_backward | v6_direct | 13.733 |
| 512x512 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | 7.268 |
| 512x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 15.725 |
| 512x512 | 4 | 512 | overflow_adversarial | forward | v6_direct | 9.651 |
| 512x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | 26.094 |
| 512x512 | 4 | 512 | sparse_screen | forward | v6_direct | 7.812 |
| 512x512 | 4 | 512 | sparse_screen | forward_backward | v6_auto | 11.990 |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | 10.370 |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | 20.904 |
| 512x512 | 4 | 2048 | layered_depth | forward | v6_direct | 10.732 |
| 512x512 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | 19.795 |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward | v6_auto | 9.881 |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 22.116 |
| 512x512 | 4 | 2048 | overflow_adversarial | forward | v6_direct | 15.218 |
| 512x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | 57.657 |
| 512x512 | 4 | 2048 | sparse_screen | forward | v6_auto | 6.909 |
| 512x512 | 4 | 2048 | sparse_screen | forward_backward | v6_auto | 14.573 |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | 125.890 |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | 243.227 |
| 512x512 | 4 | 65536 | layered_depth | forward | v6_direct | 391.225 |
| 512x512 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | 520.653 |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward | v6_direct | 17.689 |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 59.308 |
| 512x512 | 4 | 65536 | overflow_adversarial | forward | v6_direct | 3269.691 |
| 512x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | 5627.223 |
| 512x512 | 4 | 65536 | sparse_screen | forward | v6_direct | 292.527 |
| 512x512 | 4 | 65536 | sparse_screen | forward_backward | v6_direct | 217.652 |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward | v6_direct | 7.843 |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_direct | 12.509 |
| 1024x512 | 1 | 512 | layered_depth | forward | v6_upgrade_direct | 7.138 |
| 1024x512 | 1 | 512 | layered_depth | forward_backward | v6_auto | 13.356 |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | 6.248 |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | 10.909 |
| 1024x512 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | 8.252 |
| 1024x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_direct | 15.080 |
| 1024x512 | 1 | 512 | sparse_screen | forward | v6_direct | 7.801 |
| 1024x512 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | 10.571 |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_direct | 6.880 |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 12.077 |
| 1024x512 | 1 | 2048 | layered_depth | forward | v6_direct | 6.864 |
| 1024x512 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | 12.152 |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward | v6_auto | 7.465 |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | 11.156 |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | 8.878 |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | 20.583 |
| 1024x512 | 1 | 2048 | sparse_screen | forward | v6_upgrade_direct | 6.872 |
| 1024x512 | 1 | 2048 | sparse_screen | forward_backward | v6_direct | 10.733 |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | 44.999 |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 86.063 |
| 1024x512 | 1 | 65536 | layered_depth | forward | v6_direct | 187.631 |
| 1024x512 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | 183.656 |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | 8.917 |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_auto | 28.573 |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward | v6_direct | 800.767 |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | 939.235 |
| 1024x512 | 1 | 65536 | sparse_screen | forward | v6_upgrade_direct | 50.218 |
| 1024x512 | 1 | 65536 | sparse_screen | forward_backward | v6_auto | 79.653 |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | 8.488 |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_auto | 18.383 |
| 1024x512 | 4 | 512 | layered_depth | forward | v6_auto | 7.661 |
| 1024x512 | 4 | 512 | layered_depth | forward_backward | v6_direct | 19.735 |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward | v6_direct | 8.965 |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 14.630 |
| 1024x512 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_direct | 10.535 |
| 1024x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_direct | 26.375 |
| 1024x512 | 4 | 512 | sparse_screen | forward | v6_upgrade_direct | 7.871 |
| 1024x512 | 4 | 512 | sparse_screen | forward_backward | v6_direct | 13.875 |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_auto | 9.368 |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | 18.847 |
| 1024x512 | 4 | 2048 | layered_depth | forward | v6_upgrade_direct | 10.418 |
| 1024x512 | 4 | 2048 | layered_depth | forward_backward | v6_direct | 20.767 |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | 9.220 |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | 19.096 |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward | v6_direct | 14.936 |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | 53.859 |
| 1024x512 | 4 | 2048 | sparse_screen | forward | v6_direct | 9.289 |
| 1024x512 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | 16.019 |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | 161.056 |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 297.145 |
| 1024x512 | 4 | 65536 | layered_depth | forward | v6_auto | 836.914 |
| 1024x512 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | 674.355 |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | 20.585 |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_direct | 87.780 |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | 3267.516 |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | 4952.709 |
| 1024x512 | 4 | 65536 | sparse_screen | forward | v6_direct | 196.378 |
| 1024x512 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | 235.174 |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | 7.708 |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 15.599 |
| 1920x1080 | 1 | 512 | layered_depth | forward | v6_direct | 7.430 |
| 1920x1080 | 1 | 512 | layered_depth | forward_backward | v6_direct | 16.770 |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward | v6_direct | 6.685 |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward_backward | v6_direct | 13.586 |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | 8.437 |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward_backward | v6_auto | 20.675 |
| 1920x1080 | 1 | 512 | sparse_screen | forward | v6_direct | 6.891 |
| 1920x1080 | 1 | 512 | sparse_screen | forward_backward | v6_direct | 11.955 |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward | v6_direct | 8.658 |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 16.177 |
| 1920x1080 | 1 | 2048 | layered_depth | forward | v6_upgrade_direct | 7.617 |
| 1920x1080 | 1 | 2048 | layered_depth | forward_backward | v6_direct | 14.567 |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward | v6_direct | 6.824 |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_direct | 11.979 |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | 9.853 |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | 22.775 |
| 1920x1080 | 1 | 2048 | sparse_screen | forward | v6_direct | 8.060 |
| 1920x1080 | 1 | 2048 | sparse_screen | forward_backward | v6_auto | 15.688 |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward | v6_direct | 118.825 |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | 103.832 |
| 1920x1080 | 1 | 65536 | layered_depth | forward | v6_upgrade_direct | 164.657 |
| 1920x1080 | 1 | 65536 | layered_depth | forward_backward | v6_direct | 182.487 |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward | v6_auto | 11.181 |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | 33.525 |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward | v6_auto | 464.530 |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward_backward | v6_direct | 938.766 |
| 1920x1080 | 1 | 65536 | sparse_screen | forward | v6_upgrade_direct | 44.779 |
| 1920x1080 | 1 | 65536 | sparse_screen | forward_backward | v6_direct | 57.720 |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | 9.296 |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_direct | 27.220 |
| 1920x1080 | 4 | 512 | layered_depth | forward | v6_upgrade_direct | 7.303 |
| 1920x1080 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_direct | 28.533 |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward | v6_direct | 9.670 |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward_backward | v6_direct | 24.362 |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward | v6_direct | 9.723 |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward_backward | v6_direct | 43.468 |
| 1920x1080 | 4 | 512 | sparse_screen | forward | v6_direct | 9.064 |
| 1920x1080 | 4 | 512 | sparse_screen | forward_backward | v6_direct | 25.446 |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward | v6_direct | 12.976 |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 30.957 |
| 1920x1080 | 4 | 2048 | layered_depth | forward | v6_upgrade_direct | 8.863 |
| 1920x1080 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | 35.483 |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | 10.974 |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | 29.660 |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward | v6_direct | 15.712 |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | 67.194 |
| 1920x1080 | 4 | 2048 | sparse_screen | forward | v6_upgrade_direct | 10.898 |
| 1920x1080 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | 28.393 |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward | v6_auto | 388.241 |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | 381.166 |
| 1920x1080 | 4 | 65536 | layered_depth | forward | v6_upgrade_auto | 726.508 |
| 1920x1080 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | 500.336 |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward | v6_auto | 24.006 |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_auto | 112.713 |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | 1851.890 |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward_backward | v6_direct | 5555.706 |
| 1920x1080 | 4 | 65536 | sparse_screen | forward | v6_direct | 97.464 |
| 1920x1080 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | 148.342 |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward | v6_direct | 11.414 |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_direct | 34.896 |
| 4096x4096 | 1 | 512 | layered_depth | forward | v6_direct | 10.435 |
| 4096x4096 | 1 | 512 | layered_depth | forward_backward | v6_direct | 39.438 |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | 9.681 |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 35.972 |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | 13.734 |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward_backward | v6_direct | 39.598 |
| 4096x4096 | 1 | 512 | sparse_screen | forward | v6_direct | 11.487 |
| 4096x4096 | 1 | 512 | sparse_screen | forward_backward | v6_direct | 34.643 |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | 10.763 |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | 39.627 |
| 4096x4096 | 1 | 2048 | layered_depth | forward | v6_direct | 9.777 |
| 4096x4096 | 1 | 2048 | layered_depth | forward_backward | v6_auto | 36.880 |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | 10.615 |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 36.937 |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward | v6_direct | 12.266 |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward_backward | v6_direct | 44.432 |
| 4096x4096 | 1 | 2048 | sparse_screen | forward | v6_direct | 10.900 |
| 4096x4096 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | 35.724 |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | 127.332 |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | 176.944 |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v6_direct | 14.572 |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | 74.644 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | 11.752 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 62.657 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v6_direct | 585.299 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v6_auto | 1089.448 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v6_direct | 10.650 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v6_direct | 52.268 |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | 15.853 |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_direct | 130.018 |
| 4096x4096 | 4 | 512 | layered_depth | forward | v6_direct | 13.335 |
| 4096x4096 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_direct | 128.917 |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward | v6_direct | 14.993 |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward_backward | v6_direct | 129.257 |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward | v6_direct | 19.124 |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward_backward | v6_direct | 143.334 |
| 4096x4096 | 4 | 512 | sparse_screen | forward | v6_direct | 15.280 |
| 4096x4096 | 4 | 512 | sparse_screen | forward_backward | v6_direct | 128.044 |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward | v6_direct | 15.673 |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | 129.901 |
| 4096x4096 | 4 | 2048 | layered_depth | forward | v6_upgrade_direct | 17.185 |
| 4096x4096 | 4 | 2048 | layered_depth | forward_backward | v6_direct | 132.490 |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | 15.021 |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_direct | 128.799 |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward | v6_direct | 23.391 |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward_backward | v6_direct | 167.694 |
| 4096x4096 | 4 | 2048 | sparse_screen | forward | v6_direct | 14.201 |
| 4096x4096 | 4 | 2048 | sparse_screen | forward_backward | v6_direct | 127.143 |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | 819.863 |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | 684.783 |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v6_direct | 39.768 |
| 4096x4096 | 4 | 65536 | layered_depth | forward_backward | v6_auto | 269.955 |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v6_direct | 33.943 |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_auto | 229.856 |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v6_auto | 4358.096 |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | 7555.238 |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v6_direct | 25.037 |
| 4096x4096 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | 188.677 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Slower Than Best | % vs Torch | % vs v6 Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 512 | microbench_uniform_random | forward | v6_direct | ok | 6.168 | 6.168 | 6.168 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 512 | microbench_uniform_random | forward | v6_auto | ok | 8.226 | 8.226 | 8.226 | 0.000 | +33.4% |  | +33.4% |  |
| 512x512 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 9.280 | 9.280 | 9.279 | 0.000 | +50.4% |  | +50.4% |  |
| 512x512 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 6.944 | 6.944 | 6.944 | 0.000 | +12.6% |  | +12.6% |  |
| 512x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 10.330 | 10.330 | 7.064 | 3.267 | +0.0% |  |  |  |
| 512x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 12.437 | 12.437 | 9.047 | 3.389 | +20.4% |  | +20.4% |  |
| 512x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 12.455 | 12.455 | 8.880 | 3.576 | +20.6% |  | +20.6% |  |
| 512x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 12.821 | 12.821 | 9.601 | 3.221 | +24.1% |  | +24.1% |  |
| 512x512 | 1 | 512 | sparse_screen | forward | v6_direct | ok | 6.741 | 6.741 | 6.741 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 512 | sparse_screen | forward | v6_auto | ok | 7.398 | 7.398 | 7.398 | 0.000 | +9.7% |  | +9.7% |  |
| 512x512 | 1 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 6.803 | 6.803 | 6.803 | 0.000 | +0.9% |  | +0.9% |  |
| 512x512 | 1 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 9.713 | 9.713 | 9.713 | 0.000 | +44.1% |  | +44.1% |  |
| 512x512 | 1 | 512 | sparse_screen | forward_backward | v6_direct | ok | 10.506 | 10.506 | 6.626 | 3.879 | +0.0% |  |  |  |
| 512x512 | 1 | 512 | sparse_screen | forward_backward | v6_auto | ok | 11.613 | 11.613 | 8.220 | 3.392 | +10.5% |  | +10.5% |  |
| 512x512 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 11.104 | 11.104 | 7.655 | 3.449 | +5.7% |  | +5.7% |  |
| 512x512 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 15.694 | 15.694 | 12.097 | 3.597 | +49.4% |  | +49.4% |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 6.648 | 6.648 | 6.647 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 11.221 | 11.221 | 11.221 | 0.000 | +68.8% |  | +68.8% |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 8.042 | 8.042 | 8.042 | 0.000 | +21.0% |  | +21.0% |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 9.153 | 9.153 | 9.153 | 0.000 | +37.7% |  | +37.7% |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 12.119 | 12.119 | 7.265 | 4.854 | +0.0% |  |  |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 12.738 | 12.738 | 10.291 | 2.447 | +5.1% |  | +5.1% |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 14.630 | 14.630 | 10.420 | 4.210 | +20.7% |  | +20.7% |  |
| 512x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 14.753 | 14.753 | 10.817 | 3.936 | +21.7% |  | +21.7% |  |
| 512x512 | 1 | 512 | layered_depth | forward | v6_direct | ok | 7.069 | 7.069 | 7.069 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 512 | layered_depth | forward | v6_auto | ok | 8.342 | 8.342 | 8.342 | 0.000 | +18.0% |  | +18.0% |  |
| 512x512 | 1 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 7.647 | 7.647 | 7.647 | 0.000 | +8.2% |  | +8.2% |  |
| 512x512 | 1 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 9.825 | 9.825 | 9.824 | 0.001 | +39.0% |  | +39.0% |  |
| 512x512 | 1 | 512 | layered_depth | forward_backward | v6_direct | ok | 13.109 | 13.109 | 7.799 | 5.310 | +18.1% |  |  |  |
| 512x512 | 1 | 512 | layered_depth | forward_backward | v6_auto | ok | 12.362 | 12.362 | 9.045 | 3.317 | +11.4% |  | -5.7% |  |
| 512x512 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 11.098 | 11.098 | 7.237 | 3.861 | +0.0% |  | -15.3% |  |
| 512x512 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 17.658 | 17.658 | 13.218 | 4.440 | +59.1% |  | +34.7% |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward | v6_direct | ok | 12.353 | 12.353 | 12.352 | 0.000 | +4.4% |  |  |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward | v6_auto | ok | 12.962 | 12.962 | 12.962 | 0.000 | +9.6% |  | +4.9% |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 11.828 | 11.828 | 11.827 | 0.000 | +0.0% |  | -4.3% |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 12.803 | 12.803 | 12.803 | 0.000 | +8.2% |  | +3.6% |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 18.252 | 18.252 | 11.941 | 6.311 | +18.7% |  |  |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 17.594 | 17.594 | 10.812 | 6.782 | +14.5% |  | -3.6% |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 15.370 | 15.370 | 9.463 | 5.907 | +0.0% |  | -15.8% |  |
| 512x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 21.703 | 21.703 | 13.720 | 7.983 | +41.2% |  | +18.9% |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward | v6_direct | ok | 7.340 | 7.340 | 7.339 | 0.000 | +1.0% |  |  |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward | v6_auto | ok | 10.530 | 10.530 | 10.530 | 0.000 | +44.9% |  | +43.5% |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 7.745 | 7.745 | 7.744 | 0.000 | +6.6% |  | +5.5% |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 7.268 | 7.268 | 7.267 | 0.001 | +0.0% |  | -1.0% |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 19.842 | 19.842 | 10.050 | 9.791 | +26.2% |  |  |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 17.351 | 17.351 | 9.331 | 8.020 | +10.3% |  | -12.6% |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 15.725 | 15.725 | 8.002 | 7.724 | +0.0% |  | -20.7% |  |
| 512x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 17.527 | 17.527 | 9.741 | 7.786 | +11.5% |  | -11.7% |  |
| 512x512 | 4 | 512 | sparse_screen | forward | v6_direct | ok | 7.812 | 7.812 | 7.811 | 0.000 | +0.0% |  |  |  |
| 512x512 | 4 | 512 | sparse_screen | forward | v6_auto | ok | 8.200 | 8.200 | 8.200 | 0.000 | +5.0% |  | +5.0% |  |
| 512x512 | 4 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 9.240 | 9.240 | 9.240 | 0.000 | +18.3% |  | +18.3% |  |
| 512x512 | 4 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 10.500 | 10.500 | 10.500 | 0.000 | +34.4% |  | +34.4% |  |
| 512x512 | 4 | 512 | sparse_screen | forward_backward | v6_direct | ok | 16.377 | 16.377 | 8.031 | 8.346 | +36.6% |  |  |  |
| 512x512 | 4 | 512 | sparse_screen | forward_backward | v6_auto | ok | 11.990 | 11.990 | 7.116 | 4.874 | +0.0% |  | -26.8% |  |
| 512x512 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 14.660 | 14.660 | 7.893 | 6.767 | +22.3% |  | -10.5% |  |
| 512x512 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 18.184 | 18.184 | 12.671 | 5.513 | +51.7% |  | +11.0% |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 7.995 | 7.995 | 7.994 | 0.000 | +0.0% |  |  |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 9.484 | 9.484 | 9.484 | 0.000 | +18.6% |  | +18.6% |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 8.885 | 8.885 | 8.885 | 0.000 | +11.1% |  | +11.1% |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 11.441 | 11.441 | 11.441 | 0.000 | +43.1% |  | +43.1% |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 20.172 | 20.172 | 9.241 | 10.930 | +15.6% |  |  |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 17.759 | 17.759 | 12.148 | 5.610 | +1.7% |  | -12.0% |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 17.455 | 17.455 | 7.855 | 9.600 | +0.0% |  | -13.5% |  |
| 512x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 19.664 | 19.664 | 13.457 | 6.207 | +12.7% |  | -2.5% |  |
| 512x512 | 4 | 512 | layered_depth | forward | v6_direct | ok | 9.034 | 9.034 | 9.034 | 0.000 | +10.0% |  |  |  |
| 512x512 | 4 | 512 | layered_depth | forward | v6_auto | ok | 8.209 | 8.209 | 8.209 | 0.000 | +0.0% |  | -9.1% |  |
| 512x512 | 4 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 8.312 | 8.312 | 8.312 | 0.000 | +1.2% |  | -8.0% |  |
| 512x512 | 4 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 12.484 | 12.484 | 12.483 | 0.000 | +52.1% |  | +38.2% |  |
| 512x512 | 4 | 512 | layered_depth | forward_backward | v6_direct | ok | 13.733 | 13.733 | 6.858 | 6.875 | +0.0% |  |  |  |
| 512x512 | 4 | 512 | layered_depth | forward_backward | v6_auto | ok | 18.962 | 18.962 | 9.229 | 9.734 | +38.1% |  | +38.1% |  |
| 512x512 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 17.819 | 17.819 | 8.259 | 9.560 | +29.8% |  | +29.8% |  |
| 512x512 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 20.672 | 20.672 | 13.063 | 7.609 | +50.5% |  | +50.5% |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward | v6_direct | ok | 9.651 | 9.651 | 9.651 | 0.001 | +0.0% |  |  |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward | v6_auto | ok | 10.558 | 10.558 | 10.558 | 0.000 | +9.4% |  | +9.4% |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 13.480 | 13.480 | 13.480 | 0.000 | +39.7% |  | +39.7% |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 16.432 | 16.432 | 16.432 | 0.000 | +70.3% |  | +70.3% |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 33.399 | 33.399 | 11.042 | 22.357 | +28.0% |  |  |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 26.523 | 26.523 | 9.415 | 17.108 | +1.6% |  | -20.6% |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 26.957 | 26.957 | 9.347 | 17.610 | +3.3% |  | -19.3% |  |
| 512x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 26.094 | 26.094 | 9.431 | 16.664 | +0.0% |  | -21.9% |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 6.814 | 6.814 | 6.814 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 7.858 | 7.858 | 7.858 | 0.000 | +15.3% |  | +15.3% |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 8.277 | 8.277 | 8.276 | 0.000 | +21.5% |  | +21.5% |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 7.930 | 7.930 | 7.930 | 0.000 | +16.4% |  | +16.4% |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 12.367 | 12.367 | 6.664 | 5.702 | +0.0% |  |  |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 15.163 | 15.163 | 8.377 | 6.785 | +22.6% |  | +22.6% |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 13.110 | 13.110 | 8.167 | 4.943 | +6.0% |  | +6.0% |  |
| 512x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 14.331 | 14.331 | 7.919 | 6.412 | +15.9% |  | +15.9% |  |
| 512x512 | 1 | 2048 | sparse_screen | forward | v6_direct | ok | 6.674 | 6.674 | 6.674 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 2048 | sparse_screen | forward | v6_auto | ok | 8.544 | 8.544 | 8.543 | 0.000 | +28.0% |  | +28.0% |  |
| 512x512 | 1 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 8.776 | 8.776 | 8.776 | 0.000 | +31.5% |  | +31.5% |  |
| 512x512 | 1 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 11.739 | 11.739 | 11.739 | 0.000 | +75.9% |  | +75.9% |  |
| 512x512 | 1 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 13.809 | 13.809 | 8.803 | 5.006 | +29.0% |  |  |  |
| 512x512 | 1 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 13.635 | 13.635 | 8.195 | 5.440 | +27.3% |  | -1.3% |  |
| 512x512 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 10.708 | 10.708 | 8.340 | 2.368 | +0.0% |  | -22.5% |  |
| 512x512 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 14.643 | 14.643 | 11.134 | 3.509 | +36.8% |  | +6.0% |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 8.192 | 8.192 | 8.191 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 12.720 | 12.720 | 12.719 | 0.000 | +55.3% |  | +55.3% |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 9.357 | 9.357 | 9.357 | 0.000 | +14.2% |  | +14.2% |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 10.561 | 10.561 | 10.560 | 0.000 | +28.9% |  | +28.9% |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 14.270 | 14.270 | 7.808 | 6.462 | +4.3% |  |  |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 13.677 | 13.677 | 10.045 | 3.631 | +0.0% |  | -4.2% |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 14.144 | 14.144 | 8.162 | 5.982 | +3.4% |  | -0.9% |  |
| 512x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 14.752 | 14.752 | 10.245 | 4.507 | +7.9% |  | +3.4% |  |
| 512x512 | 1 | 2048 | layered_depth | forward | v6_direct | ok | 7.230 | 7.230 | 7.230 | 0.000 | +0.0% |  |  |  |
| 512x512 | 1 | 2048 | layered_depth | forward | v6_auto | ok | 9.404 | 9.404 | 9.404 | 0.000 | +30.1% |  | +30.1% |  |
| 512x512 | 1 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 9.138 | 9.138 | 9.138 | 0.000 | +26.4% |  | +26.4% |  |
| 512x512 | 1 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 10.922 | 10.922 | 10.922 | 0.000 | +51.1% |  | +51.1% |  |
| 512x512 | 1 | 2048 | layered_depth | forward_backward | v6_direct | ok | 11.673 | 11.673 | 6.979 | 4.694 | +0.0% |  |  |  |
| 512x512 | 1 | 2048 | layered_depth | forward_backward | v6_auto | ok | 16.096 | 16.096 | 9.427 | 6.669 | +37.9% |  | +37.9% |  |
| 512x512 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 14.509 | 14.509 | 9.155 | 5.354 | +24.3% |  | +24.3% |  |
| 512x512 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 14.330 | 14.330 | 11.021 | 3.309 | +22.8% |  | +22.8% |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward | v6_direct | ok | 11.502 | 11.502 | 11.502 | 0.000 | +14.7% |  |  |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward | v6_auto | ok | 11.539 | 11.539 | 11.538 | 0.000 | +15.1% |  | +0.3% |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 10.643 | 10.643 | 10.643 | 0.000 | +6.2% |  | -7.5% |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 10.025 | 10.025 | 10.024 | 0.000 | +0.0% |  | -12.8% |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 25.051 | 25.051 | 10.761 | 14.289 | +6.4% |  |  |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 24.432 | 24.432 | 11.685 | 12.747 | +3.7% |  | -2.5% |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 23.550 | 23.550 | 11.629 | 11.920 | +0.0% |  | -6.0% |  |
| 512x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 26.195 | 26.195 | 12.820 | 13.375 | +11.2% |  | +4.6% |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 10.872 | 10.872 | 10.871 | 0.000 | +10.0% |  |  |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 9.881 | 9.881 | 9.881 | 0.000 | +0.0% |  | -9.1% |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 10.688 | 10.688 | 10.687 | 0.000 | +8.2% |  | -1.7% |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 12.232 | 12.232 | 12.231 | 0.001 | +23.8% |  | +12.5% |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 33.234 | 33.234 | 12.646 | 20.588 | +50.3% |  |  |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 29.076 | 29.076 | 13.229 | 15.847 | +31.5% |  | -12.5% |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 22.116 | 22.116 | 9.571 | 12.545 | +0.0% |  | -33.5% |  |
| 512x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 24.397 | 24.397 | 11.891 | 12.506 | +10.3% |  | -26.6% |  |
| 512x512 | 4 | 2048 | sparse_screen | forward | v6_direct | ok | 8.166 | 8.166 | 8.166 | 0.000 | +18.2% |  |  |  |
| 512x512 | 4 | 2048 | sparse_screen | forward | v6_auto | ok | 6.909 | 6.909 | 6.909 | 0.000 | +0.0% |  | -15.4% |  |
| 512x512 | 4 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 8.982 | 8.982 | 8.981 | 0.000 | +30.0% |  | +10.0% |  |
| 512x512 | 4 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 11.971 | 11.971 | 11.970 | 0.000 | +73.2% |  | +46.6% |  |
| 512x512 | 4 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 15.730 | 15.730 | 7.874 | 7.856 | +7.9% |  |  |  |
| 512x512 | 4 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 14.573 | 14.573 | 7.894 | 6.679 | +0.0% |  | -7.4% |  |
| 512x512 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 19.285 | 19.285 | 8.098 | 11.188 | +32.3% |  | +22.6% |  |
| 512x512 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 23.599 | 23.599 | 12.118 | 11.481 | +61.9% |  | +50.0% |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 10.379 | 10.379 | 10.379 | 0.000 | +0.1% |  |  |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 14.121 | 14.121 | 14.121 | 0.000 | +36.2% |  | +36.1% |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 10.370 | 10.370 | 10.370 | 0.000 | +0.0% |  | -0.1% |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 12.603 | 12.603 | 12.602 | 0.000 | +21.5% |  | +21.4% |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 20.904 | 20.904 | 9.804 | 11.101 | +0.0% |  |  |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 22.822 | 22.822 | 11.267 | 11.555 | +9.2% |  | +9.2% |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 24.756 | 24.756 | 10.485 | 14.271 | +18.4% |  | +18.4% |  |
| 512x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 23.366 | 23.366 | 11.374 | 11.992 | +11.8% |  | +11.8% |  |
| 512x512 | 4 | 2048 | layered_depth | forward | v6_direct | ok | 10.732 | 10.732 | 10.732 | 0.000 | +0.0% |  |  |  |
| 512x512 | 4 | 2048 | layered_depth | forward | v6_auto | ok | 11.865 | 11.865 | 11.865 | 0.000 | +10.6% |  | +10.6% |  |
| 512x512 | 4 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 11.922 | 11.922 | 11.921 | 0.000 | +11.1% |  | +11.1% |  |
| 512x512 | 4 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 11.852 | 11.852 | 11.852 | 0.000 | +10.4% |  | +10.4% |  |
| 512x512 | 4 | 2048 | layered_depth | forward_backward | v6_direct | ok | 35.357 | 35.357 | 13.469 | 21.888 | +78.6% |  |  |  |
| 512x512 | 4 | 2048 | layered_depth | forward_backward | v6_auto | ok | 28.400 | 28.400 | 11.145 | 17.255 | +43.5% |  | -19.7% |  |
| 512x512 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 28.568 | 28.568 | 11.187 | 17.382 | +44.3% |  | -19.2% |  |
| 512x512 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 19.795 | 19.795 | 13.250 | 6.545 | +0.0% |  | -44.0% |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward | v6_direct | ok | 15.218 | 15.218 | 15.217 | 0.000 | +0.0% |  |  |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward | v6_auto | ok | 16.518 | 16.518 | 16.518 | 0.001 | +8.5% |  | +8.5% |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 21.953 | 21.953 | 21.952 | 0.000 | +44.3% |  | +44.3% |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 20.336 | 20.336 | 20.335 | 0.001 | +33.6% |  | +33.6% |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 57.689 | 57.689 | 16.562 | 41.127 | +0.1% |  |  |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 58.958 | 58.958 | 17.960 | 40.998 | +2.3% |  | +2.2% |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 59.752 | 59.752 | 17.318 | 42.435 | +3.6% |  | +3.6% |  |
| 512x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 57.657 | 57.657 | 16.532 | 41.125 | +0.0% |  | -0.1% |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 16.115 | 16.115 | 16.115 | 0.000 | +36.2% |  |  |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 15.542 | 15.542 | 15.542 | 0.000 | +31.3% |  | -3.6% |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 14.544 | 14.544 | 14.544 | 0.000 | +22.9% |  | -9.7% |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 11.833 | 11.833 | 11.833 | 0.000 | +0.0% |  | -26.6% |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 24.742 | 24.742 | 9.720 | 15.022 | +27.0% |  |  |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 19.485 | 19.485 | 9.102 | 10.383 | +0.0% |  | -21.2% |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 27.035 | 27.035 | 11.613 | 15.422 | +38.7% |  | +9.3% |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 21.131 | 21.131 | 10.673 | 10.458 | +8.4% |  | -14.6% |  |
| 512x512 | 1 | 65536 | sparse_screen | forward | v6_direct | ok | 52.583 | 52.583 | 52.582 | 0.001 | +0.0% |  |  |  |
| 512x512 | 1 | 65536 | sparse_screen | forward | v6_auto | ok | 83.323 | 83.323 | 83.322 | 0.001 | +58.5% |  | +58.5% |  |
| 512x512 | 1 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 79.818 | 79.818 | 79.818 | 0.000 | +51.8% |  | +51.8% |  |
| 512x512 | 1 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 88.808 | 88.808 | 88.806 | 0.001 | +68.9% |  | +68.9% |  |
| 512x512 | 1 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 79.866 | 79.866 | 52.882 | 26.984 | +4.5% |  |  |  |
| 512x512 | 1 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 79.048 | 79.048 | 54.813 | 24.235 | +3.4% |  | -1.0% |  |
| 512x512 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 76.442 | 76.442 | 51.638 | 24.804 | +0.0% |  | -4.3% |  |
| 512x512 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 100.985 | 100.985 | 61.916 | 39.069 | +32.1% |  | +26.4% |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 78.267 | 78.267 | 78.266 | 0.001 | +91.7% |  |  |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 73.083 | 73.083 | 73.082 | 0.001 | +79.0% |  | -6.6% |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 40.827 | 40.827 | 40.827 | 0.000 | +0.0% |  | -47.8% |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 75.829 | 75.829 | 75.828 | 0.000 | +85.7% |  | -3.1% |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 86.556 | 86.556 | 39.110 | 47.446 | +1.4% |  |  |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 86.877 | 86.877 | 40.291 | 46.585 | +1.8% |  | +0.4% |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 85.346 | 85.346 | 38.340 | 47.006 | +0.0% |  | -1.4% |  |
| 512x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 86.470 | 86.470 | 39.625 | 46.845 | +1.3% |  | -0.1% |  |
| 512x512 | 1 | 65536 | layered_depth | forward | v6_direct | ok | 165.182 | 165.182 | 165.182 | 0.001 | +0.0% |  |  |  |
| 512x512 | 1 | 65536 | layered_depth | forward | v6_auto | ok | 169.682 | 169.682 | 169.681 | 0.001 | +2.7% |  | +2.7% |  |
| 512x512 | 1 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 182.367 | 182.367 | 182.367 | 0.001 | +10.4% |  | +10.4% |  |
| 512x512 | 1 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 188.881 | 188.881 | 188.880 | 0.001 | +14.3% |  | +14.3% |  |
| 512x512 | 1 | 65536 | layered_depth | forward_backward | v6_direct | ok | 170.114 | 170.114 | 120.538 | 49.576 | +25.5% |  |  |  |
| 512x512 | 1 | 65536 | layered_depth | forward_backward | v6_auto | ok | 135.544 | 135.544 | 103.361 | 32.183 | +0.0% |  | -20.3% |  |
| 512x512 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 137.568 | 137.568 | 104.318 | 33.250 | +1.5% |  | -19.1% |  |
| 512x512 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 139.011 | 139.011 | 104.998 | 34.013 | +2.6% |  | -18.3% |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward | v6_direct | ok | 787.045 | 787.045 | 787.044 | 0.001 | +0.0% |  |  |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward | v6_auto | ok | 817.155 | 817.155 | 817.154 | 0.001 | +3.8% |  | +3.8% |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 819.141 | 819.141 | 819.140 | 0.001 | +4.1% |  | +4.1% |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 906.084 | 906.084 | 906.083 | 0.001 | +15.1% |  | +15.1% |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 926.269 | 926.269 | 462.720 | 463.549 | +0.0% |  |  |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 940.699 | 940.699 | 471.751 | 468.949 | +1.6% |  | +1.6% |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 944.825 | 944.825 | 475.049 | 469.775 | +2.0% |  | +2.0% |  |
| 512x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 952.887 | 952.887 | 480.703 | 472.184 | +2.9% |  | +2.9% |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 17.689 | 17.689 | 17.688 | 0.001 | +0.0% |  |  |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 23.025 | 23.025 | 23.025 | 0.000 | +30.2% |  | +30.2% |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 20.399 | 20.399 | 20.399 | 0.000 | +15.3% |  | +15.3% |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 23.441 | 23.441 | 23.440 | 0.000 | +32.5% |  | +32.5% |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 70.424 | 70.424 | 22.812 | 47.613 | +18.7% |  |  |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 61.997 | 61.997 | 19.527 | 42.470 | +4.5% |  | -12.0% |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 59.308 | 59.308 | 19.025 | 40.283 | +0.0% |  | -15.8% |  |
| 512x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 59.721 | 59.721 | 18.795 | 40.926 | +0.7% |  | -15.2% |  |
| 512x512 | 4 | 65536 | sparse_screen | forward | v6_direct | ok | 292.527 | 292.527 | 292.526 | 0.001 | +0.0% |  |  |  |
| 512x512 | 4 | 65536 | sparse_screen | forward | v6_auto | ok | 309.308 | 309.308 | 309.307 | 0.001 | +5.7% |  | +5.7% |  |
| 512x512 | 4 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 294.444 | 294.444 | 294.444 | 0.000 | +0.7% |  | +0.7% |  |
| 512x512 | 4 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 294.128 | 294.128 | 294.127 | 0.001 | +0.5% |  | +0.5% |  |
| 512x512 | 4 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 217.652 | 217.652 | 160.678 | 56.973 | +0.0% |  |  |  |
| 512x512 | 4 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 222.173 | 222.173 | 168.822 | 53.350 | +2.1% |  | +2.1% |  |
| 512x512 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 218.765 | 218.765 | 163.267 | 55.497 | +0.5% |  | +0.5% |  |
| 512x512 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 219.396 | 219.396 | 167.281 | 52.116 | +0.8% |  | +0.8% |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 181.334 | 181.334 | 181.334 | 0.000 | +44.0% |  |  |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 189.049 | 189.049 | 189.048 | 0.001 | +50.2% |  | +4.3% |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 216.388 | 216.388 | 216.388 | 0.001 | +71.9% |  | +19.3% |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 125.890 | 125.890 | 125.890 | 0.000 | +0.0% |  | -30.6% |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 243.227 | 243.227 | 124.588 | 118.639 | +0.0% |  |  |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 273.408 | 273.408 | 149.882 | 123.526 | +12.4% |  | +12.4% |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 251.623 | 251.623 | 128.197 | 123.426 | +3.5% |  | +3.5% |  |
| 512x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 249.131 | 249.131 | 130.602 | 118.530 | +2.4% |  | +2.4% |  |
| 512x512 | 4 | 65536 | layered_depth | forward | v6_direct | ok | 391.225 | 391.225 | 391.224 | 0.001 | +0.0% |  |  |  |
| 512x512 | 4 | 65536 | layered_depth | forward | v6_auto | ok | 391.888 | 391.888 | 391.886 | 0.001 | +0.2% |  | +0.2% |  |
| 512x512 | 4 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 394.074 | 394.074 | 394.073 | 0.001 | +0.7% |  | +0.7% |  |
| 512x512 | 4 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 466.154 | 466.154 | 466.153 | 0.001 | +19.2% |  | +19.2% |  |
| 512x512 | 4 | 65536 | layered_depth | forward_backward | v6_direct | ok | 534.048 | 534.048 | 417.045 | 117.002 | +2.6% |  |  |  |
| 512x512 | 4 | 65536 | layered_depth | forward_backward | v6_auto | ok | 615.889 | 615.889 | 484.088 | 131.801 | +18.3% |  | +15.3% |  |
| 512x512 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 520.653 | 520.653 | 402.887 | 117.767 | +0.0% |  | -2.5% |  |
| 512x512 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 531.394 | 531.394 | 416.624 | 114.770 | +2.1% |  | -0.5% |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward | v6_direct | ok | 3269.691 | 3269.691 | 3269.689 | 0.002 | +0.0% |  |  |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward | v6_auto | ok | 3352.421 | 3352.421 | 3352.420 | 0.001 | +2.5% |  | +2.5% |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 3293.501 | 3293.501 | 3293.501 | 0.001 | +0.7% |  | +0.7% |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 3672.184 | 3672.184 | 3672.183 | 0.001 | +12.3% |  | +12.3% |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 6003.144 | 6003.144 | 4165.061 | 1838.083 | +6.7% |  |  |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 5787.979 | 5787.979 | 3940.123 | 1847.856 | +2.9% |  | -3.6% |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 5686.742 | 5686.742 | 3900.582 | 1786.160 | +1.1% |  | -5.3% |  |
| 512x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 5627.223 | 5627.223 | 3771.042 | 1856.180 | +0.0% |  | -6.3% |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward | v6_direct | ok | 7.704 | 7.704 | 7.704 | 0.000 | +23.3% |  |  |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward | v6_auto | ok | 8.143 | 8.143 | 8.143 | 0.000 | +30.3% |  | +5.7% |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 9.152 | 9.152 | 9.152 | 0.000 | +46.5% |  | +18.8% |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 6.248 | 6.248 | 6.248 | 0.000 | +0.0% |  | -18.9% |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 13.975 | 13.975 | 9.216 | 4.758 | +28.1% |  |  |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 14.901 | 14.901 | 9.790 | 5.112 | +36.6% |  | +6.6% |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 13.433 | 13.433 | 8.834 | 4.599 | +23.1% |  | -3.9% |  |
| 1024x512 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 10.909 | 10.909 | 8.402 | 2.508 | +0.0% |  | -21.9% |  |
| 1024x512 | 1 | 512 | sparse_screen | forward | v6_direct | ok | 7.801 | 7.801 | 7.801 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 1 | 512 | sparse_screen | forward | v6_auto | ok | 8.297 | 8.297 | 8.297 | 0.000 | +6.4% |  | +6.4% |  |
| 1024x512 | 1 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 7.875 | 7.875 | 7.875 | 0.000 | +0.9% |  | +0.9% |  |
| 1024x512 | 1 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 9.900 | 9.900 | 9.900 | 0.000 | +26.9% |  | +26.9% |  |
| 1024x512 | 1 | 512 | sparse_screen | forward_backward | v6_direct | ok | 11.578 | 11.578 | 7.692 | 3.886 | +9.5% |  |  |  |
| 1024x512 | 1 | 512 | sparse_screen | forward_backward | v6_auto | ok | 13.273 | 13.273 | 9.401 | 3.872 | +25.6% |  | +14.6% |  |
| 1024x512 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 11.621 | 11.621 | 8.164 | 3.457 | +9.9% |  | +0.4% |  |
| 1024x512 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 10.571 | 10.571 | 8.015 | 2.556 | +0.0% |  | -8.7% |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 7.843 | 7.843 | 7.843 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 12.929 | 12.929 | 12.929 | 0.000 | +64.8% |  | +64.8% |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 8.435 | 8.435 | 8.434 | 0.000 | +7.5% |  | +7.5% |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 13.085 | 13.085 | 13.084 | 0.000 | +66.8% |  | +66.8% |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 12.509 | 12.509 | 8.178 | 4.331 | +0.0% |  |  |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 16.007 | 16.007 | 11.437 | 4.570 | +28.0% |  | +28.0% |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 13.318 | 13.318 | 10.139 | 3.180 | +6.5% |  | +6.5% |  |
| 1024x512 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 15.906 | 15.906 | 10.959 | 4.947 | +27.2% |  | +27.2% |  |
| 1024x512 | 1 | 512 | layered_depth | forward | v6_direct | ok | 8.259 | 8.259 | 8.259 | 0.000 | +15.7% |  |  |  |
| 1024x512 | 1 | 512 | layered_depth | forward | v6_auto | ok | 9.138 | 9.138 | 9.138 | 0.000 | +28.0% |  | +10.6% |  |
| 1024x512 | 1 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 7.138 | 7.138 | 7.138 | 0.000 | +0.0% |  | -13.6% |  |
| 1024x512 | 1 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 11.655 | 11.655 | 11.655 | 0.000 | +63.3% |  | +41.1% |  |
| 1024x512 | 1 | 512 | layered_depth | forward_backward | v6_direct | ok | 14.503 | 14.503 | 8.852 | 5.651 | +8.6% |  |  |  |
| 1024x512 | 1 | 512 | layered_depth | forward_backward | v6_auto | ok | 13.356 | 13.356 | 9.113 | 4.243 | +0.0% |  | -7.9% |  |
| 1024x512 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 14.439 | 14.439 | 8.933 | 5.506 | +8.1% |  | -0.4% |  |
| 1024x512 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 13.684 | 13.684 | 10.786 | 2.898 | +2.5% |  | -5.6% |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward | v6_direct | ok | 9.377 | 9.377 | 9.376 | 0.000 | +13.6% |  |  |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward | v6_auto | ok | 11.619 | 11.619 | 11.619 | 0.000 | +40.8% |  | +23.9% |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 8.252 | 8.252 | 8.251 | 0.000 | +0.0% |  | -12.0% |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 14.150 | 14.150 | 14.149 | 0.000 | +71.5% |  | +50.9% |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 15.080 | 15.080 | 9.100 | 5.980 | +0.0% |  |  |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 20.306 | 20.306 | 14.720 | 5.586 | +34.7% |  | +34.7% |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 19.813 | 19.813 | 7.815 | 11.998 | +31.4% |  | +31.4% |  |
| 1024x512 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 19.806 | 19.806 | 13.577 | 6.230 | +31.3% |  | +31.3% |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward | v6_direct | ok | 8.965 | 8.965 | 8.965 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward | v6_auto | ok | 10.335 | 10.335 | 10.335 | 0.000 | +15.3% |  | +15.3% |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 9.360 | 9.360 | 9.360 | 0.000 | +4.4% |  | +4.4% |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 9.087 | 9.087 | 9.087 | 0.000 | +1.4% |  | +1.4% |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 23.015 | 23.015 | 10.532 | 12.482 | +57.3% |  |  |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 24.787 | 24.787 | 11.133 | 13.654 | +69.4% |  | +7.7% |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 14.630 | 14.630 | 7.130 | 7.500 | +0.0% |  | -36.4% |  |
| 1024x512 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 15.545 | 15.545 | 7.628 | 7.917 | +6.3% |  | -32.5% |  |
| 1024x512 | 4 | 512 | sparse_screen | forward | v6_direct | ok | 7.922 | 7.922 | 7.922 | 0.000 | +0.6% |  |  |  |
| 1024x512 | 4 | 512 | sparse_screen | forward | v6_auto | ok | 9.075 | 9.075 | 9.075 | 0.000 | +15.3% |  | +14.6% |  |
| 1024x512 | 4 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 7.871 | 7.871 | 7.871 | 0.000 | +0.0% |  | -0.6% |  |
| 1024x512 | 4 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 10.074 | 10.074 | 10.074 | 0.000 | +28.0% |  | +27.2% |  |
| 1024x512 | 4 | 512 | sparse_screen | forward_backward | v6_direct | ok | 13.875 | 13.875 | 6.632 | 7.243 | +0.0% |  |  |  |
| 1024x512 | 4 | 512 | sparse_screen | forward_backward | v6_auto | ok | 17.581 | 17.581 | 9.642 | 7.939 | +26.7% |  | +26.7% |  |
| 1024x512 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 15.945 | 15.945 | 7.476 | 8.469 | +14.9% |  | +14.9% |  |
| 1024x512 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 15.759 | 15.759 | 10.012 | 5.747 | +13.6% |  | +13.6% |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 8.488 | 8.488 | 8.487 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 11.761 | 11.761 | 11.761 | 0.000 | +38.6% |  | +38.6% |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 9.916 | 9.916 | 9.916 | 0.000 | +16.8% |  | +16.8% |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 12.222 | 12.222 | 12.221 | 0.000 | +44.0% |  | +44.0% |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 25.621 | 25.621 | 10.351 | 15.270 | +39.4% |  |  |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 18.383 | 18.383 | 12.131 | 6.252 | +0.0% |  | -28.3% |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 21.314 | 21.314 | 11.545 | 9.769 | +15.9% |  | -16.8% |  |
| 1024x512 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 19.672 | 19.672 | 13.170 | 6.502 | +7.0% |  | -23.2% |  |
| 1024x512 | 4 | 512 | layered_depth | forward | v6_direct | ok | 9.233 | 9.233 | 9.232 | 0.000 | +20.5% |  |  |  |
| 1024x512 | 4 | 512 | layered_depth | forward | v6_auto | ok | 7.661 | 7.661 | 7.661 | 0.001 | +0.0% |  | -17.0% |  |
| 1024x512 | 4 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 9.679 | 9.679 | 9.679 | 0.001 | +26.3% |  | +4.8% |  |
| 1024x512 | 4 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 14.856 | 14.856 | 14.855 | 0.000 | +93.9% |  | +60.9% |  |
| 1024x512 | 4 | 512 | layered_depth | forward_backward | v6_direct | ok | 19.735 | 19.735 | 10.189 | 9.546 | +0.0% |  |  |  |
| 1024x512 | 4 | 512 | layered_depth | forward_backward | v6_auto | ok | 22.740 | 22.740 | 11.650 | 11.090 | +15.2% |  | +15.2% |  |
| 1024x512 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 23.978 | 23.978 | 10.121 | 13.857 | +21.5% |  | +21.5% |  |
| 1024x512 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 30.419 | 30.419 | 16.326 | 14.092 | +54.1% |  | +54.1% |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward | v6_direct | ok | 11.725 | 11.725 | 11.725 | 0.000 | +11.3% |  |  |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward | v6_auto | ok | 14.034 | 14.034 | 14.034 | 0.000 | +33.2% |  | +19.7% |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 10.535 | 10.535 | 10.535 | 0.000 | +0.0% |  | -10.1% |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 12.793 | 12.793 | 12.793 | 0.000 | +21.4% |  | +9.1% |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 26.375 | 26.375 | 8.088 | 18.287 | +0.0% |  |  |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 32.619 | 32.619 | 14.001 | 18.618 | +23.7% |  | +23.7% |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 27.840 | 27.840 | 9.543 | 18.297 | +5.6% |  | +5.6% |  |
| 1024x512 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 31.648 | 31.648 | 14.093 | 17.555 | +20.0% |  | +20.0% |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 9.199 | 9.199 | 9.199 | 0.000 | +23.2% |  |  |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 7.465 | 7.465 | 7.465 | 0.000 | +0.0% |  | -18.8% |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 9.451 | 9.451 | 9.450 | 0.000 | +26.6% |  | +2.7% |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 8.958 | 8.958 | 8.958 | 0.000 | +20.0% |  | -2.6% |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 12.524 | 12.524 | 7.572 | 4.952 | +12.3% |  |  |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 13.229 | 13.229 | 9.396 | 3.834 | +18.6% |  | +5.6% |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 14.119 | 14.119 | 8.034 | 6.085 | +26.6% |  | +12.7% |  |
| 1024x512 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 11.156 | 11.156 | 8.154 | 3.002 | +0.0% |  | -10.9% |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward | v6_direct | ok | 7.674 | 7.674 | 7.674 | 0.000 | +11.7% |  |  |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward | v6_auto | ok | 7.287 | 7.287 | 7.287 | 0.000 | +6.0% |  | -5.0% |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 6.872 | 6.872 | 6.872 | 0.000 | +0.0% |  | -10.4% |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 11.606 | 11.606 | 11.606 | 0.000 | +68.9% |  | +51.2% |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 10.733 | 10.733 | 7.829 | 2.904 | +0.0% |  |  |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 12.906 | 12.906 | 9.525 | 3.381 | +20.2% |  | +20.2% |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 13.796 | 13.796 | 8.539 | 5.257 | +28.5% |  | +28.5% |  |
| 1024x512 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 11.859 | 11.859 | 8.584 | 3.274 | +10.5% |  | +10.5% |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 6.880 | 6.880 | 6.880 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 11.121 | 11.121 | 11.121 | 0.000 | +61.6% |  | +61.6% |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 9.016 | 9.016 | 9.016 | 0.000 | +31.0% |  | +31.0% |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 10.496 | 10.496 | 10.495 | 0.000 | +52.5% |  | +52.5% |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 13.818 | 13.818 | 7.226 | 6.592 | +14.4% |  |  |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 14.431 | 14.431 | 10.958 | 3.473 | +19.5% |  | +4.4% |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 12.077 | 12.077 | 8.589 | 3.489 | +0.0% |  | -12.6% |  |
| 1024x512 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 13.160 | 13.160 | 9.697 | 3.463 | +9.0% |  | -4.8% |  |
| 1024x512 | 1 | 2048 | layered_depth | forward | v6_direct | ok | 6.864 | 6.864 | 6.864 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 1 | 2048 | layered_depth | forward | v6_auto | ok | 8.692 | 8.692 | 8.692 | 0.000 | +26.6% |  | +26.6% |  |
| 1024x512 | 1 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 8.411 | 8.411 | 8.411 | 0.000 | +22.5% |  | +22.5% |  |
| 1024x512 | 1 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 9.428 | 9.428 | 9.428 | 0.000 | +37.4% |  | +37.4% |  |
| 1024x512 | 1 | 2048 | layered_depth | forward_backward | v6_direct | ok | 14.704 | 14.704 | 7.843 | 6.861 | +21.0% |  |  |  |
| 1024x512 | 1 | 2048 | layered_depth | forward_backward | v6_auto | ok | 12.333 | 12.333 | 8.570 | 3.763 | +1.5% |  | -16.1% |  |
| 1024x512 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 12.152 | 12.152 | 8.007 | 4.145 | +0.0% |  | -17.4% |  |
| 1024x512 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 13.039 | 13.039 | 9.194 | 3.845 | +7.3% |  | -11.3% |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward | v6_direct | ok | 9.828 | 9.828 | 9.827 | 0.001 | +10.7% |  |  |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward | v6_auto | ok | 10.587 | 10.587 | 10.587 | 0.000 | +19.3% |  | +7.7% |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 8.878 | 8.878 | 8.878 | 0.000 | +0.0% |  | -9.7% |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 14.292 | 14.292 | 14.291 | 0.000 | +61.0% |  | +45.4% |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 24.241 | 24.241 | 8.565 | 15.676 | +17.8% |  |  |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 21.632 | 21.632 | 10.749 | 10.882 | +5.1% |  | -10.8% |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 21.525 | 21.525 | 10.359 | 11.166 | +4.6% |  | -11.2% |  |
| 1024x512 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 20.583 | 20.583 | 11.038 | 9.544 | +0.0% |  | -15.1% |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 10.133 | 10.133 | 10.133 | 0.000 | +9.9% |  |  |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 11.320 | 11.320 | 11.319 | 0.000 | +22.8% |  | +11.7% |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 10.848 | 10.848 | 10.848 | 0.000 | +17.7% |  | +7.1% |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 9.220 | 9.220 | 9.220 | 0.000 | +0.0% |  | -9.0% |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 28.971 | 28.971 | 10.076 | 18.895 | +51.7% |  |  |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 32.302 | 32.302 | 11.804 | 20.497 | +69.2% |  | +11.5% |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 24.091 | 24.091 | 10.007 | 14.084 | +26.2% |  | -16.8% |  |
| 1024x512 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 19.096 | 19.096 | 7.273 | 11.823 | +0.0% |  | -34.1% |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward | v6_direct | ok | 9.289 | 9.289 | 9.289 | 0.001 | +0.0% |  |  |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward | v6_auto | ok | 9.367 | 9.367 | 9.367 | 0.000 | +0.8% |  | +0.8% |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 9.864 | 9.864 | 9.864 | 0.000 | +6.2% |  | +6.2% |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 10.903 | 10.903 | 10.902 | 0.000 | +17.4% |  | +17.4% |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 19.634 | 19.634 | 8.549 | 11.085 | +22.6% |  |  |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 22.536 | 22.536 | 9.566 | 12.970 | +40.7% |  | +14.8% |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 17.235 | 17.235 | 7.091 | 10.144 | +7.6% |  | -12.2% |  |
| 1024x512 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 16.019 | 16.019 | 9.182 | 6.837 | +0.0% |  | -18.4% |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 9.941 | 9.941 | 9.941 | 0.000 | +6.1% |  |  |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 9.368 | 9.368 | 9.368 | 0.000 | +0.0% |  | -5.8% |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 9.492 | 9.492 | 9.492 | 0.000 | +1.3% |  | -4.5% |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 13.129 | 13.129 | 13.129 | 0.000 | +40.1% |  | +32.1% |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 18.847 | 18.847 | 7.453 | 11.394 | +0.0% |  |  |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 20.528 | 20.528 | 13.127 | 7.402 | +8.9% |  | +8.9% |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 28.558 | 28.558 | 10.917 | 17.641 | +51.5% |  | +51.5% |  |
| 1024x512 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 19.858 | 19.858 | 12.290 | 7.568 | +5.4% |  | +5.4% |  |
| 1024x512 | 4 | 2048 | layered_depth | forward | v6_direct | ok | 11.399 | 11.399 | 11.398 | 0.000 | +9.4% |  |  |  |
| 1024x512 | 4 | 2048 | layered_depth | forward | v6_auto | ok | 11.411 | 11.411 | 11.411 | 0.000 | +9.5% |  | +0.1% |  |
| 1024x512 | 4 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 10.418 | 10.418 | 10.418 | 0.000 | +0.0% |  | -8.6% |  |
| 1024x512 | 4 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 11.655 | 11.655 | 11.655 | 0.000 | +11.9% |  | +2.3% |  |
| 1024x512 | 4 | 2048 | layered_depth | forward_backward | v6_direct | ok | 20.767 | 20.767 | 7.843 | 12.925 | +0.0% |  |  |  |
| 1024x512 | 4 | 2048 | layered_depth | forward_backward | v6_auto | ok | 29.733 | 29.733 | 10.638 | 19.095 | +43.2% |  | +43.2% |  |
| 1024x512 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 29.652 | 29.652 | 10.389 | 19.263 | +42.8% |  | +42.8% |  |
| 1024x512 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 21.653 | 21.653 | 12.796 | 8.857 | +4.3% |  | +4.3% |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward | v6_direct | ok | 14.936 | 14.936 | 14.936 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward | v6_auto | ok | 16.393 | 16.393 | 16.393 | 0.000 | +9.8% |  | +9.8% |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 20.366 | 20.366 | 20.365 | 0.000 | +36.4% |  | +36.4% |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 19.346 | 19.346 | 19.345 | 0.000 | +29.5% |  | +29.5% |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 58.365 | 58.365 | 16.832 | 41.532 | +8.4% |  |  |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 58.351 | 58.351 | 17.337 | 41.014 | +8.3% |  | -0.0% |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 59.094 | 59.094 | 17.498 | 41.597 | +9.7% |  | +1.3% |  |
| 1024x512 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 53.859 | 53.859 | 19.117 | 34.742 | +0.0% |  | -7.7% |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 8.917 | 8.917 | 8.917 | 0.000 | +0.0% |  |  |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 11.562 | 11.562 | 11.561 | 0.000 | +29.7% |  | +29.7% |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 10.915 | 10.915 | 10.914 | 0.000 | +22.4% |  | +22.4% |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 10.035 | 10.035 | 10.035 | 0.000 | +12.5% |  | +12.5% |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 35.420 | 35.420 | 11.129 | 24.292 | +24.0% |  |  |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 28.573 | 28.573 | 10.876 | 17.697 | +0.0% |  | -19.3% |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 29.562 | 29.562 | 11.550 | 18.012 | +3.5% |  | -16.5% |  |
| 1024x512 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 31.485 | 31.485 | 12.102 | 19.383 | +10.2% |  | -11.1% |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward | v6_direct | ok | 82.694 | 82.694 | 82.694 | 0.001 | +64.7% |  |  |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward | v6_auto | ok | 51.773 | 51.773 | 51.772 | 0.000 | +3.1% |  | -37.4% |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 50.218 | 50.218 | 50.218 | 0.000 | +0.0% |  | -39.3% |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 56.580 | 56.580 | 56.580 | 0.000 | +12.7% |  | -31.6% |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 88.720 | 88.720 | 59.521 | 29.199 | +11.4% |  |  |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 79.653 | 79.653 | 59.986 | 19.667 | +0.0% |  | -10.2% |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 93.833 | 93.833 | 67.374 | 26.459 | +17.8% |  | +5.8% |  |
| 1024x512 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 90.361 | 90.361 | 64.967 | 25.394 | +13.4% |  | +1.8% |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 63.094 | 63.094 | 63.093 | 0.001 | +40.2% |  |  |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 95.354 | 95.354 | 95.353 | 0.001 | +111.9% |  | +51.1% |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 44.999 | 44.999 | 44.999 | 0.000 | +0.0% |  | -28.7% |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 85.318 | 85.318 | 85.317 | 0.001 | +89.6% |  | +35.2% |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 106.379 | 106.379 | 64.177 | 42.202 | +23.6% |  |  |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 108.089 | 108.089 | 66.386 | 41.703 | +25.6% |  | +1.6% |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 86.063 | 86.063 | 44.544 | 41.520 | +0.0% |  | -19.1% |  |
| 1024x512 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 87.896 | 87.896 | 47.761 | 40.135 | +2.1% |  | -17.4% |  |
| 1024x512 | 1 | 65536 | layered_depth | forward | v6_direct | ok | 187.631 | 187.631 | 187.630 | 0.001 | +0.0% |  |  |  |
| 1024x512 | 1 | 65536 | layered_depth | forward | v6_auto | ok | 201.910 | 201.910 | 201.909 | 0.001 | +7.6% |  | +7.6% |  |
| 1024x512 | 1 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 253.691 | 253.691 | 253.689 | 0.002 | +35.2% |  | +35.2% |  |
| 1024x512 | 1 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 201.180 | 201.180 | 201.178 | 0.001 | +7.2% |  | +7.2% |  |
| 1024x512 | 1 | 65536 | layered_depth | forward_backward | v6_direct | ok | 225.755 | 225.755 | 182.248 | 43.507 | +22.9% |  |  |  |
| 1024x512 | 1 | 65536 | layered_depth | forward_backward | v6_auto | ok | 231.195 | 231.195 | 178.678 | 52.517 | +25.9% |  | +2.4% |  |
| 1024x512 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 183.656 | 183.656 | 142.427 | 41.229 | +0.0% |  | -18.6% |  |
| 1024x512 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 185.554 | 185.554 | 150.771 | 34.783 | +1.0% |  | -17.8% |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward | v6_direct | ok | 800.767 | 800.767 | 800.766 | 0.001 | +0.0% |  |  |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward | v6_auto | ok | 968.589 | 968.589 | 968.588 | 0.001 | +21.0% |  | +21.0% |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 802.118 | 802.118 | 802.117 | 0.001 | +0.2% |  | +0.2% |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 819.777 | 819.777 | 819.776 | 0.001 | +2.4% |  | +2.4% |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 975.902 | 975.902 | 503.615 | 472.287 | +3.9% |  |  |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 986.564 | 986.564 | 517.827 | 468.737 | +5.0% |  | +1.1% |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 939.235 | 939.235 | 470.362 | 468.873 | +0.0% |  | -3.8% |  |
| 1024x512 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 1133.043 | 1133.043 | 641.237 | 491.806 | +20.6% |  | +16.1% |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 22.047 | 22.047 | 22.046 | 0.000 | +7.1% |  |  |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 38.866 | 38.866 | 38.866 | 0.001 | +88.8% |  | +76.3% |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 20.630 | 20.630 | 20.630 | 0.000 | +0.2% |  | -6.4% |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 20.585 | 20.585 | 20.585 | 0.000 | +0.0% |  | -6.6% |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 87.780 | 87.780 | 21.399 | 66.381 | +0.0% |  |  |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 89.144 | 89.144 | 22.576 | 66.568 | +1.6% |  | +1.6% |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 90.050 | 90.050 | 23.062 | 66.988 | +2.6% |  | +2.6% |  |
| 1024x512 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 90.148 | 90.148 | 23.344 | 66.804 | +2.7% |  | +2.7% |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward | v6_direct | ok | 196.378 | 196.378 | 196.377 | 0.001 | +0.0% |  |  |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward | v6_auto | ok | 322.087 | 322.087 | 322.086 | 0.001 | +64.0% |  | +64.0% |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 302.278 | 302.278 | 302.277 | 0.000 | +53.9% |  | +53.9% |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 299.662 | 299.662 | 299.661 | 0.001 | +52.6% |  | +52.6% |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 261.966 | 261.966 | 197.380 | 64.586 | +11.4% |  |  |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 247.388 | 247.388 | 184.533 | 62.855 | +5.2% |  | -5.6% |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 235.174 | 235.174 | 177.908 | 57.266 | +0.0% |  | -10.2% |  |
| 1024x512 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 235.840 | 235.840 | 179.464 | 56.376 | +0.3% |  | -10.0% |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 285.654 | 285.654 | 285.654 | 0.000 | +77.4% |  |  |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 353.166 | 353.166 | 353.165 | 0.001 | +119.3% |  | +23.6% |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 256.082 | 256.082 | 256.081 | 0.001 | +59.0% |  | -10.4% |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 161.056 | 161.056 | 161.055 | 0.001 | +0.0% |  | -43.6% |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 306.079 | 306.079 | 180.806 | 125.273 | +3.0% |  |  |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 356.493 | 356.493 | 223.208 | 133.285 | +20.0% |  | +16.5% |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 297.145 | 297.145 | 165.520 | 131.625 | +0.0% |  | -2.9% |  |
| 1024x512 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 373.940 | 373.940 | 240.361 | 133.578 | +25.8% |  | +22.2% |  |
| 1024x512 | 4 | 65536 | layered_depth | forward | v6_direct | ok | 1082.428 | 1082.428 | 1082.427 | 0.001 | +29.3% |  |  |  |
| 1024x512 | 4 | 65536 | layered_depth | forward | v6_auto | ok | 836.914 | 836.914 | 836.913 | 0.001 | +0.0% |  | -22.7% |  |
| 1024x512 | 4 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 878.010 | 878.010 | 878.010 | 0.001 | +4.9% |  | -18.9% |  |
| 1024x512 | 4 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 1015.139 | 1015.139 | 1015.138 | 0.001 | +21.3% |  | -6.2% |  |
| 1024x512 | 4 | 65536 | layered_depth | forward_backward | v6_direct | ok | 738.369 | 738.369 | 592.795 | 145.574 | +9.5% |  |  |  |
| 1024x512 | 4 | 65536 | layered_depth | forward_backward | v6_auto | ok | 693.446 | 693.446 | 561.939 | 131.507 | +2.8% |  | -6.1% |  |
| 1024x512 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 674.355 | 674.355 | 546.735 | 127.620 | +0.0% |  | -8.7% |  |
| 1024x512 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 675.860 | 675.860 | 553.628 | 122.232 | +0.2% |  | -8.5% |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward | v6_direct | ok | 3339.122 | 3339.122 | 3339.121 | 0.001 | +2.2% |  |  |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward | v6_auto | ok | 3659.213 | 3659.213 | 3659.212 | 0.001 | +12.0% |  | +9.6% |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 3267.516 | 3267.516 | 3267.515 | 0.001 | +0.0% |  | -2.1% |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 3800.972 | 3800.972 | 3800.971 | 0.001 | +16.3% |  | +13.8% |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 5631.338 | 5631.338 | 3821.524 | 1809.815 | +13.7% |  |  |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 5019.032 | 5019.032 | 3218.978 | 1800.053 | +1.3% |  | -10.9% |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 4975.209 | 4975.209 | 3181.231 | 1793.978 | +0.5% |  | -11.7% |  |
| 1024x512 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 4952.709 | 4952.709 | 3149.462 | 1803.247 | +0.0% |  | -12.1% |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward | v6_direct | ok | 6.685 | 6.685 | 6.684 | 0.001 | +0.0% |  |  |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward | v6_auto | ok | 9.096 | 9.096 | 9.096 | 0.000 | +36.1% |  | +36.1% |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 9.118 | 9.118 | 9.118 | 0.000 | +36.4% |  | +36.4% |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 11.054 | 11.054 | 11.054 | 0.000 | +65.4% |  | +65.4% |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 13.586 | 13.586 | 7.819 | 5.767 | +0.0% |  |  |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 16.277 | 16.277 | 8.632 | 7.644 | +19.8% |  | +19.8% |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 20.097 | 20.097 | 9.640 | 10.457 | +47.9% |  | +47.9% |  |
| 1920x1080 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 15.798 | 15.798 | 10.080 | 5.717 | +16.3% |  | +16.3% |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward | v6_direct | ok | 6.891 | 6.891 | 6.891 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward | v6_auto | ok | 14.172 | 14.172 | 14.172 | 0.000 | +105.7% |  | +105.7% |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 8.119 | 8.119 | 8.119 | 0.000 | +17.8% |  | +17.8% |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 12.309 | 12.309 | 12.309 | 0.000 | +78.6% |  | +78.6% |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward_backward | v6_direct | ok | 11.955 | 11.955 | 6.202 | 5.752 | +0.0% |  |  |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward_backward | v6_auto | ok | 15.712 | 15.712 | 10.222 | 5.490 | +31.4% |  | +31.4% |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 15.922 | 15.922 | 8.072 | 7.851 | +33.2% |  | +33.2% |  |
| 1920x1080 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 18.558 | 18.558 | 13.320 | 5.238 | +55.2% |  | +55.2% |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 8.175 | 8.175 | 8.175 | 0.000 | +6.1% |  |  |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 13.337 | 13.337 | 13.336 | 0.000 | +73.0% |  | +63.1% |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 7.708 | 7.708 | 7.708 | 0.000 | +0.0% |  | -5.7% |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 11.900 | 11.900 | 11.900 | 0.000 | +54.4% |  | +45.6% |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 19.497 | 19.497 | 9.448 | 10.049 | +25.0% |  |  |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 15.851 | 15.851 | 10.255 | 5.597 | +1.6% |  | -18.7% |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 15.599 | 15.599 | 7.976 | 7.623 | +0.0% |  | -20.0% |  |
| 1920x1080 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 17.776 | 17.776 | 12.047 | 5.730 | +14.0% |  | -8.8% |  |
| 1920x1080 | 1 | 512 | layered_depth | forward | v6_direct | ok | 7.430 | 7.430 | 7.430 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 1 | 512 | layered_depth | forward | v6_auto | ok | 8.983 | 8.983 | 8.983 | 0.000 | +20.9% |  | +20.9% |  |
| 1920x1080 | 1 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 7.962 | 7.962 | 7.962 | 0.000 | +7.2% |  | +7.2% |  |
| 1920x1080 | 1 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 16.578 | 16.578 | 16.577 | 0.000 | +123.1% |  | +123.1% |  |
| 1920x1080 | 1 | 512 | layered_depth | forward_backward | v6_direct | ok | 16.770 | 16.770 | 7.499 | 9.270 | +0.0% |  |  |  |
| 1920x1080 | 1 | 512 | layered_depth | forward_backward | v6_auto | ok | 17.770 | 17.770 | 9.395 | 8.375 | +6.0% |  | +6.0% |  |
| 1920x1080 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 19.388 | 19.388 | 9.440 | 9.948 | +15.6% |  | +15.6% |  |
| 1920x1080 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 17.063 | 17.063 | 11.759 | 5.304 | +1.7% |  | +1.7% |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward | v6_direct | ok | 12.381 | 12.381 | 12.380 | 0.000 | +46.7% |  |  |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward | v6_auto | ok | 16.934 | 16.934 | 16.934 | 0.000 | +100.7% |  | +36.8% |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 8.437 | 8.437 | 8.437 | 0.000 | +0.0% |  | -31.9% |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 10.937 | 10.937 | 10.937 | 0.000 | +29.6% |  | -11.7% |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 31.441 | 31.441 | 12.609 | 18.831 | +52.1% |  |  |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 20.675 | 20.675 | 12.794 | 7.882 | +0.0% |  | -34.2% |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 26.996 | 26.996 | 11.065 | 15.931 | +30.6% |  | -14.1% |  |
| 1920x1080 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 20.766 | 20.766 | 12.361 | 8.406 | +0.4% |  | -34.0% |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward | v6_direct | ok | 9.670 | 9.670 | 9.670 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward | v6_auto | ok | 11.098 | 11.098 | 11.097 | 0.000 | +14.8% |  | +14.8% |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 10.407 | 10.407 | 10.407 | 0.000 | +7.6% |  | +7.6% |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 17.002 | 17.002 | 17.002 | 0.000 | +75.8% |  | +75.8% |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 24.362 | 24.362 | 8.032 | 16.330 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 27.065 | 27.065 | 8.419 | 18.646 | +11.1% |  | +11.1% |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 28.134 | 28.134 | 9.100 | 19.034 | +15.5% |  | +15.5% |  |
| 1920x1080 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 31.922 | 31.922 | 16.239 | 15.683 | +31.0% |  | +31.0% |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward | v6_direct | ok | 9.064 | 9.064 | 9.063 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward | v6_auto | ok | 15.726 | 15.726 | 15.726 | 0.000 | +73.5% |  | +73.5% |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 10.419 | 10.419 | 10.419 | 0.000 | +15.0% |  | +15.0% |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 15.537 | 15.537 | 15.537 | 0.000 | +71.4% |  | +71.4% |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward_backward | v6_direct | ok | 25.446 | 25.446 | 6.781 | 18.664 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward_backward | v6_auto | ok | 30.928 | 30.928 | 15.419 | 15.510 | +21.5% |  | +21.5% |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 28.132 | 28.132 | 7.973 | 20.159 | +10.6% |  | +10.6% |  |
| 1920x1080 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 31.934 | 31.934 | 16.166 | 15.768 | +25.5% |  | +25.5% |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 9.296 | 9.296 | 9.295 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 16.353 | 16.353 | 16.353 | 0.000 | +75.9% |  | +75.9% |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 10.930 | 10.930 | 10.929 | 0.000 | +17.6% |  | +17.6% |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 15.547 | 15.547 | 15.547 | 0.000 | +67.2% |  | +67.2% |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 27.220 | 27.220 | 7.832 | 19.388 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 31.856 | 31.856 | 16.095 | 15.761 | +17.0% |  | +17.0% |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 30.139 | 30.139 | 8.848 | 21.291 | +10.7% |  | +10.7% |  |
| 1920x1080 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 31.610 | 31.610 | 15.626 | 15.984 | +16.1% |  | +16.1% |  |
| 1920x1080 | 4 | 512 | layered_depth | forward | v6_direct | ok | 9.416 | 9.416 | 9.415 | 0.000 | +28.9% |  |  |  |
| 1920x1080 | 4 | 512 | layered_depth | forward | v6_auto | ok | 10.336 | 10.336 | 10.336 | 0.000 | +41.5% |  | +9.8% |  |
| 1920x1080 | 4 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 7.303 | 7.303 | 7.303 | 0.000 | +0.0% |  | -22.4% |  |
| 1920x1080 | 4 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 15.632 | 15.632 | 15.632 | 0.000 | +114.0% |  | +66.0% |  |
| 1920x1080 | 4 | 512 | layered_depth | forward_backward | v6_direct | ok | 29.148 | 29.148 | 9.263 | 19.885 | +2.2% |  |  |  |
| 1920x1080 | 4 | 512 | layered_depth | forward_backward | v6_auto | ok | 29.294 | 29.294 | 9.324 | 19.970 | +2.7% |  | +0.5% |  |
| 1920x1080 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 28.533 | 28.533 | 8.916 | 19.617 | +0.0% |  | -2.1% |  |
| 1920x1080 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 31.260 | 31.260 | 15.370 | 15.890 | +9.6% |  | +7.2% |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward | v6_direct | ok | 9.723 | 9.723 | 9.723 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward | v6_auto | ok | 19.236 | 19.236 | 19.236 | 0.000 | +97.8% |  | +97.8% |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 12.872 | 12.872 | 12.872 | 0.000 | +32.4% |  | +32.4% |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 20.339 | 20.339 | 20.339 | 0.000 | +109.2% |  | +109.2% |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 43.468 | 43.468 | 10.376 | 33.092 | +0.0% |  |  |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 46.611 | 46.611 | 19.506 | 27.106 | +7.2% |  | +7.2% |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 51.951 | 51.951 | 15.222 | 36.729 | +19.5% |  | +19.5% |  |
| 1920x1080 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 46.596 | 46.596 | 19.951 | 26.645 | +7.2% |  | +7.2% |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 6.824 | 6.824 | 6.824 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 9.474 | 9.474 | 9.474 | 0.000 | +38.8% |  | +38.8% |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 8.126 | 8.126 | 8.126 | 0.000 | +19.1% |  | +19.1% |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 7.452 | 7.452 | 7.452 | 0.000 | +9.2% |  | +9.2% |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 11.979 | 11.979 | 5.495 | 6.484 | +0.0% |  |  |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 16.754 | 16.754 | 8.700 | 8.054 | +39.9% |  | +39.9% |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 16.433 | 16.433 | 7.551 | 8.882 | +37.2% |  | +37.2% |  |
| 1920x1080 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 18.486 | 18.486 | 9.346 | 9.140 | +54.3% |  | +54.3% |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward | v6_direct | ok | 8.060 | 8.060 | 8.060 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward | v6_auto | ok | 17.004 | 17.004 | 17.004 | 0.000 | +111.0% |  | +111.0% |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 10.007 | 10.007 | 10.007 | 0.000 | +24.2% |  | +24.2% |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 13.058 | 13.058 | 13.058 | 0.000 | +62.0% |  | +62.0% |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 18.158 | 18.158 | 9.774 | 8.384 | +15.7% |  |  |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 15.688 | 15.688 | 10.025 | 5.663 | +0.0% |  | -13.6% |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 18.049 | 18.049 | 8.959 | 9.090 | +15.1% |  | -0.6% |  |
| 1920x1080 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 22.486 | 22.486 | 12.389 | 10.096 | +43.3% |  | +23.8% |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 8.658 | 8.658 | 8.658 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 12.271 | 12.271 | 12.270 | 0.001 | +41.7% |  | +41.7% |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 10.613 | 10.613 | 10.612 | 0.000 | +22.6% |  | +22.6% |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 11.482 | 11.482 | 11.482 | 0.001 | +32.6% |  | +32.6% |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 20.834 | 20.834 | 11.038 | 9.796 | +28.8% |  |  |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 21.055 | 21.055 | 15.095 | 5.960 | +30.1% |  | +1.1% |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 16.177 | 16.177 | 8.049 | 8.128 | +0.0% |  | -22.4% |  |
| 1920x1080 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 19.244 | 19.244 | 13.500 | 5.743 | +19.0% |  | -7.6% |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward | v6_direct | ok | 10.133 | 10.133 | 10.132 | 0.000 | +33.0% |  |  |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward | v6_auto | ok | 11.240 | 11.240 | 11.240 | 0.000 | +47.6% |  | +10.9% |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 7.617 | 7.617 | 7.617 | 0.000 | +0.0% |  | -24.8% |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 15.576 | 15.576 | 15.575 | 0.000 | +104.5% |  | +53.7% |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward_backward | v6_direct | ok | 14.567 | 14.567 | 6.236 | 8.330 | +0.0% |  |  |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward_backward | v6_auto | ok | 20.490 | 20.490 | 11.019 | 9.471 | +40.7% |  | +40.7% |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 19.461 | 19.461 | 10.825 | 8.636 | +33.6% |  | +33.6% |  |
| 1920x1080 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 20.738 | 20.738 | 14.090 | 6.648 | +42.4% |  | +42.4% |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward | v6_direct | ok | 10.485 | 10.485 | 10.484 | 0.000 | +6.4% |  |  |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward | v6_auto | ok | 10.576 | 10.576 | 10.576 | 0.000 | +7.3% |  | +0.9% |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 9.853 | 9.853 | 9.853 | 0.000 | +0.0% |  | -6.0% |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 15.790 | 15.790 | 15.789 | 0.000 | +60.3% |  | +50.6% |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 23.818 | 23.818 | 9.962 | 13.856 | +4.6% |  |  |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 23.611 | 23.611 | 10.011 | 13.600 | +3.7% |  | -0.9% |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 22.775 | 22.775 | 9.575 | 13.200 | +0.0% |  | -4.4% |  |
| 1920x1080 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 26.926 | 26.926 | 14.687 | 12.239 | +18.2% |  | +13.0% |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 13.896 | 13.896 | 13.896 | 0.001 | +26.6% |  |  |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 13.597 | 13.597 | 13.596 | 0.000 | +23.9% |  | -2.2% |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 11.670 | 11.670 | 11.670 | 0.000 | +6.3% |  | -16.0% |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 10.974 | 10.974 | 10.974 | 0.001 | +0.0% |  | -21.0% |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 32.240 | 32.240 | 9.677 | 22.563 | +8.7% |  |  |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 31.780 | 31.780 | 9.977 | 21.803 | +7.1% |  | -1.4% |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 37.694 | 37.694 | 11.869 | 25.825 | +27.1% |  | +16.9% |  |
| 1920x1080 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 29.660 | 29.660 | 8.261 | 21.398 | +0.0% |  | -8.0% |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward | v6_direct | ok | 12.855 | 12.855 | 12.855 | 0.000 | +18.0% |  |  |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward | v6_auto | ok | 16.591 | 16.591 | 16.591 | 0.000 | +52.2% |  | +29.1% |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 10.898 | 10.898 | 10.897 | 0.000 | +0.0% |  | -15.2% |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 16.199 | 16.199 | 16.199 | 0.000 | +48.6% |  | +26.0% |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 29.165 | 29.165 | 9.163 | 20.002 | +2.7% |  |  |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 34.946 | 34.946 | 18.900 | 16.046 | +23.1% |  | +19.8% |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 28.393 | 28.393 | 8.323 | 20.069 | +0.0% |  | -2.6% |  |
| 1920x1080 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 33.242 | 33.242 | 16.364 | 16.878 | +17.1% |  | +14.0% |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 12.976 | 12.976 | 12.975 | 0.001 | +0.0% |  |  |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 14.998 | 14.998 | 14.997 | 0.000 | +15.6% |  | +15.6% |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 13.916 | 13.916 | 13.916 | 0.000 | +7.2% |  | +7.2% |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 16.195 | 16.195 | 16.194 | 0.000 | +24.8% |  | +24.8% |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 45.496 | 45.496 | 11.261 | 34.235 | +47.0% |  |  |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 34.340 | 34.340 | 17.435 | 16.905 | +10.9% |  | -24.5% |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 30.957 | 30.957 | 9.348 | 21.610 | +0.0% |  | -32.0% |  |
| 1920x1080 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 33.138 | 33.138 | 16.125 | 17.013 | +7.0% |  | -27.2% |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward | v6_direct | ok | 9.748 | 9.748 | 9.748 | 0.000 | +10.0% |  |  |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward | v6_auto | ok | 13.674 | 13.674 | 13.674 | 0.000 | +54.3% |  | +40.3% |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 8.863 | 8.863 | 8.863 | 0.000 | +0.0% |  | -9.1% |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 17.236 | 17.236 | 17.235 | 0.000 | +94.5% |  | +76.8% |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward_backward | v6_direct | ok | 36.200 | 36.200 | 10.166 | 26.034 | +2.0% |  |  |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward_backward | v6_auto | ok | 41.598 | 41.598 | 11.358 | 30.240 | +17.2% |  | +14.9% |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 38.959 | 38.959 | 10.458 | 28.501 | +9.8% |  | +7.6% |  |
| 1920x1080 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 35.483 | 35.483 | 16.283 | 19.200 | +0.0% |  | -2.0% |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward | v6_direct | ok | 15.712 | 15.712 | 15.712 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward | v6_auto | ok | 16.370 | 16.370 | 16.370 | 0.000 | +4.2% |  | +4.2% |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 15.989 | 15.989 | 15.989 | 0.000 | +1.8% |  | +1.8% |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 24.682 | 24.682 | 24.682 | 0.000 | +57.1% |  | +57.1% |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 67.594 | 67.594 | 17.213 | 50.381 | +0.6% |  |  |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 68.183 | 68.183 | 18.008 | 50.175 | +1.5% |  | +0.9% |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 68.458 | 68.458 | 17.690 | 50.768 | +1.9% |  | +1.3% |  |
| 1920x1080 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 67.194 | 67.194 | 23.698 | 43.496 | +0.0% |  | -0.6% |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 13.202 | 13.202 | 13.201 | 0.001 | +18.1% |  |  |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 11.181 | 11.181 | 11.181 | 0.000 | +0.0% |  | -15.3% |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 13.443 | 13.443 | 13.442 | 0.000 | +20.2% |  | +1.8% |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 13.281 | 13.281 | 13.280 | 0.001 | +18.8% |  | +0.6% |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 35.762 | 35.762 | 13.407 | 22.356 | +6.7% |  |  |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 35.614 | 35.614 | 13.038 | 22.576 | +6.2% |  | -0.4% |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 36.858 | 36.858 | 14.280 | 22.578 | +9.9% |  | +3.1% |  |
| 1920x1080 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 33.525 | 33.525 | 10.893 | 22.632 | +0.0% |  | -6.3% |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward | v6_direct | ok | 49.313 | 49.313 | 49.312 | 0.001 | +10.1% |  |  |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward | v6_auto | ok | 57.298 | 57.298 | 57.298 | 0.000 | +28.0% |  | +16.2% |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 44.779 | 44.779 | 44.778 | 0.000 | +0.0% |  | -9.2% |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 53.912 | 53.912 | 53.912 | 0.000 | +20.4% |  | +9.3% |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 57.720 | 57.720 | 37.636 | 20.084 | +0.0% |  |  |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 76.007 | 76.007 | 48.604 | 27.403 | +31.7% |  | +31.7% |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 58.613 | 58.613 | 38.200 | 20.413 | +1.5% |  | +1.5% |  |
| 1920x1080 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 63.691 | 63.691 | 44.104 | 19.587 | +10.3% |  | +10.3% |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 118.825 | 118.825 | 118.824 | 0.001 | +0.0% |  |  |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 135.263 | 135.263 | 135.262 | 0.001 | +13.8% |  | +13.8% |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 121.049 | 121.049 | 121.048 | 0.001 | +1.9% |  | +1.9% |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 146.241 | 146.241 | 146.240 | 0.001 | +23.1% |  | +23.1% |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 103.832 | 103.832 | 68.743 | 35.089 | +0.0% |  |  |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 138.721 | 138.721 | 94.442 | 44.279 | +33.6% |  | +33.6% |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 132.666 | 132.666 | 89.606 | 43.060 | +27.8% |  | +27.8% |  |
| 1920x1080 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 127.497 | 127.497 | 84.269 | 43.228 | +22.8% |  | +22.8% |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward | v6_direct | ok | 194.863 | 194.863 | 194.862 | 0.001 | +18.3% |  |  |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward | v6_auto | ok | 267.090 | 267.090 | 267.089 | 0.001 | +62.2% |  | +37.1% |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 164.657 | 164.657 | 164.656 | 0.001 | +0.0% |  | -15.5% |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 208.814 | 208.814 | 208.814 | 0.000 | +26.8% |  | +7.2% |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward_backward | v6_direct | ok | 182.487 | 182.487 | 149.884 | 32.602 | +0.0% |  |  |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward_backward | v6_auto | ok | 234.844 | 234.844 | 183.324 | 51.520 | +28.7% |  | +28.7% |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 227.419 | 227.419 | 177.397 | 50.021 | +24.6% |  | +24.6% |  |
| 1920x1080 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 281.520 | 281.520 | 231.956 | 49.565 | +54.3% |  | +54.3% |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward | v6_direct | ok | 774.606 | 774.606 | 774.605 | 0.001 | +66.8% |  |  |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward | v6_auto | ok | 464.530 | 464.530 | 464.529 | 0.001 | +0.0% |  | -40.0% |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 467.864 | 467.864 | 467.864 | 0.001 | +0.7% |  | -39.6% |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 1354.232 | 1354.232 | 1354.231 | 0.001 | +191.5% |  | +74.8% |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 938.766 | 938.766 | 463.426 | 475.340 | +0.0% |  |  |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 1139.999 | 1139.999 | 630.826 | 509.174 | +21.4% |  | +21.4% |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 1046.913 | 1046.913 | 576.377 | 470.536 | +11.5% |  | +11.5% |  |
| 1920x1080 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 1019.749 | 1019.749 | 547.669 | 472.079 | +8.6% |  | +8.6% |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 25.159 | 25.159 | 25.159 | 0.000 | +4.8% |  |  |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 24.006 | 24.006 | 24.006 | 0.000 | +0.0% |  | -4.6% |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 24.814 | 24.814 | 24.814 | 0.000 | +3.4% |  | -1.4% |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 35.263 | 35.263 | 35.263 | 0.000 | +46.9% |  | +40.2% |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 116.159 | 116.159 | 25.317 | 90.842 | +3.1% |  |  |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 112.713 | 112.713 | 26.258 | 86.455 | +0.0% |  | -3.0% |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 113.743 | 113.743 | 25.877 | 87.866 | +0.9% |  | -2.1% |  |
| 1920x1080 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 113.715 | 113.715 | 26.410 | 87.305 | +0.9% |  | -2.1% |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward | v6_direct | ok | 97.464 | 97.464 | 97.464 | 0.000 | +0.0% |  |  |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward | v6_auto | ok | 156.453 | 156.453 | 156.453 | 0.001 | +60.5% |  | +60.5% |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 98.885 | 98.885 | 98.884 | 0.000 | +1.5% |  | +1.5% |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 160.102 | 160.102 | 160.101 | 0.001 | +64.3% |  | +64.3% |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 153.173 | 153.173 | 101.444 | 51.729 | +3.3% |  |  |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 183.732 | 183.732 | 131.365 | 52.367 | +23.9% |  | +20.0% |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 148.342 | 148.342 | 101.009 | 47.333 | +0.0% |  | -3.2% |  |
| 1920x1080 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 198.391 | 198.391 | 139.690 | 58.701 | +33.7% |  | +29.5% |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 429.719 | 429.719 | 429.718 | 0.001 | +10.7% |  |  |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 388.241 | 388.241 | 388.240 | 0.001 | +0.0% |  | -9.7% |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 587.375 | 587.375 | 587.374 | 0.001 | +51.3% |  | +36.7% |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 423.968 | 423.968 | 423.967 | 0.001 | +9.2% |  | -1.3% |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 388.268 | 388.268 | 274.047 | 114.221 | +1.9% |  |  |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 381.166 | 381.166 | 271.816 | 109.350 | +0.0% |  | -1.8% |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 386.651 | 386.651 | 276.495 | 110.156 | +1.4% |  | -0.4% |  |
| 1920x1080 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 388.317 | 388.317 | 277.097 | 111.220 | +1.9% |  | +0.0% |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward | v6_direct | ok | 733.890 | 733.890 | 733.890 | 0.001 | +1.0% |  |  |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward | v6_auto | ok | 827.652 | 827.652 | 827.651 | 0.001 | +13.9% |  | +12.8% |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 822.610 | 822.610 | 822.609 | 0.001 | +13.2% |  | +12.1% |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 726.508 | 726.508 | 726.507 | 0.001 | +0.0% |  | -1.0% |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward_backward | v6_direct | ok | 788.865 | 788.865 | 658.030 | 130.835 | +57.7% |  |  |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward_backward | v6_auto | ok | 807.955 | 807.955 | 677.911 | 130.044 | +61.5% |  | +2.4% |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 500.336 | 500.336 | 416.019 | 84.317 | +0.0% |  | -36.6% |  |
| 1920x1080 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 860.180 | 860.180 | 738.042 | 122.137 | +71.9% |  | +9.0% |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward | v6_direct | ok | 4640.531 | 4640.531 | 4640.530 | 0.001 | +150.6% |  |  |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward | v6_auto | ok | 3141.842 | 3141.842 | 3141.841 | 0.001 | +69.7% |  | -32.3% |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 1851.890 | 1851.890 | 1851.889 | 0.001 | +0.0% |  | -60.1% |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 3961.425 | 3961.425 | 3961.424 | 0.001 | +113.9% |  | -14.6% |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 5555.706 | 5555.706 | 3689.981 | 1865.724 | +0.0% |  |  |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 7639.549 | 7639.549 | 5666.414 | 1973.135 | +37.5% |  | +37.5% |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 7091.385 | 7091.385 | 5168.080 | 1923.305 | +27.6% |  | +27.6% |  |
| 1920x1080 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 6834.950 | 6834.950 | 4989.310 | 1845.640 | +23.0% |  | +23.0% |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward | v6_direct | ok | 11.670 | 11.670 | 11.669 | 0.001 | +20.5% |  |  |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward | v6_auto | ok | 27.750 | 27.750 | 27.750 | 0.000 | +186.6% |  | +137.8% |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 9.681 | 9.681 | 9.681 | 0.000 | +0.0% |  | -17.0% |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 26.502 | 26.502 | 26.502 | 0.000 | +173.8% |  | +127.1% |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 41.527 | 41.527 | 9.127 | 32.400 | +15.4% |  |  |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 55.029 | 55.029 | 25.809 | 29.220 | +53.0% |  | +32.5% |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 35.972 | 35.972 | 7.165 | 28.807 | +0.0% |  | -13.4% |  |
| 4096x4096 | 1 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 54.930 | 54.930 | 26.485 | 28.445 | +52.7% |  | +32.3% |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward | v6_direct | ok | 11.487 | 11.487 | 11.487 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward | v6_auto | ok | 24.143 | 24.143 | 24.143 | 0.000 | +110.2% |  | +110.2% |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 12.297 | 12.297 | 12.296 | 0.000 | +7.0% |  | +7.0% |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 23.917 | 23.917 | 23.917 | 0.000 | +108.2% |  | +108.2% |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward_backward | v6_direct | ok | 34.643 | 34.643 | 6.506 | 28.137 | +0.0% |  |  |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward_backward | v6_auto | ok | 51.107 | 51.107 | 23.009 | 28.099 | +47.5% |  | +47.5% |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 42.056 | 42.056 | 8.892 | 33.165 | +21.4% |  | +21.4% |  |
| 4096x4096 | 1 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 53.793 | 53.793 | 25.565 | 28.228 | +55.3% |  | +55.3% |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 11.414 | 11.414 | 11.413 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 23.196 | 23.196 | 23.196 | 0.000 | +103.2% |  | +103.2% |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 13.034 | 13.034 | 13.033 | 0.000 | +14.2% |  | +14.2% |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 25.619 | 25.619 | 25.619 | 0.000 | +124.5% |  | +124.5% |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 34.896 | 34.896 | 6.376 | 28.520 | +0.0% |  |  |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 51.432 | 51.432 | 23.276 | 28.156 | +47.4% |  | +47.4% |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 36.405 | 36.405 | 7.653 | 28.753 | +4.3% |  | +4.3% |  |
| 4096x4096 | 1 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 49.956 | 49.956 | 21.601 | 28.355 | +43.2% |  | +43.2% |  |
| 4096x4096 | 1 | 512 | layered_depth | forward | v6_direct | ok | 10.435 | 10.435 | 10.435 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 512 | layered_depth | forward | v6_auto | ok | 21.830 | 21.830 | 21.830 | 0.000 | +109.2% |  | +109.2% |  |
| 4096x4096 | 1 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 12.209 | 12.209 | 12.209 | 0.000 | +17.0% |  | +17.0% |  |
| 4096x4096 | 1 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 23.141 | 23.141 | 23.141 | 0.000 | +121.8% |  | +121.8% |  |
| 4096x4096 | 1 | 512 | layered_depth | forward_backward | v6_direct | ok | 39.438 | 39.438 | 6.661 | 32.777 | +0.0% |  |  |  |
| 4096x4096 | 1 | 512 | layered_depth | forward_backward | v6_auto | ok | 56.273 | 56.273 | 27.313 | 28.960 | +42.7% |  | +42.7% |  |
| 4096x4096 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 43.720 | 43.720 | 10.615 | 33.105 | +10.9% |  | +10.9% |  |
| 4096x4096 | 1 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 53.758 | 53.758 | 24.323 | 29.435 | +36.3% |  | +36.3% |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward | v6_direct | ok | 16.465 | 16.465 | 16.464 | 0.001 | +19.9% |  |  |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward | v6_auto | ok | 25.302 | 25.302 | 25.302 | 0.001 | +84.2% |  | +53.7% |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 13.734 | 13.734 | 13.734 | 0.000 | +0.0% |  | -16.6% |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 27.925 | 27.925 | 27.925 | 0.000 | +103.3% |  | +69.6% |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 39.598 | 39.598 | 7.709 | 31.889 | +0.0% |  |  |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 58.436 | 58.436 | 26.608 | 31.828 | +47.6% |  | +47.6% |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 47.561 | 47.561 | 10.517 | 37.044 | +20.1% |  | +20.1% |  |
| 4096x4096 | 1 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 57.202 | 57.202 | 25.456 | 31.746 | +44.5% |  | +44.5% |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward | v6_direct | ok | 14.993 | 14.993 | 14.992 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward | v6_auto | ok | 75.887 | 75.887 | 75.886 | 0.000 | +406.2% |  | +406.2% |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 16.054 | 16.054 | 16.054 | 0.000 | +7.1% |  | +7.1% |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 76.773 | 76.773 | 76.773 | 0.000 | +412.1% |  | +412.1% |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward_backward | v6_direct | ok | 129.257 | 129.257 | 14.378 | 114.879 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward_backward | v6_auto | ok | 199.202 | 199.202 | 83.671 | 115.532 | +54.1% |  | +54.1% |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 129.546 | 129.546 | 15.099 | 114.447 | +0.2% |  | +0.2% |  |
| 4096x4096 | 4 | 512 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 197.037 | 197.037 | 82.242 | 114.794 | +52.4% |  | +52.4% |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward | v6_direct | ok | 15.280 | 15.280 | 15.279 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward | v6_auto | ok | 76.271 | 76.271 | 76.271 | 0.000 | +399.2% |  | +399.2% |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward | v6_upgrade_direct | ok | 16.165 | 16.165 | 16.165 | 0.000 | +5.8% |  | +5.8% |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward | v6_upgrade_auto | ok | 76.038 | 76.038 | 76.037 | 0.000 | +397.6% |  | +397.6% |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward_backward | v6_direct | ok | 128.044 | 128.044 | 14.453 | 113.591 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward_backward | v6_auto | ok | 194.442 | 194.442 | 81.170 | 113.273 | +51.9% |  | +51.9% |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 129.672 | 129.672 | 14.792 | 114.880 | +1.3% |  | +1.3% |  |
| 4096x4096 | 4 | 512 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 194.916 | 194.916 | 80.477 | 114.439 | +52.2% |  | +52.2% |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward | v6_direct | ok | 15.853 | 15.853 | 15.853 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward | v6_auto | ok | 76.832 | 76.832 | 76.832 | 0.000 | +384.6% |  | +384.6% |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 16.463 | 16.463 | 16.462 | 0.000 | +3.8% |  | +3.8% |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 77.408 | 77.408 | 77.407 | 0.000 | +388.3% |  | +388.3% |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_direct | ok | 130.018 | 130.018 | 13.947 | 116.071 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_auto | ok | 197.714 | 197.714 | 82.931 | 114.783 | +52.1% |  | +52.1% |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 130.516 | 130.516 | 15.435 | 115.081 | +0.4% |  | +0.4% |  |
| 4096x4096 | 4 | 512 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 194.111 | 194.111 | 79.888 | 114.223 | +49.3% |  | +49.3% |  |
| 4096x4096 | 4 | 512 | layered_depth | forward | v6_direct | ok | 13.335 | 13.335 | 13.334 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | layered_depth | forward | v6_auto | ok | 82.211 | 82.211 | 82.211 | 0.000 | +516.5% |  | +516.5% |  |
| 4096x4096 | 4 | 512 | layered_depth | forward | v6_upgrade_direct | ok | 16.788 | 16.788 | 16.787 | 0.000 | +25.9% |  | +25.9% |  |
| 4096x4096 | 4 | 512 | layered_depth | forward | v6_upgrade_auto | ok | 78.498 | 78.498 | 78.498 | 0.000 | +488.7% |  | +488.7% |  |
| 4096x4096 | 4 | 512 | layered_depth | forward_backward | v6_direct | ok | 129.837 | 129.837 | 14.821 | 115.015 | +0.7% |  |  |  |
| 4096x4096 | 4 | 512 | layered_depth | forward_backward | v6_auto | ok | 199.585 | 199.585 | 83.293 | 116.292 | +54.8% |  | +53.7% |  |
| 4096x4096 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_direct | ok | 128.917 | 128.917 | 15.406 | 113.512 | +0.0% |  | -0.7% |  |
| 4096x4096 | 4 | 512 | layered_depth | forward_backward | v6_upgrade_auto | ok | 196.272 | 196.272 | 81.716 | 114.557 | +52.2% |  | +51.2% |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward | v6_direct | ok | 19.124 | 19.124 | 19.124 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward | v6_auto | ok | 78.995 | 78.995 | 78.995 | 0.000 | +313.1% |  | +313.1% |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_direct | ok | 19.206 | 19.206 | 19.206 | 0.000 | +0.4% |  | +0.4% |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward | v6_upgrade_auto | ok | 80.036 | 80.036 | 80.036 | 0.000 | +318.5% |  | +318.5% |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward_backward | v6_direct | ok | 143.334 | 143.334 | 17.317 | 126.017 | +0.0% |  |  |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward_backward | v6_auto | ok | 210.493 | 210.493 | 84.403 | 126.090 | +46.9% |  | +46.9% |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 146.472 | 146.472 | 18.860 | 127.611 | +2.2% |  | +2.2% |  |
| 4096x4096 | 4 | 512 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 212.397 | 212.397 | 86.195 | 126.202 | +48.2% |  | +48.2% |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 12.170 | 12.170 | 12.170 | 0.000 | +14.6% |  |  |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 13.786 | 13.786 | 13.786 | 0.000 | +29.9% |  | +13.3% |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 10.615 | 10.615 | 10.615 | 0.000 | +0.0% |  | -12.8% |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 25.595 | 25.595 | 25.594 | 0.001 | +141.1% |  | +110.3% |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 37.669 | 37.669 | 7.328 | 30.342 | +2.0% |  |  |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 45.024 | 45.024 | 10.508 | 34.517 | +21.9% |  | +19.5% |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 36.937 | 36.937 | 7.025 | 29.912 | +0.0% |  | -1.9% |  |
| 4096x4096 | 1 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 53.654 | 53.654 | 24.060 | 29.594 | +45.3% |  | +42.4% |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward | v6_direct | ok | 10.900 | 10.900 | 10.900 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward | v6_auto | ok | 28.287 | 28.287 | 28.286 | 0.000 | +159.5% |  | +159.5% |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 12.605 | 12.605 | 12.605 | 0.000 | +15.6% |  | +15.6% |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 25.625 | 25.625 | 25.625 | 0.000 | +135.1% |  | +135.1% |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 41.432 | 41.432 | 7.406 | 34.026 | +16.0% |  |  |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 55.650 | 55.650 | 26.250 | 29.400 | +55.8% |  | +34.3% |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 35.724 | 35.724 | 6.930 | 28.793 | +0.0% |  | -13.8% |  |
| 4096x4096 | 1 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 52.168 | 52.168 | 23.586 | 28.582 | +46.0% |  | +25.9% |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 11.876 | 11.876 | 11.876 | 0.000 | +10.3% |  |  |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 23.407 | 23.407 | 23.407 | 0.000 | +117.5% |  | +97.1% |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 10.763 | 10.763 | 10.762 | 0.000 | +0.0% |  | -9.4% |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 23.900 | 23.900 | 23.900 | 0.000 | +122.1% |  | +101.2% |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 39.627 | 39.627 | 7.562 | 32.065 | +0.0% |  |  |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 52.347 | 52.347 | 23.030 | 29.317 | +32.1% |  | +32.1% |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 40.629 | 40.629 | 8.603 | 32.026 | +2.5% |  | +2.5% |  |
| 4096x4096 | 1 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 57.857 | 57.857 | 25.837 | 32.021 | +46.0% |  | +46.0% |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward | v6_direct | ok | 9.777 | 9.777 | 9.776 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward | v6_auto | ok | 12.568 | 12.568 | 12.568 | 0.000 | +28.5% |  | +28.5% |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 12.274 | 12.274 | 12.274 | 0.001 | +25.5% |  | +25.5% |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 23.820 | 23.820 | 23.820 | 0.000 | +143.6% |  | +143.6% |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward_backward | v6_direct | ok | 38.208 | 38.208 | 7.446 | 30.762 | +3.6% |  |  |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward_backward | v6_auto | ok | 36.880 | 36.880 | 7.597 | 29.282 | +0.0% |  | -3.5% |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 44.768 | 44.768 | 9.659 | 35.109 | +21.4% |  | +17.2% |  |
| 4096x4096 | 1 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 54.315 | 54.315 | 25.089 | 29.227 | +47.3% |  | +42.2% |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward | v6_direct | ok | 12.266 | 12.266 | 12.266 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward | v6_auto | ok | 25.903 | 25.903 | 25.903 | 0.000 | +111.2% |  | +111.2% |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 14.467 | 14.467 | 14.466 | 0.001 | +17.9% |  | +17.9% |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 26.034 | 26.034 | 26.034 | 0.000 | +112.2% |  | +112.2% |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 44.432 | 44.432 | 9.110 | 35.322 | +0.0% |  |  |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 60.457 | 60.457 | 25.830 | 34.627 | +36.1% |  | +36.1% |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 55.867 | 55.867 | 13.941 | 41.925 | +25.7% |  | +25.7% |  |
| 4096x4096 | 1 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 62.472 | 62.472 | 27.829 | 34.643 | +40.6% |  | +40.6% |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward | v6_direct | ok | 15.175 | 15.175 | 15.175 | 0.000 | +1.0% |  |  |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward | v6_auto | ok | 17.069 | 17.069 | 17.068 | 0.000 | +13.6% |  | +12.5% |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 15.021 | 15.021 | 15.021 | 0.000 | +0.0% |  | -1.0% |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 73.662 | 73.662 | 73.662 | 0.000 | +390.4% |  | +385.4% |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_direct | ok | 128.799 | 128.799 | 14.126 | 114.673 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_auto | ok | 129.517 | 129.517 | 15.190 | 114.326 | +0.6% |  | +0.6% |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 130.143 | 130.143 | 14.972 | 115.172 | +1.0% |  | +1.0% |  |
| 4096x4096 | 4 | 2048 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 195.559 | 195.559 | 79.747 | 115.811 | +51.8% |  | +51.8% |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward | v6_direct | ok | 14.201 | 14.201 | 14.201 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward | v6_auto | ok | 71.176 | 71.176 | 71.176 | 0.000 | +401.2% |  | +401.2% |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward | v6_upgrade_direct | ok | 16.229 | 16.229 | 16.229 | 0.001 | +14.3% |  | +14.3% |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward | v6_upgrade_auto | ok | 75.076 | 75.076 | 75.076 | 0.000 | +428.7% |  | +428.7% |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward_backward | v6_direct | ok | 127.143 | 127.143 | 13.846 | 113.296 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward_backward | v6_auto | ok | 187.789 | 187.789 | 75.623 | 112.166 | +47.7% |  | +47.7% |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 128.129 | 128.129 | 14.521 | 113.608 | +0.8% |  | +0.8% |  |
| 4096x4096 | 4 | 2048 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 189.871 | 189.871 | 76.319 | 113.551 | +49.3% |  | +49.3% |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward | v6_direct | ok | 15.673 | 15.673 | 15.672 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward | v6_auto | ok | 71.837 | 71.837 | 71.837 | 0.000 | +358.4% |  | +358.4% |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 18.093 | 18.093 | 18.093 | 0.000 | +15.4% |  | +15.4% |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 71.700 | 71.700 | 71.700 | 0.000 | +357.5% |  | +357.5% |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_direct | ok | 129.901 | 129.901 | 14.618 | 115.283 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_auto | ok | 195.709 | 195.709 | 80.410 | 115.299 | +50.7% |  | +50.7% |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 130.256 | 130.256 | 15.338 | 114.918 | +0.3% |  | +0.3% |  |
| 4096x4096 | 4 | 2048 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 195.171 | 195.171 | 80.855 | 114.316 | +50.2% |  | +50.2% |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward | v6_direct | ok | 17.610 | 17.610 | 17.610 | 0.000 | +2.5% |  |  |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward | v6_auto | ok | 19.696 | 19.696 | 19.696 | 0.000 | +14.6% |  | +11.8% |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward | v6_upgrade_direct | ok | 17.185 | 17.185 | 17.184 | 0.000 | +0.0% |  | -2.4% |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward | v6_upgrade_auto | ok | 90.309 | 90.309 | 90.309 | 0.000 | +425.5% |  | +412.8% |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward_backward | v6_direct | ok | 132.490 | 132.490 | 14.600 | 117.890 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward_backward | v6_auto | ok | 133.468 | 133.468 | 16.396 | 117.072 | +0.7% |  | +0.7% |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_direct | ok | 133.412 | 133.412 | 15.080 | 118.332 | +0.7% |  | +0.7% |  |
| 4096x4096 | 4 | 2048 | layered_depth | forward_backward | v6_upgrade_auto | ok | 205.005 | 205.005 | 84.968 | 120.037 | +54.7% |  | +54.7% |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward | v6_direct | ok | 23.391 | 23.391 | 23.391 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward | v6_auto | ok | 86.243 | 86.243 | 86.243 | 0.000 | +268.7% |  | +268.7% |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_direct | ok | 24.516 | 24.516 | 24.516 | 0.000 | +4.8% |  | +4.8% |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward | v6_upgrade_auto | ok | 87.460 | 87.460 | 87.459 | 0.000 | +273.9% |  | +273.9% |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward_backward | v6_direct | ok | 167.694 | 167.694 | 24.468 | 143.226 | +0.0% |  |  |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward_backward | v6_auto | ok | 233.467 | 233.467 | 92.452 | 141.015 | +39.2% |  | +39.2% |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 168.292 | 168.292 | 25.534 | 142.758 | +0.4% |  | +0.4% |  |
| 4096x4096 | 4 | 2048 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 234.932 | 234.932 | 92.855 | 142.077 | +40.1% |  | +40.1% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 11.752 | 11.752 | 11.751 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 12.915 | 12.915 | 12.915 | 0.000 | +9.9% |  | +9.9% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 12.065 | 12.065 | 12.064 | 0.000 | +2.7% |  | +2.7% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 12.462 | 12.462 | 12.461 | 0.000 | +6.0% |  | +6.0% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 69.749 | 69.749 | 15.181 | 54.568 | +11.3% |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 62.798 | 62.798 | 13.378 | 49.420 | +0.2% |  | -10.0% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 62.657 | 62.657 | 14.119 | 48.538 | +0.0% |  | -10.2% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 68.348 | 68.348 | 15.630 | 52.719 | +9.1% |  | -2.0% |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v6_direct | ok | 10.650 | 10.650 | 10.650 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v6_auto | ok | 12.605 | 12.605 | 12.605 | 0.001 | +18.4% |  | +18.4% |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 11.928 | 11.928 | 11.928 | 0.000 | +12.0% |  | +12.0% |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 27.063 | 27.063 | 27.063 | 0.000 | +154.1% |  | +154.1% |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 52.268 | 52.268 | 10.383 | 41.885 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 59.363 | 59.363 | 13.027 | 46.336 | +13.6% |  | +13.6% |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 63.181 | 63.181 | 14.283 | 48.898 | +20.9% |  | +20.9% |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 69.354 | 69.354 | 28.159 | 41.196 | +32.7% |  | +32.7% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 131.833 | 131.833 | 131.832 | 0.001 | +3.5% |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 144.147 | 144.147 | 144.146 | 0.001 | +13.2% |  | +9.3% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 127.332 | 127.332 | 127.331 | 0.000 | +0.0% |  | -3.4% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 146.567 | 146.567 | 146.567 | 0.001 | +15.1% |  | +11.2% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 184.195 | 184.195 | 135.944 | 48.250 | +4.1% |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 192.942 | 192.942 | 145.777 | 47.166 | +9.0% |  | +4.7% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 176.944 | 176.944 | 129.131 | 47.813 | +0.0% |  | -3.9% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 195.778 | 195.778 | 147.212 | 48.565 | +10.6% |  | +6.3% |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v6_direct | ok | 14.572 | 14.572 | 14.571 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v6_auto | ok | 14.627 | 14.627 | 14.627 | 0.000 | +0.4% |  | +0.4% |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 16.256 | 16.256 | 16.256 | 0.000 | +11.6% |  | +11.6% |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 31.177 | 31.177 | 31.177 | 0.000 | +114.0% |  | +114.0% |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v6_direct | ok | 75.928 | 75.928 | 15.550 | 60.378 | +1.7% |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v6_auto | ok | 77.624 | 77.624 | 16.676 | 60.949 | +4.0% |  | +2.2% |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 74.644 | 74.644 | 14.810 | 59.834 | +0.0% |  | -1.7% |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 88.455 | 88.455 | 31.167 | 57.288 | +18.5% |  | +16.5% |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v6_direct | ok | 585.299 | 585.299 | 585.298 | 0.001 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v6_auto | ok | 601.332 | 601.332 | 601.331 | 0.001 | +2.7% |  | +2.7% |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 1289.412 | 1289.412 | 1289.411 | 0.002 | +120.3% |  | +120.3% |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 628.754 | 628.754 | 628.753 | 0.001 | +7.4% |  | +7.4% |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 1468.462 | 1468.462 | 941.542 | 526.920 | +34.8% |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 1089.448 | 1089.448 | 605.439 | 484.009 | +0.0% |  | -25.8% |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 1154.462 | 1154.462 | 657.740 | 496.722 | +6.0% |  | -21.4% |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 1514.028 | 1514.028 | 1005.059 | 508.969 | +39.0% |  | +3.1% |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 33.943 | 33.943 | 33.943 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v6_auto | ok | 35.467 | 35.467 | 35.467 | 0.000 | +4.5% |  | +4.5% |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 34.292 | 34.292 | 34.292 | 0.000 | +1.0% |  | +1.0% |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 34.555 | 34.555 | 34.555 | 0.000 | +1.8% |  | +1.8% |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 231.469 | 231.469 | 39.725 | 191.745 | +0.7% |  |  |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_auto | ok | 229.856 | 229.856 | 39.869 | 189.987 | +0.0% |  | -0.7% |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 230.142 | 230.142 | 40.553 | 189.589 | +0.1% |  | -0.6% |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 229.991 | 229.991 | 40.226 | 189.765 | +0.1% |  | -0.6% |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v6_direct | ok | 25.037 | 25.037 | 25.036 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v6_auto | ok | 28.029 | 28.029 | 28.029 | 0.000 | +12.0% |  | +12.0% |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v6_upgrade_direct | ok | 27.424 | 27.424 | 27.423 | 0.001 | +9.5% |  | +9.5% |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v6_upgrade_auto | ok | 86.221 | 86.221 | 86.221 | 0.000 | +244.4% |  | +244.4% |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward_backward | v6_direct | ok | 197.872 | 197.872 | 27.123 | 170.749 | +4.9% |  |  |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward_backward | v6_auto | ok | 191.498 | 191.498 | 28.527 | 162.971 | +1.5% |  | -3.2% |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 188.677 | 188.677 | 26.654 | 162.023 | +0.0% |  | -4.6% |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 252.630 | 252.630 | 92.840 | 159.789 | +33.9% |  | +27.7% |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 1381.861 | 1381.861 | 1381.861 | 0.001 | +68.5% |  |  |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v6_auto | ok | 1006.525 | 1006.525 | 1006.524 | 0.001 | +22.8% |  | -27.2% |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 1075.467 | 1075.467 | 1075.466 | 0.001 | +31.2% |  | -22.2% |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 819.863 | 819.863 | 819.862 | 0.001 | +0.0% |  | -40.7% |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 684.783 | 684.783 | 501.565 | 183.218 | +0.0% |  |  |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_auto | ok | 754.457 | 754.457 | 572.034 | 182.423 | +10.2% |  | +10.2% |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 925.505 | 925.505 | 723.378 | 202.127 | +35.2% |  | +35.2% |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 1070.714 | 1070.714 | 864.576 | 206.138 | +56.4% |  | +56.4% |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v6_direct | ok | 39.768 | 39.768 | 39.768 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v6_auto | ok | 40.326 | 40.326 | 40.326 | 0.000 | +1.4% |  | +1.4% |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v6_upgrade_direct | ok | 39.836 | 39.836 | 39.836 | 0.000 | +0.2% |  | +0.2% |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v6_upgrade_auto | ok | 102.303 | 102.303 | 102.303 | 0.000 | +157.3% |  | +157.3% |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward_backward | v6_direct | ok | 271.098 | 271.098 | 44.208 | 226.890 | +0.4% |  |  |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward_backward | v6_auto | ok | 269.955 | 269.955 | 43.076 | 226.879 | +0.0% |  | -0.4% |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_direct | ok | 270.532 | 270.532 | 44.798 | 225.734 | +0.2% |  | -0.2% |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward_backward | v6_upgrade_auto | ok | 331.298 | 331.298 | 111.399 | 219.899 | +22.7% |  | +22.2% |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v6_direct | ok | 5253.447 | 5253.447 | 5253.446 | 0.001 | +20.5% |  |  |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v6_auto | ok | 4358.096 | 4358.096 | 4358.093 | 0.003 | +0.0% |  | -17.0% |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_direct | ok | 4519.327 | 4519.327 | 4519.324 | 0.002 | +3.7% |  | -14.0% |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v6_upgrade_auto | ok | 4674.138 | 4674.138 | 4674.137 | 0.001 | +7.3% |  | -11.0% |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward_backward | v6_direct | ok | 8928.235 | 8928.235 | 6799.099 | 2129.136 | +18.2% |  |  |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward_backward | v6_auto | ok | 8367.094 | 8367.094 | 6242.504 | 2124.589 | +10.7% |  | -6.3% |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 7555.238 | 7555.238 | 5426.138 | 2129.100 | +0.0% |  | -15.4% |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 8054.878 | 8054.878 | 5902.705 | 2152.174 | +6.6% |  | -9.8% |  |
