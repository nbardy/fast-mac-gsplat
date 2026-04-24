
# Torch+Metal gsplat v6 refined

This bundle is the corrected v6 handoff for Apple Silicon / PyTorch MPS.

Integration note: the raw archive is preserved under
`source_artifacts/torch_metal_gsplat_v6_refined.tar.gz`. This checked-in
variant additionally carries the repo's saturated-backward barrier fix used by
`variants/v6` and `variants/v6_upgrade`.

What is different from the earlier v6 branch:
- direct tile kernels are the default fast path
- active-tile scheduling is opt-in or auto-selected, not the default
- benchmarks include uniform, sparse-screen, clustered, layered-depth, overflow, temporal, and real-trace scenarios
- same-process alternating compare harness is included for fair v5/v6 comparisons
- reference checks cover both direct and active modes

Recommended default:
- `active_policy='off'`
- only enable `active_policy='auto'` or `--active-tiles` for sparse-screen / overflow-heavy workloads

Quick start:
```bash
python setup.py build_ext --inplace
python tests/reference_check.py
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case medium_sigma_3_8 --backward --profile
python benchmarks/benchmark_compare.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case medium_sigma_3_8 --backward --shuffle-order
```
