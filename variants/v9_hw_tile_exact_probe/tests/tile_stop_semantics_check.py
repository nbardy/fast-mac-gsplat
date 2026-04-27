from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_hw_tile_exact import assert_v8_candidate_prefix_gap_exposed


def main() -> None:
    reports = assert_v8_candidate_prefix_gap_exposed()
    by_case = {report["case"]: report for report in reports}

    skipped = by_case["one_tile_skipped_candidate"]
    assert skipped["v8_tile_candidates"] == [[0, 1]], skipped
    assert skipped["v8_candidate_prefix_tile_stop_counts"] == [2], skipped
    assert skipped["v9_fullscreen_draw_candidates"] == [[0, 1, 2]], skipped
    assert skipped["v9_fullscreen_diagnostic_tile_stop_counts"] == [3], skipped

    clipping = by_case["two_tile_clipping_gap"]
    assert clipping["v8_tile_candidates"] == [[0], [1]], clipping
    assert clipping["v8_candidate_prefix_tile_stop_counts"] == [1, 1], clipping
    assert clipping["v9_fullscreen_draw_candidates"] == [[0, 1, 2], [0, 1, 2]], clipping
    assert clipping["v9_fullscreen_diagnostic_tile_stop_counts"] == [3, 3], clipping

    print(json.dumps({"v9_v8_candidate_prefix_gap": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
