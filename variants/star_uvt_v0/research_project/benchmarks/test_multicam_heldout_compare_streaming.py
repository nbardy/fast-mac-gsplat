from __future__ import annotations

import torch

from research_project.benchmarks.multicam_heldout_compare import (
    VideoMetricAccumulator,
    append_chunk_media,
    media_frame_positions,
    video_metrics,
)


def test_streamed_video_metrics_match_full_tensor_metrics() -> None:
    generator = torch.Generator(device="cpu").manual_seed(17)
    target = torch.rand((7, 12, 16, 3), generator=generator)
    rendered = (target + 0.05 * torch.randn(target.shape, generator=generator)).clamp(0.0, 1.0)
    accumulator = VideoMetricAccumulator()

    for start, stop in ((0, 3), (3, 5), (5, 7)):
        accumulator.update(rendered[start:stop], target[start:stop])

    expected = video_metrics(rendered, target)
    actual = accumulator.metrics()
    for key in expected:
        assert abs(actual[key] - expected[key]) < 5.0e-6


def test_chunk_media_retains_only_the_bounded_global_positions() -> None:
    target = torch.arange(10 * 2 * 2 * 3, dtype=torch.float32).reshape(10, 2, 2, 3)
    rendered = target + 1.0
    alpha = torch.ones((10, 2, 2))
    selected = media_frame_positions(10, 4)
    targets_out: list[torch.Tensor] = []
    rendered_out: list[torch.Tensor] = []
    alpha_out: list[torch.Tensor] = []

    for start, stop in ((0, 3), (3, 7), (7, 10)):
        append_chunk_media(
            start=start,
            stop=stop,
            selected=selected,
            target=target[start:stop],
            rendered=rendered[start:stop],
            alpha=alpha[start:stop],
            targets_out=targets_out,
            rendered_out=rendered_out,
            alpha_out=alpha_out,
        )

    positions = sorted(selected)
    assert torch.equal(torch.cat(targets_out), target[positions])
    assert torch.equal(torch.cat(rendered_out), rendered[positions])
    assert torch.equal(torch.cat(alpha_out), alpha[positions])
