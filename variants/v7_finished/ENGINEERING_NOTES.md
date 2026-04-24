# v7 hardware handoff notes

This package is a Torch custom-op source handoff for a hardware-forward Gaussian rasterizer on Apple Silicon.

What is present:
- Python API and autograd wrapper
- Torch custom-op registration
- Objective-C++ bridge
- Metal vertex + fragment forward path
- Metal compute backward kernel
- reference and benchmark scripts
- packaging files (`setup.py`, `pyproject.toml`, `__init__.py`)

What is still risky / unfinished:
- no Apple-machine validation in this handoff session
- the current backward design replays all gaussians per pixel and is expected to be too slow for serious training
- real performance must be measured on Mac against v3/v5

Practical status:
- source-complete handoff
- not benchmark-proven
- use for engineering exploration, not as the current production training path
