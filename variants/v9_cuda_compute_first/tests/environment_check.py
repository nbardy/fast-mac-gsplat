import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_gsplat_bridge_v9_cuda_compute_first import cuda_environment


def main() -> None:
    print(json.dumps(cuda_environment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
