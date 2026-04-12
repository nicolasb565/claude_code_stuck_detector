"""Entry point: load manifest → train → eval.

Usage:
  python train.py [--manifest training_manifest.json]
"""

import argparse
import sys

sys.path.insert(0, ".")  # pylint: disable=wrong-import-position

from src.training.train import (
    train,
)  # noqa: E402  # pylint: disable=wrong-import-position


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="training_manifest.json")
    args = parser.parse_args()
    train(args.manifest)


if __name__ == "__main__":
    main()
