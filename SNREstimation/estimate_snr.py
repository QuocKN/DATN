#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from iq_readers import read_interleaved_iq
from snr_core import add_common_snr_args, run_snr_estimation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate SNR from one interleaved IQ .bin/.dat/.iq file. "
            "Use estimate_bin_snr.py or estimate_dat_float32_snr.py for format-specific entrypoints."
        )
    )
    add_common_snr_args(parser)
    parser.add_argument(
        "--dtype",
        choices=["float32_iq", "int16_iq"],
        default="float32_iq",
        help="Sample format: interleaved I,Q values (default: float32_iq)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize int16 IQ by 32768 before analysis. This does not change SNR, only scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def reader(path: Path, max_iq_samples: int | None):
        return read_interleaved_iq(
            path,
            dtype=args.dtype,
            max_iq_samples=max_iq_samples,
            normalize=args.normalize,
        )

    run_snr_estimation(args, reader)


if __name__ == "__main__":
    main()
