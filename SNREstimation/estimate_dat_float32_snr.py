#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from iq_readers import read_float32_iq
from snr_core import add_common_snr_args, run_snr_estimation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate SNR from one float32 interleaved IQ .dat file.")
    add_common_snr_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_snr_estimation(args, read_float32_iq)


if __name__ == "__main__":
    main()
