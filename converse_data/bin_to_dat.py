import argparse
import os
import shutil
import numpy as np


def convert_bin_to_dat(input_path, output_path, mode="copy", normalize=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if mode == "copy":
        # Fast path: .dat is just another binary container extension.
        shutil.copyfile(input_path, output_path)
        return

    if mode == "float32":
        data = np.fromfile(input_path, dtype=np.float32)
        data.tofile(output_path)
        return

    if mode == "int16_iq_to_float32":
        raw = np.fromfile(input_path, dtype=np.int16)
        if raw.size % 2 != 0:
            raise ValueError("IQ int16 data must have even number of samples.")

        if normalize:
            out = (raw.astype(np.float32) / 32768.0)
        else:
            out = raw.astype(np.float32)

        out.tofile(output_path)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Convert BIN file to DAT file")
    parser.add_argument("-i", "--input", required=True, help="Input .bin path")
    parser.add_argument("-o", "--output", required=True, help="Output .dat path")
    parser.add_argument(
        "-m",
        "--mode",
        default="copy",
        choices=["copy", "float32", "int16_iq_to_float32"],
        help="Conversion mode (default: copy)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Only used with int16_iq_to_float32 mode: scale to [-1, 1)",
    )

    args = parser.parse_args()
    convert_bin_to_dat(args.input, args.output, args.mode, args.normalize)

    print(f"Done: {args.input} -> {args.output} (mode={args.mode})")


if __name__ == "__main__":
    main()
