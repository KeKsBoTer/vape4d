import argparse
import os
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render scales for a given model.")
    parser.add_argument(
        "--volume",
        type=str,
        required=True,
        help="Path to the volume file",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        required=True,
        help="cmap file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        dest="output_dir",
        help="output folder",
    )

    args = parser.parse_args()

    methods = ["Nearest", "Bicubic", "Lanczos", "Spline"]
    scales = range(1,9)

    for scale in scales:
        for method in methods:
            if scale == 1 and method != "Nearest":
                continue
            out_folder = os.path.join(args.output_dir, f"{method}_{scale}")
            os.makedirs(out_folder, exist_ok=True)
            subprocess.run(
                [
                    "cargo", "run", "--release", "--bin", "images",
                    args.volume,
                    out_folder,
                    args.cmap,
                    "--render-scale", str(scale),
                    "--upscaling", method
                ]
            )