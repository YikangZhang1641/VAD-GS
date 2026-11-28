"""Generate only the offline VGGT visibility cache in its inference environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from script.generate_vggt_visibility import _load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    generation = config.get("data", {}).get("vggt_prediction_generation", {})
    conda_env = generation.get("conda_env")
    generator = Path(__file__).resolve().parent / "script" / "generate_vggt_visibility.py"
    if conda_env:
        command = [os.environ.get("CONDA_EXE") or "conda", "run", "--no-capture-output", "-n", str(conda_env), "python"]
    else:
        command = [sys.executable]
    command += [str(generator), "--config", str(config_path)]
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    print("Running offline VGGT generation:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
