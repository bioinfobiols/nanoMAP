"""Run the bundled real d2 analysis and figures from a JOINT source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joint import JointPipeline, load_config
from scripts.render_d2_figures import D2DemoError, render

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = (
    "adatas/d2.h5ad",
    "pics/02/white.jpeg",
    "rawdata/02.imzML",
    "rawdata/02.ibd",
    "segementation_d2/segmentation_cell_raw.npz",
)


def verify_d2_data(data_root: str | Path | None = None) -> Path:
    """Check that all bundled real d2 inputs match the distribution manifest."""
    root = ROOT / "data/d2" if data_root is None else Path(data_root).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if set(manifest["files"]) != set(DATA_FILES):
            raise ValueError("unexpected file inventory")
        for name in DATA_FILES:
            path = root / name
            if not path.is_file():
                raise D2DemoError(f"Missing d2 input: {path}. Extract the complete JOINT source ZIP.")
            record = manifest["files"][name]
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
                raise D2DemoError(f"d2 input checksum mismatch: {path}. Restore the original file.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise D2DemoError(f"Cannot verify bundled d2 data in {root}: {exc}") from exc
    return root


def run_d2_demo(
    *, output_dir: str | Path | None = None, overwrite: bool = False
) -> Path:
    """Validate d2 inputs, run all configured stages, and render 371 PNG figures.

    By default, resume valid existing results in the source checkout's results/d2.
    Set overwrite=True only when existing artifacts may be replaced.
    """
    data = verify_d2_data()
    config = load_config(ROOT / "configs/d2.yaml")
    if output_dir is not None:
        config.project.output_dir = Path(output_dir).expanduser().resolve()
    result = JointPipeline(config).run(resume=not overwrite, overwrite=overwrite)
    if result.shape != (2936, 360):
        raise D2DemoError(f"Unexpected d2 result shape: {result.shape}; expected (2936, 360)")
    run = config.project.output_dir
    render(data_root=data, run_root=run)
    manifest = json.loads((run / "d2-figures-manifest.json").read_text(encoding="utf-8"))
    # The renderer records all overview and individual feature PNGs.
    actual = {str(path.relative_to(run)) for path in (run / "figures").rglob("*.png")}
    if (manifest["feature_maps"] != 360 or len(actual) != 371
            or actual != set(manifest["files"])):
        raise D2DemoError("Incomplete d2 figures; expected 371 PNGs including 360 feature maps")
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: results/d2)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing analysis outputs")
    parser.add_argument("--check-data", action="store_true", help="Verify bundled d2 checksums only")
    args = parser.parse_args(argv)
    if args.check_data:
        print(f"Verified real d2 dataset: {verify_d2_data()}")
    else:
        print(f"d2 analysis and figures: {run_d2_demo(output_dir=args.output_dir, overwrite=args.overwrite)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
