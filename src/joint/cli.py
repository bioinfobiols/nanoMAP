"""Command-line access to the checkpointed JOINT pipeline."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from joint.pipeline import JointPipeline

app = typer.Typer(no_args_is_help=True, help="JOINT spatial MSI analysis")


def _pipeline(config: Path) -> JointPipeline:
    return JointPipeline.from_config(config)


def _execute(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


_CONFIG_OPTION = typer.Option(..., exists=True, dir_okay=False)


@app.command()
def run(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Run every configured pipeline stage."""
    _execute(lambda: _pipeline(config).run(resume=resume, overwrite=overwrite))


@app.command()
def preprocess(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Preprocess the MSI input."""
    _execute(lambda: _pipeline(config).preprocess(resume=resume, overwrite=overwrite))


@app.command()
def segment(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Segment laser marks after resuming preprocessing."""

    def action() -> None:
        pipeline = _pipeline(config)
        msi = pipeline.preprocess(resume=True, overwrite=False)
        pipeline.segment(msi, resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def register(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Register segmented laser marks after resuming prerequisites."""

    def action() -> None:
        pipeline = _pipeline(config)
        msi = pipeline.preprocess(resume=True, overwrite=False)
        segmented = pipeline.segment(msi, resume=True, overwrite=False)
        pipeline.register(msi, segmented, resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def quantify(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Quantify cells after resuming spatial-core prerequisites."""

    def action() -> None:
        pipeline = _pipeline(config)
        msi = pipeline.preprocess(resume=True, overwrite=False)
        segmented = pipeline.segment(msi, resume=True, overwrite=False)
        registered = pipeline.register(msi, segmented, resume=True, overwrite=False)
        pipeline.quantify(registered, segmented, resume=resume, overwrite=overwrite)

    _execute(action)


def _prepared_cells(pipeline: JointPipeline, target: str) -> Any:
    result = pipeline.run_spatial_core(resume=True, overwrite=False).adata
    if target in {"qc", "annotate", "analyze", "trajectory"} and pipeline.config.qc.enabled:
        result = pipeline.qc(result, resume=True, overwrite=False)
    if target in {"analyze", "trajectory"} and pipeline.config.annotation.enabled:
        result = pipeline.annotate(result, resume=True, overwrite=False)
    if target == "trajectory":
        result = pipeline.analyze(result, resume=True, overwrite=False)
    return result


@app.command()
def qc(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Perform spectral quality control after resuming the spatial core."""

    def action() -> None:
        pipeline = _pipeline(config)
        cells = pipeline.run_spatial_core(resume=True, overwrite=False).adata
        pipeline.qc(cells, resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def annotate(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Annotate cells after resuming enabled prerequisites."""

    def action() -> None:
        pipeline = _pipeline(config)
        pipeline.annotate(_prepared_cells(pipeline, "annotate"), resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def analyze(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Analyze cells after resuming enabled prerequisites."""

    def action() -> None:
        pipeline = _pipeline(config)
        pipeline.analyze(_prepared_cells(pipeline, "analyze"), resume=resume, overwrite=overwrite)

    _execute(action)


@app.command()
def trajectory(
    config: Path = _CONFIG_OPTION,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Fit trajectory trends after resuming enabled prerequisites."""

    def action() -> None:
        pipeline = _pipeline(config)
        pipeline.trajectory(
            _prepared_cells(pipeline, "trajectory"), resume=resume, overwrite=overwrite
        )

    _execute(action)
