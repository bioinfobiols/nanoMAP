from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from joint.cli import app

runner = CliRunner()


def test_help_lists_all_stage_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "run",
        "preprocess",
        "segment",
        "register",
        "quantify",
        "qc",
        "annotate",
        "analyze",
        "trajectory",
    ]:
        assert command in result.stdout


def test_run_forwards_resume_and_overwrite(monkeypatch, tmp_path: Path):
    calls = []

    class FakePipeline:
        @classmethod
        def from_config(cls, path):
            calls.append(("config", Path(path)))
            return cls()

        def run(self, *, resume, overwrite):
            calls.append(("run", resume, overwrite))

    monkeypatch.setattr("joint.cli.JointPipeline", FakePipeline)
    config = tmp_path / "config.yaml"
    config.write_text("config")
    result = runner.invoke(app, ["run", "--config", str(config), "--resume", "--overwrite"])
    assert result.exit_code == 0
    assert calls[-1] == ("run", True, True)


def test_domain_error_returns_nonzero_status(monkeypatch, tmp_path: Path):
    class BrokenPipeline:
        @classmethod
        def from_config(cls, path):
            raise RuntimeError("bad config")

    monkeypatch.setattr("joint.cli.JointPipeline", BrokenPipeline)
    bad = tmp_path / "bad.yaml"
    bad.write_text("bad")
    result = runner.invoke(app, ["run", "--config", str(bad)])
    assert result.exit_code != 0
    assert "bad config" in result.stderr


@pytest.mark.parametrize(
    ("command", "expected_calls"),
    [
        ("preprocess", [("preprocess", False, True)]),
        (
            "segment",
            [("preprocess", True, False), ("segment", "msi", False, True)],
        ),
        (
            "register",
            [
                ("preprocess", True, False),
                ("segment", "msi", True, False),
                ("register", "msi", "segmented", False, True),
            ],
        ),
        (
            "quantify",
            [
                ("preprocess", True, False),
                ("segment", "msi", True, False),
                ("register", "msi", "segmented", True, False),
                ("quantify", "registered", "segmented", False, True),
            ],
        ),
        ("qc", [("spatial", True, False), ("qc", "cells", False, True)]),
    ],
)
def test_stage_commands_resume_prerequisites(monkeypatch, tmp_path: Path, command, expected_calls):
    calls = []

    class FakePipeline:
        config = SimpleNamespace(
            qc=SimpleNamespace(enabled=False), annotation=SimpleNamespace(enabled=False)
        )

        @classmethod
        def from_config(cls, path):
            return cls()

        def preprocess(self, *, resume, overwrite):
            calls.append(("preprocess", resume, overwrite))
            return "msi"

        def segment(self, msi, *, resume, overwrite):
            calls.append(("segment", msi, resume, overwrite))
            return "segmented"

        def register(self, msi, segmented, *, resume, overwrite):
            calls.append(("register", msi, segmented, resume, overwrite))
            return "registered"

        def quantify(self, registered, segmented, *, resume, overwrite):
            calls.append(("quantify", registered, segmented, resume, overwrite))

        def run_spatial_core(self, *, resume, overwrite):
            calls.append(("spatial", resume, overwrite))
            return SimpleNamespace(adata="cells")

        def qc(self, cells, *, resume, overwrite):
            calls.append(("qc", cells, resume, overwrite))

    monkeypatch.setattr("joint.cli.JointPipeline", FakePipeline)
    config = tmp_path / "config.yaml"
    config.write_text("config")
    result = runner.invoke(app, [command, "--config", str(config), "--overwrite"])
    assert result.exit_code == 0
    assert calls == expected_calls


@pytest.mark.parametrize(
    ("command", "expected_calls"),
    [
        (
            "annotate",
            [
                ("spatial", True, False),
                ("qc", "cells", True, False),
                ("annotate", "qc-cells", False, True),
            ],
        ),
        (
            "analyze",
            [
                ("spatial", True, False),
                ("qc", "cells", True, False),
                ("annotate", "qc-cells", True, False),
                ("analyze", "annotated-cells", False, True),
            ],
        ),
        (
            "trajectory",
            [
                ("spatial", True, False),
                ("qc", "cells", True, False),
                ("annotate", "qc-cells", True, False),
                ("analyze", "annotated-cells", True, False),
                ("trajectory", "analyzed-cells", False, True),
            ],
        ),
    ],
)
def test_late_stage_commands_resume_enabled_prerequisites(
    monkeypatch, tmp_path: Path, command, expected_calls
):
    calls = []

    class FakePipeline:
        config = SimpleNamespace(
            qc=SimpleNamespace(enabled=True), annotation=SimpleNamespace(enabled=True)
        )

        @classmethod
        def from_config(cls, path):
            return cls()

        def run_spatial_core(self, *, resume, overwrite):
            calls.append(("spatial", resume, overwrite))
            return SimpleNamespace(adata="cells")

        def qc(self, cells, *, resume, overwrite):
            calls.append(("qc", cells, resume, overwrite))
            return "qc-cells"

        def annotate(self, cells, *, resume, overwrite):
            calls.append(("annotate", cells, resume, overwrite))
            return "annotated-cells"

        def analyze(self, cells, *, resume, overwrite):
            calls.append(("analyze", cells, resume, overwrite))
            return "analyzed-cells"

        def trajectory(self, cells, *, resume, overwrite):
            calls.append(("trajectory", cells, resume, overwrite))

    monkeypatch.setattr("joint.cli.JointPipeline", FakePipeline)
    config = tmp_path / "config.yaml"
    config.write_text("config")
    result = runner.invoke(app, [command, "--config", str(config), "--overwrite"])
    assert result.exit_code == 0
    assert calls == expected_calls
