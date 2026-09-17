from pathlib import Path

import pytest
import yaml

from joint.config import (
    InputConfig,
    JointConfig,
    ProjectConfig,
    load_config,
    write_resolved_config,
)
from joint.errors import ConfigurationError

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_load_config_resolves_paths_relative_to_yaml(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "sample.yaml"
    config_path.write_text(
        """
project:
  name: sample
  output_dir: ../results/sample
input:
  imzml: ../data/sample.imzML
laser_segmentation:
  method: simulated
  radius: 18
quantification:
  method: specificity_filtered
""".strip()
    )

    config = load_config(config_path)

    assert config.project.output_dir == (tmp_path / "results/sample").resolve()
    assert config.input.imzml == (tmp_path / "data/sample.imzML").resolve()
    assert config.laser_segmentation.radius == 18


def test_invalid_quantification_method_has_domain_error(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
project: {name: bad, output_dir: results}
input: {h5ad: input.h5ad}
quantification: {method: unsupported}
""".strip()
    )

    with pytest.raises(ConfigurationError, match="unsupported"):
        load_config(config_path)


def test_write_resolved_config_emits_absolute_paths(tmp_path: Path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        "project: {name: sample, output_dir: results}\ninput: {h5ad: input.h5ad}\n"
    )
    config = load_config(config_path)

    written = write_resolved_config(config, tmp_path / "resolved.yaml")

    assert written.is_file()
    assert str((tmp_path / "input.h5ad").resolve()) in written.read_text()


def test_write_resolved_config_resolves_relative_paths_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    config = JointConfig(
        project=ProjectConfig(name="sample", output_dir=Path("results")),
        input=InputConfig(h5ad=Path("input.h5ad")),
    )

    written = write_resolved_config(config, tmp_path / "resolved.yaml")

    output = written.read_text()
    assert str((tmp_path / "results").resolve()) in output
    assert str((tmp_path / "input.h5ad").resolve()) in output
    assert config.project.output_dir == Path("results")
    assert config.input.h5ad == Path("input.h5ad")


def test_load_config_expands_absolute_environment_path(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "reference"
    monkeypatch.setenv("JOINT_DATA_ROOT", str(data_root))
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        "project: {name: sample, output_dir: results}\n"
        'input: {h5ad: "${JOINT_DATA_ROOT}/adatas/d2.h5ad"}\n'
    )

    config = load_config(config_path)

    assert config.input.h5ad == (data_root / "adatas/d2.h5ad").resolve()


def test_load_config_expands_relative_and_multiple_environment_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("JOINT_DATA_ROOT", "reference")
    monkeypatch.setenv("DATASET", "d2")
    config_path = tmp_path / "configs" / "sample.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "project: {name: sample, output_dir: results}\n"
        'input: {h5ad: "$JOINT_DATA_ROOT/adatas/${DATASET}.h5ad"}\n'
    )

    config = load_config(config_path)

    assert config.input.h5ad == (config_path.parent / "reference/adatas/d2.h5ad").resolve()


def test_load_config_expands_environment_variables_on_new_path_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("JOINT_REF_ROOT", str(tmp_path / "refs"))
    path = tmp_path / "sample.yaml"
    path.write_text(
        "project: {name: sample, output_dir: results}\n"
        "input: {h5ad: input.h5ad}\n"
        'annotation: {hmdb_reference: "${JOINT_REF_ROOT}/hmdb.csv"}\n'
        'trajectory: {path_file: "$JOINT_REF_ROOT/trajectory.csv"}\n'
    )

    config = load_config(path)

    assert config.annotation.hmdb_reference == (tmp_path / "refs/hmdb.csv").resolve()
    assert config.trajectory.path_file == (tmp_path / "refs/trajectory.csv").resolve()


@pytest.mark.parametrize("syntax", ["${JOINT_DATA_ROOT}", "$JOINT_DATA_ROOT"])
def test_load_config_rejects_undefined_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, syntax: str
):
    monkeypatch.delenv("JOINT_DATA_ROOT", raising=False)
    path = tmp_path / "sample.yaml"
    path.write_text(
        f"project: {{name: sample, output_dir: results}}\ninput: {{h5ad: '{syntax}/d2.h5ad'}}\n"
    )

    with pytest.raises(ConfigurationError, match="JOINT_DATA_ROOT"):
        load_config(path)


def test_load_config_rejects_malformed_environment_variable_reference(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text(
        "project: {name: sample, output_dir: results}\ninput: {h5ad: '${JOINT_DATA_ROOT/d2.h5ad'}\n"
    )

    with pytest.raises(ConfigurationError, match="Malformed environment variable"):
        load_config(path)


def test_load_config_does_not_expand_environment_variables_in_non_path_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("JOINT_PROJECT", "expanded")
    path = tmp_path / "sample.yaml"
    path.write_text(
        'project: {name: "$JOINT_PROJECT", output_dir: results}\ninput: {h5ad: input.h5ad}\n'
    )

    config = load_config(path)

    assert config.project.name == "$JOINT_PROJECT"


def test_load_config_does_not_expand_environment_variables_in_features_or_groupby(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("JOINT_GROUP", "expanded")
    path = tmp_path / "sample.yaml"
    path.write_text(
        "project: {name: sample, output_dir: results}\n"
        "input: {h5ad: input.h5ad}\n"
        'analysis: {cosg: {groupby: "$JOINT_GROUP"}}\n'
        'trajectory: {features: ["$JOINT_GROUP"]}\n'
    )

    config = load_config(path)

    assert config.analysis.cosg.groupby == "$JOINT_GROUP"
    assert config.trajectory.features == ["$JOINT_GROUP"]


def test_load_config_does_not_expand_environment_variables_in_untyped_parameters(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text(
        "project: {name: sample, output_dir: results}\n"
        "input: {h5ad: input.h5ad}\n"
        "laser_segmentation:\n"
        "  parameters: {h5ad: '$JOINT_DATA_ROOT/metadata'}\n"
    )

    config = load_config(path)

    assert config.laser_segmentation.parameters["h5ad"] == "$JOINT_DATA_ROOT/metadata"


def test_downstream_sections_have_stable_defaults(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text("project: {name: sample, output_dir: results}\ninput: {h5ad: input.h5ad}\n")

    config = load_config(path)
    other = load_config(path)

    assert config.qc.enabled is True
    assert config.qc.outlier_contamination == 0.05
    assert config.clustering.resolution == 0.6
    assert config.annotation.enabled is False
    assert config.trajectory.enabled is False
    config.trajectory.features.append("mz_100")
    assert other.trajectory.features == []


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("qc: {outlier_contamination: 0}", "outlier_contamination"),
        ("annotation: {ion_mode: invalid}", "ion_mode"),
        ("clustering: {n_neighbors: 1}", "n_neighbors"),
        ("trajectory: {points: 1}", "points"),
        ("qc: {unexpected: true}", "unexpected"),
    ],
)
def test_load_config_rejects_invalid_downstream_configuration(
    tmp_path: Path, section: str, value: str
):
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"project: {{name: bad, output_dir: results}}\ninput: {{h5ad: input.h5ad}}\n{section}\n"
    )

    with pytest.raises(ConfigurationError, match=value):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("qc: {outlier_contamination: 0.5}", "outlier_contamination"),
        ("annotation: {hmdb_ppm: 0}", "hmdb_ppm"),
        ("annotation: {metaboscape_ppm: 0}", "metaboscape_ppm"),
        ("clustering: {resolution: 0}", "resolution"),
        ("clustering: {n_top_features: 0}", "n_top_features"),
        ("trajectory: {features: [1]}", "features"),
    ],
)
def test_load_config_rejects_downstream_boundary_and_type_values(
    tmp_path: Path, section: str, field: str
):
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"project: {{name: bad, output_dir: results}}\ninput: {{h5ad: input.h5ad}}\n{section}\n"
    )

    with pytest.raises(ConfigurationError, match=field):
        load_config(path)


def test_load_config_rejects_non_string_path_values(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("project: {name: bad, output_dir: results}\ninput: {h5ad: [not, a, path]}\n")

    with pytest.raises(ConfigurationError, match="h5ad"):
        load_config(path)


def test_load_config_wraps_malformed_yaml_without_parser_details(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("project: [name: sample\ninput: {h5ad: input.h5ad}\n")

    with pytest.raises(ConfigurationError, match="Malformed YAML configuration") as error:
        load_config(path)

    assert "while parsing" not in str(error.value)


def test_load_config_wraps_configuration_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "sample.yaml"
    path.write_text("project: {name: sample, output_dir: results}\ninput: {h5ad: input.h5ad}\n")

    def fail_to_read(_: Path) -> str:
        raise OSError("test-only read failure")

    monkeypatch.setattr(Path, "read_text", fail_to_read)

    with pytest.raises(ConfigurationError, match="Unable to read configuration") as error:
        load_config(path)

    assert "test-only" not in str(error.value)


def test_input_trusted_pickle_defaults_to_false_and_accepts_explicit_boolean(tmp_path: Path):
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "project: {name: default, output_dir: results}\ninput: {h5ad: input.h5ad}\n"
    )
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(
        "project: {name: explicit, output_dir: results}\n"
        "input: {h5ad: input.h5ad, trusted_pickle: true}\n"
    )

    assert load_config(default_path).input.trusted_pickle is False
    assert load_config(explicit_path).input.trusted_pickle is True


@pytest.mark.parametrize(
    ("input_value", "section", "field"),
    [
        ('trusted_pickle: "true"', "", "trusted_pickle"),
        ("", 'clustering: {n_neighbors: "15"}', "n_neighbors"),
    ],
)
def test_load_config_rejects_string_values_for_typed_configuration(
    tmp_path: Path, input_value: str, section: str, field: str
):
    path = tmp_path / "bad.yaml"
    input_line = "input: {h5ad: input.h5ad"
    if input_value:
        input_line = f"{input_line}, {input_value}"
    path.write_text(f"project: {{name: bad, output_dir: results}}\n{input_line}}}\n{section}\n")

    with pytest.raises(ConfigurationError, match=field):
        load_config(path)


def test_load_config_resolves_new_path_keys_relative_to_yaml(tmp_path: Path):
    config_path = tmp_path / "configs" / "sample.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "project: {name: sample, output_dir: results}\n"
        "input: {h5ad: input.h5ad}\n"
        "annotation:\n"
        "  hmdb_reference: refs/hmdb.csv\n"
        "  metaboscape_reference: refs/metaboscape.csv\n"
        "trajectory: {path_file: paths/trajectory.csv}\n"
    )

    config = load_config(config_path)

    assert config.annotation.hmdb_reference == (config_path.parent / "refs/hmdb.csv").resolve()
    assert (
        config.annotation.metaboscape_reference
        == (config_path.parent / "refs/metaboscape.csv").resolve()
    )
    assert config.trajectory.path_file == (config_path.parent / "paths/trajectory.csv").resolve()


def test_write_resolved_config_round_trips_new_paths_without_mutating_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    config = JointConfig.model_validate(
        {
            "project": {"name": "sample", "output_dir": "results"},
            "input": {"h5ad": "input.h5ad", "trusted_pickle": True},
            "annotation": {"hmdb_reference": "refs/hmdb.csv"},
            "trajectory": {"path_file": "paths/trajectory.csv"},
        }
    )

    written = write_resolved_config(config, tmp_path / "resolved.yaml")
    reloaded = load_config(written)

    assert reloaded.input.trusted_pickle is True
    assert reloaded.annotation.hmdb_reference == (tmp_path / "refs/hmdb.csv").resolve()
    assert reloaded.trajectory.path_file == (tmp_path / "paths/trajectory.csv").resolve()
    assert config.annotation.hmdb_reference == Path("refs/hmdb.csv")
    assert config.trajectory.path_file == Path("paths/trajectory.csv")


@pytest.mark.parametrize(
    ("config_name", "project_name", "input_key", "input_name", "method"),
    [
        ("d2.yaml", "d2", "h5ad", "d2.h5ad", "legacy_proportional"),
        ("d2-raw.yaml", "d2-raw", "imzml", "02.imzML", "legacy_proportional"),
        ("d8.yaml", "d8", "h5ad", "d8.h5ad", "specificity_filtered"),
        ("d8-raw.yaml", "d8-raw", "imzml", "08.imzML", "specificity_filtered"),
    ],
)
def test_reference_configs_are_portable_and_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_name: str,
    project_name: str,
    input_key: str,
    input_name: str,
    method: str,
):
    monkeypatch.setenv("JOINT_DATA_ROOT", str(tmp_path / "reference"))
    config_path = CONFIG_DIR / config_name
    payload = yaml.safe_load(config_path.read_text())

    config = load_config(config_path)

    assert payload["project"]["output_dir"] == f"../results/{project_name}"
    assert payload["input"][input_key].startswith("${JOINT_DATA_ROOT}/")
    assert not any("/Users/" in str(value) for value in payload["input"].values())
    assert config.project.name == project_name
    assert config.project.output_dir == (CONFIG_DIR.parent / "results" / project_name).resolve()
    assert getattr(config.input, input_key).name == input_name
    assert config.input.trusted_pickle is True
    assert config.laser_segmentation.parameters["top_hat_mode"] == "legacy_opening"
    assert config.laser_segmentation.parameters["top_hat_radius"] == 1
    assert config.laser_segmentation.parameters["peak_detection_mode"] == "legacy_global"
    assert config.quantification.method == method


def test_reference_configs_preserve_library_defaults_and_exact_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("JOINT_DATA_ROOT", str(tmp_path / "reference"))

    d2 = load_config(CONFIG_DIR / "d2.yaml")
    d8 = load_config(CONFIG_DIR / "d8.yaml")
    default = JointConfig(
        project=ProjectConfig(name="default", output_dir=Path("results")),
        input=InputConfig(h5ad=Path("input.h5ad")),
    )

    assert d2.laser_segmentation.parameters["post_closing_size"] == 1
    assert d2.laser_segmentation.parameters["marker_connectivity"] == 2
    assert d8.laser_segmentation.parameters["post_closing_size"] == 10
    assert d8.laser_segmentation.parameters["marker_connectivity"] == 1
    assert d8.quantification.unique_min_overlap == 0.0
    assert default.input.trusted_pickle is False
    assert default.laser_segmentation.parameters == {}
    assert default.quantification.unique_min_overlap == 0.10


@pytest.mark.parametrize("dataset", ["d2", "d8"])
def test_raw_reference_configs_only_change_project_input_and_preprocessing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset: str
):
    monkeypatch.setenv("JOINT_DATA_ROOT", str(tmp_path / "reference"))

    exact = load_config(CONFIG_DIR / f"{dataset}.yaml")
    raw = load_config(CONFIG_DIR / f"{dataset}-raw.yaml")

    for section in ("laser_segmentation", "registration", "quantification", "analysis"):
        assert getattr(raw, section) == getattr(exact, section)


def test_d2_merge_rules_match_notebook_corrections():
    expected = [
        (7, "276|284"),
        (9, "370|363"),
        (11, "448|455"),
        (13, "539|540"),
        (13, "536|532"),
        (15, "618|616"),
        (15, "607|603"),
        (21, "871|870"),
        (23, "951|952"),
        (23, "964|959"),
        (25, "1024|1027"),
        (25, "1045|1047"),
        (28, "1181|1176"),
        (28, "1159|1149"),
        (29, "1216|1220"),
        (29, "1210|1208"),
        (31, "1304|1302"),
        (32, "1339|1336"),
        (32, "1334|1335"),
        (33, "1383|1371"),
        (33, "1380|1381"),
        (34, "1427|1425"),
        (34, "1422|1423"),
        (35, "1477|1478"),
        (35, "1464|1465"),
        (36, "1527|1520"),
        (36, "1522|1516"),
        (36, "1513|1514"),
        (38, "1587|1582"),
        (40, "1692|1693"),
    ]

    payload = yaml.safe_load((CONFIG_DIR / "d2.yaml").read_text())
    rules = (CONFIG_DIR / payload["laser_segmentation"]["merge_rules"]).read_text().splitlines()

    actual = [(int(row.split(",", 2)[0]), row.split(",", 2)[1]) for row in rules[1:]]
    assert actual == expected
