import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationError,
)

from joint.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    output_dir: Path
    random_seed: int = 0


class InputConfig(StrictModel):
    imzml: Path | None = None
    h5ad: Path | None = None
    laser_image: Path | None = None
    cell_segmentation: Path | None = None
    trusted_pickle: StrictBool = False


class MatrixRemovalConfig(StrictModel):
    enabled: bool = False
    method: Literal["reference", "blank", "denovo"] = "denovo"
    reference_file: Path | None = None
    blank_datasets: list[str] = Field(default_factory=list)
    matrix_frequency: float = Field(default=0.8, ge=0.0, le=1.0)
    tolerance: float = Field(default=10.0, gt=0.0)
    unit: Literal["ppm", "da"] = "ppm"


class PreprocessingConfig(StrictModel):
    backend: Literal["python", "cardinal"] = "python"
    normalization: Literal["rms", "tic", "none"] = "rms"
    normalized_layer: str = "normalized"
    min_occurrence: float = Field(default=0.05, ge=0.0, le=1.0)
    peak_tolerance: float = Field(default=10.0, gt=0.0)
    tolerance_unit: Literal["ppm", "da"] = "ppm"
    profile_bin_size: float | None = Field(default=None, gt=0.0)
    cardinal_snr: float = Field(default=3.0, gt=0.0)
    matrix_removal: MatrixRemovalConfig = Field(default_factory=MatrixRemovalConfig)


class LaserSegmentationConfig(StrictModel):
    method: Literal["real", "simulated"] = "real"
    radius: int = Field(default=18, gt=0)
    expected_rows: int | None = Field(default=None, gt=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    merge_rules: Path | None = None


class RegistrationConfig(StrictModel):
    orientation: Literal[
        "auto",
        "identity",
        "flip_x",
        "flip_y",
        "flip_xy",
        "swap_xy",
        "swap_flip_x",
        "swap_flip_y",
        "swap_flip_xy",
    ] = "auto"
    edge_policy: Literal["error", "truncate_left", "truncate_right"] = "error"


class QuantificationConfig(StrictModel):
    method: Literal["specificity_filtered", "legacy_proportional"] = "specificity_filtered"
    boundary_margin: float = Field(default=20.0, ge=0.0)
    unique_min_overlap: float = Field(default=0.10, ge=0.0, le=1.0)
    dominant_min_overlap: float = Field(default=0.60, ge=0.0, le=1.0)
    secondary_max_overlap: float = Field(default=0.15, ge=0.0, le=1.0)


class QcConfig(StrictModel):
    enabled: StrictBool = True
    outlier_contamination: StrictFloat = Field(default=0.05, gt=0.0, lt=0.5)


class AnnotationConfig(StrictModel):
    enabled: StrictBool = False
    hmdb_reference: Path | None = None
    metaboscape_reference: Path | None = None
    ion_mode: Literal["pos", "neg"] = "pos"
    hmdb_ppm: StrictFloat = Field(default=5.0, gt=0.0)
    metaboscape_ppm: StrictFloat = Field(default=3.0, gt=0.0)


class ClusteringConfig(StrictModel):
    enabled: StrictBool = True
    n_neighbors: StrictInt = Field(default=15, gt=1)
    resolution: StrictFloat = Field(default=0.6, gt=0.0)
    n_top_features: StrictInt | None = Field(default=None, gt=0)


class TrajectoryConfig(StrictModel):
    enabled: StrictBool = False
    path_file: Path | None = None
    features: list[str] = Field(default_factory=list)
    points: StrictInt = Field(default=100, gt=1)


class CosgConfig(StrictModel):
    enabled: bool = False
    groupby: str = "cluster"
    n_genes: int = Field(default=20, gt=0)


class CnmfConfig(StrictModel):
    enabled: bool = False
    components: list[int] = Field(default_factory=lambda: list(range(5, 20)))
    selected_k: int = 14
    seed: int = 14
    num_highvar_genes: int = Field(default=300, gt=0)


class AnalysisConfig(StrictModel):
    cosg: CosgConfig = Field(default_factory=CosgConfig)
    cnmf: CnmfConfig = Field(default_factory=CnmfConfig)


class JointConfig(StrictModel):
    project: ProjectConfig
    input: InputConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    laser_segmentation: LaserSegmentationConfig = Field(default_factory=LaserSegmentationConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    quantification: QuantificationConfig = Field(default_factory=QuantificationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    qc: QcConfig = Field(default_factory=QcConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)


_PATH_KEYS = {
    "output_dir",
    "imzml",
    "h5ad",
    "laser_image",
    "cell_segmentation",
    "merge_rules",
    "reference_file",
    "hmdb_reference",
    "metaboscape_reference",
    "path_file",
}

_PATH_FIELD_PATHS = {
    ("project", "output_dir"),
    ("input", "imzml"),
    ("input", "h5ad"),
    ("input", "laser_image"),
    ("input", "cell_segmentation"),
    ("preprocessing", "matrix_removal", "reference_file"),
    ("laser_segmentation", "merge_rules"),
    ("annotation", "hmdb_reference"),
    ("annotation", "metaboscape_reference"),
    ("trajectory", "path_file"),
}

_ENVIRONMENT_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _resolve_paths(value: Any, base: Path, field_path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            name: _resolve_paths(item, base, (*field_path, name)) for name, item in value.items()
        }
    if field_path in _PATH_FIELD_PATHS and value is not None:
        key = field_path[-1]
        if not isinstance(value, (str, Path)):
            raise ConfigurationError(f"Invalid path value for {key}")
        try:
            expanded = os.path.expandvars(str(value))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid path value for {key}") from exc
        unresolved = _ENVIRONMENT_VARIABLE.search(expanded)
        if unresolved is not None:
            variable = unresolved.group(1) or unresolved.group(2)
            raise ConfigurationError(f"Environment variable is not defined: {variable}")
        if "$" in expanded:
            raise ConfigurationError(f"Malformed environment variable reference in path: {key}")
        try:
            path = Path(expanded).expanduser()
            return path.resolve() if path.is_absolute() else (base / path).resolve()
        except (TypeError, ValueError, OSError) as exc:
            raise ConfigurationError(f"Invalid path value for {key}") from exc
    return value


def _validation_message(error: ValidationError) -> str:
    messages = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"])
        input_value = detail.get("input")
        if isinstance(input_value, (str, int, float, bool)):
            messages.append(f"{location}: {detail['msg']} (received {input_value!r})")
        else:
            messages.append(f"{location}: {detail['msg']}")
    return "; ".join(messages)


def load_config(path: str | Path) -> JointConfig:
    try:
        config_path = Path(path).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise ConfigurationError("Invalid configuration path") from exc
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text()) or {}
        resolved = _resolve_paths(payload, config_path.parent)
        config = JointConfig.model_validate(resolved)
    except ValidationError as exc:
        raise ConfigurationError(_validation_message(exc)) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError("Malformed YAML configuration") from exc
    except OSError as exc:
        raise ConfigurationError("Unable to read configuration") from exc
    except TypeError as exc:
        raise ConfigurationError("Invalid configuration data") from exc
    if (config.input.imzml is None) == (config.input.h5ad is None):
        raise ConfigurationError("Exactly one of input.imzml or input.h5ad must be provided")
    return config


def write_resolved_config(config: JointConfig, path: str | Path) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved = JointConfig.model_validate(
        _resolve_paths(config.model_dump(mode="json"), Path.cwd())
    )
    destination.write_text(yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False))
    return destination
