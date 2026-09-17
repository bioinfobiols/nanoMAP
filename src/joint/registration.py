from numbers import Integral, Real
from typing import Literal

import anndata
import numpy as np
import pandas as pd

from joint.errors import RegistrationError
from joint.models import RegistrationResult

Orientation = Literal[
    "auto",
    "identity",
    "flip_x",
    "flip_y",
    "flip_xy",
    "swap_xy",
    "swap_flip_x",
    "swap_flip_y",
    "swap_flip_xy",
]
EdgePolicy = Literal["error", "truncate_left", "truncate_right"]

ORIENTATIONS = (
    "identity",
    "flip_x",
    "flip_y",
    "flip_xy",
    "swap_xy",
    "swap_flip_x",
    "swap_flip_y",
    "swap_flip_xy",
)
EDGE_POLICIES = ("error", "truncate_left", "truncate_right")

_REQUIRED_REGION_COLUMNS = (
    "seg_label",
    "area",
    "centroid-0",
    "centroid-1",
    "row_number",
    "column_number",
    "morphology",
)


def _validate_orientation(orientation: object) -> str:
    if not isinstance(orientation, str) or orientation not in ("auto", *ORIENTATIONS):
        raise RegistrationError(
            f"orientation must be one of {('auto', *ORIENTATIONS)}, got {orientation!r}"
        )
    return orientation


def _validate_edge_policy(edge_policy: object) -> str:
    if not isinstance(edge_policy, str) or edge_policy not in EDGE_POLICIES:
        raise RegistrationError(
            f"edge_policy must be one of {EDGE_POLICIES}, got {edge_policy!r}"
        )
    return edge_policy


def _validate_spatial_adata(msi_adata: object) -> tuple[anndata.AnnData, np.ndarray]:
    if not isinstance(msi_adata, anndata.AnnData):
        raise RegistrationError("msi_adata must be an AnnData object")
    if "spatial" not in msi_adata.obsm:
        raise RegistrationError('MSI AnnData is missing obsm["spatial"]')
    raw_spatial = np.asarray(msi_adata.obsm["spatial"])
    if raw_spatial.dtype.kind not in "iuf":
        raise RegistrationError('MSI obsm["spatial"] must contain numeric data')
    spatial = raw_spatial.astype(float, copy=True)
    if spatial.ndim != 2 or spatial.shape != (msi_adata.n_obs, 2):
        raise RegistrationError(
            'MSI obsm["spatial"] must be a two-dimensional n_obs by 2 array'
        )
    if spatial.shape[0] == 0:
        raise RegistrationError('MSI obsm["spatial"] must contain at least one point')
    if not np.isfinite(spatial).all():
        raise RegistrationError('MSI obsm["spatial"] must contain finite values')
    if not msi_adata.obs_names.is_unique:
        raise RegistrationError("MSI AnnData must have unique observation identifiers")
    if np.unique(spatial, axis=0).shape[0] != spatial.shape[0]:
        raise RegistrationError(
            'MSI obsm["spatial"] must contain unique coordinate pairs for unambiguous rows'
        )
    if "dataset" in msi_adata.obs:
        datasets = msi_adata.obs["dataset"]
        if datasets.isna().any() or datasets.nunique(dropna=False) != 1:
            raise RegistrationError("Registration requires AnnData from a single dataset")
    return msi_adata, spatial.copy()


def _numeric_region_column(
    regions: pd.DataFrame,
    column: str,
    *,
    integral: bool = False,
    non_negative: bool = False,
) -> np.ndarray:
    scalars = regions[column].astype(object).tolist()
    if not all(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        for value in scalars
    ):
        raise RegistrationError(f"Laser column {column!r} must be numeric")
    try:
        values = pd.to_numeric(regions[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise RegistrationError(f"Laser column {column!r} must be numeric") from exc
    if not np.isfinite(values).all():
        raise RegistrationError(f"Laser column {column!r} must contain finite values")
    if non_negative and (values < 0).any():
        raise RegistrationError(f"Laser column {column!r} must be non-negative")
    if integral and (
        (values < 1).any() or not np.equal(values, np.floor(values)).all()
    ):
        raise RegistrationError(f"Laser column {column!r} must contain positive integers")
    return values


def _normalize_text_region_column(regions: pd.DataFrame, column: str) -> pd.Series:
    values = regions[column]
    if values.isna().any():
        raise RegistrationError(f"Laser column {column!r} must not contain missing values")
    scalars = values.astype(object).tolist()
    if not all(isinstance(value, str) for value in scalars):
        raise RegistrationError(f"Laser column {column!r} must contain strings")
    normalized = [value.strip() for value in scalars]
    if any(not value for value in normalized):
        raise RegistrationError(f"Laser column {column!r} must contain non-empty strings")
    return pd.Series(normalized, index=regions.index, dtype=object)


def _normalize_seg_labels(regions: pd.DataFrame) -> pd.Series:
    values = regions["seg_label"]
    if values.isna().any():
        raise RegistrationError("Laser column 'seg_label' must not contain missing values")
    normalized: list[str] = []
    for value in values.astype(object).tolist():
        if isinstance(value, str):
            identifier = value.strip()
            if not identifier:
                raise RegistrationError(
                    "Laser column 'seg_label' must contain non-empty identifiers"
                )
            normalized.append(identifier)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise RegistrationError(
                "Laser column 'seg_label' must contain string or integral numeric identifiers"
            )
        if isinstance(value, Integral):
            normalized.append(str(int(value)))
            continue
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise RegistrationError(
                "Laser column 'seg_label' must contain finite integral identifiers"
            )
        normalized.append(str(int(numeric)))
    return pd.Series(normalized, index=regions.index, dtype=object)


def _validate_laser_regions(laser_regions: object) -> pd.DataFrame:
    if not isinstance(laser_regions, pd.DataFrame):
        raise RegistrationError("laser_regions must be a pandas DataFrame")
    missing = set(_REQUIRED_REGION_COLUMNS) - set(laser_regions.columns)
    if missing:
        raise RegistrationError(f"Laser regions are missing columns: {sorted(missing)}")

    regions = laser_regions.copy(deep=True)
    regions["seg_label"] = _normalize_seg_labels(regions)
    regions["morphology"] = _normalize_text_region_column(regions, "morphology")
    regions["area"] = _numeric_region_column(regions, "area", non_negative=True)
    for column in ("centroid-0", "centroid-1"):
        regions[column] = _numeric_region_column(regions, column)
    for column in ("row_number", "column_number"):
        regions[column] = _numeric_region_column(regions, column, integral=True).astype(int)
    if "label" in regions:
        regions["label"] = _numeric_region_column(regions, "label", integral=True).astype(int)

    active = regions.loc[regions["morphology"] != "spoilt"].copy()
    if active.empty:
        raise RegistrationError("Laser regions contain no active non-spoilt rows")
    duplicated = active.duplicated(["row_number", "column_number"], keep=False)
    if duplicated.any():
        positions = (
            active.loc[duplicated, ["row_number", "column_number"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise RegistrationError(
            "Laser regions must have unique active (row_number, column_number) positions; "
            f"duplicates: {positions}"
        )
    return active.reset_index(drop=True)


def _orient(spatial: np.ndarray, orientation: str) -> np.ndarray:
    result = spatial.copy()
    if orientation.startswith("swap"):
        result = result[:, [1, 0]]
    if orientation in {"flip_x", "flip_xy", "swap_flip_x", "swap_flip_xy"}:
        result[:, 0] = result[:, 0].min() + result[:, 0].max() - result[:, 0]
    if orientation in {"flip_y", "flip_xy", "swap_flip_y", "swap_flip_xy"}:
        result[:, 1] = result[:, 1].min() + result[:, 1].max() - result[:, 1]
    return result


def _mapping_for_orientation(
    obs_names: pd.Index,
    spatial: np.ndarray,
    regions: pd.DataFrame,
    *,
    orientation: str,
    edge_policy: str,
) -> tuple[pd.DataFrame, dict[int, int]]:
    oriented = _orient(spatial, orientation)
    pixels = pd.DataFrame(
        {
            "pixel_id": obs_names.to_numpy(copy=True),
            "x": oriented[:, 0],
            "y": oriented[:, 1],
        }
    )
    pixel_rows = [
        part.sort_values("x", kind="stable")
        for _, part in pixels.groupby("y", sort=True)
    ]
    laser_rows = [
        (int(row_number), part.sort_values("column_number", kind="stable"))
        for row_number, part in regions.groupby("row_number", sort=True)
    ]
    if len(pixel_rows) != len(laser_rows):
        raise RegistrationError(
            f"Orientation {orientation!r} gives {len(pixel_rows)} MSI rows and "
            f"{len(laser_rows)} laser rows"
        )

    records: list[pd.DataFrame] = []
    truncated_by_row: dict[int, int] = {}
    for pixel_row, (row_number, laser_row) in zip(pixel_rows, laser_rows, strict=True):
        difference = len(pixel_row) - len(laser_row)
        if difference < 0:
            raise RegistrationError(
                f"Laser row {row_number} contains {-difference} more points than its MSI row"
            )
        if difference and edge_policy == "error":
            raise RegistrationError(
                f"MSI row paired with laser row {row_number} has {difference} unmatched pixels; "
                "choose an explicit edge_policy"
            )
        if difference:
            truncated_by_row[row_number] = difference
            if edge_policy == "truncate_left":
                pixel_row = pixel_row.iloc[difference:]
            else:
                pixel_row = pixel_row.iloc[: len(laser_row)]
        paired = laser_row.copy().reset_index(drop=True)
        paired.insert(0, "pixel_id", pixel_row["pixel_id"].to_numpy(copy=True))
        records.append(paired)

    mapping = pd.concat(records, ignore_index=True)
    if len(mapping) != len(regions) or not mapping["pixel_id"].is_unique:
        raise RegistrationError("Registration mapping is not one-to-one")
    return mapping, truncated_by_row


def register_laser_points(
    msi_adata: anndata.AnnData,
    laser_regions: pd.DataFrame,
    *,
    orientation: Orientation = "auto",
    edge_policy: EdgePolicy = "error",
) -> RegistrationResult:
    """Pair a single MSI grid with active laser regions and return registered copies."""
    checked_orientation = _validate_orientation(orientation)
    checked_edge_policy = _validate_edge_policy(edge_policy)
    adata, spatial = _validate_spatial_adata(msi_adata)
    regions = _validate_laser_regions(laser_regions)

    if checked_orientation == "auto":
        candidates: list[tuple[str, pd.DataFrame, dict[int, int]]] = []
        for candidate in ORIENTATIONS:
            try:
                mapping, truncated = _mapping_for_orientation(
                    adata.obs_names,
                    spatial,
                    regions,
                    orientation=candidate,
                    edge_policy=checked_edge_policy,
                )
            except RegistrationError:
                continue
            candidates.append((candidate, mapping, truncated))
        if not candidates:
            raise RegistrationError("Automatic orientation found no valid candidate")
        if len(candidates) != 1:
            names = [item[0] for item in candidates]
            raise RegistrationError(
                f"Automatic orientation is ambiguous; valid candidates: {names}"
            )
        resolved_orientation, mapping, truncated_by_row = candidates[0]
    else:
        resolved_orientation = checked_orientation
        mapping, truncated_by_row = _mapping_for_orientation(
            adata.obs_names,
            spatial,
            regions,
            orientation=resolved_orientation,
            edge_policy=checked_edge_policy,
        )

    registered = adata[mapping["pixel_id"].to_numpy(copy=True)].copy()
    metadata = mapping.drop(columns=["pixel_id", "label"], errors="ignore")
    overlap = sorted(set(metadata.columns) & set(registered.obs.columns))
    if overlap:
        raise RegistrationError(
            f"Laser metadata columns overlap AnnData observations: {overlap}"
        )
    if registered.n_obs != len(metadata):
        raise RegistrationError("Registration metadata would drop or duplicate observations")
    for column in metadata.columns:
        registered.obs[column] = metadata[column].to_numpy(copy=True)
    registered.obsm["spatial"] = mapping[["centroid-1", "centroid-0"]].to_numpy(
        dtype=float, copy=True
    )
    transform = {
        "orientation": resolved_orientation,
        "edge_policy": checked_edge_policy,
    }
    joint_metadata = registered.uns.setdefault("joint", {})
    if not isinstance(joint_metadata, dict):
        raise RegistrationError("AnnData uns['joint'] must be a mapping")
    joint_metadata["registration"] = transform.copy()
    truncated_pixels = int(sum(truncated_by_row.values()))
    return RegistrationResult(
        adata=registered,
        mapping=mapping,
        transform=transform,
        report={
            "registered_pixels": registered.n_obs,
            "truncated_pixels": truncated_pixels,
            "truncated_pixels_by_row": truncated_by_row,
        },
    )


def _validate_image(name: str, image: object) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise RegistrationError(f"{name} image must be a numpy array")
    if image.ndim not in {2, 3} or any(dimension == 0 for dimension in image.shape):
        raise RegistrationError(f"{name} image must be a non-empty two- or three-dimensional array")
    if image.dtype.kind not in "buif":
        raise RegistrationError(f"{name} image must contain real numeric data")
    if not np.isfinite(image).all():
        raise RegistrationError(f"{name} image must contain finite values")
    return np.array(image, copy=True)


def mount_spatial_images(
    adata: anndata.AnnData,
    *,
    laser_image: np.ndarray | None = None,
    cell_image: np.ndarray | None = None,
    library_id: str = "sample",
    spot_diameter: float = 1.0,
) -> anndata.AnnData:
    """Mount copied spatial images in Scanpy-compatible metadata on an AnnData copy."""
    if not isinstance(adata, anndata.AnnData):
        raise RegistrationError("adata must be an AnnData object")
    if not isinstance(library_id, str) or not library_id.strip():
        raise RegistrationError("library_id must be a non-empty string")
    if isinstance(spot_diameter, (bool, np.bool_)) or not isinstance(spot_diameter, Real):
        raise RegistrationError("spot_diameter must be a finite positive number")
    checked_diameter = float(spot_diameter)
    if not np.isfinite(checked_diameter) or checked_diameter <= 0:
        raise RegistrationError("spot_diameter must be a finite positive number")

    images: dict[str, np.ndarray] = {}
    if laser_image is not None:
        images["laser"] = _validate_image("laser", laser_image)
    if cell_image is not None:
        images["hires"] = _validate_image("cell", cell_image)
    if len({image.shape[:2] for image in images.values()}) > 1:
        raise RegistrationError("Mounted images must have identical height and width")

    result = adata.copy()
    if not images:
        return result
    spatial_metadata = result.uns.setdefault("spatial", {})
    if not isinstance(spatial_metadata, dict):
        raise RegistrationError("AnnData uns['spatial'] must be a mapping")
    spatial_metadata[library_id] = {
        "images": images,
        "scalefactors": {
            "tissue_hires_scalef": 1.0,
            "spot_diameter_fullres": checked_diameter,
        },
    }
    return result
