"""Copy-safe clustering and differential metabolite ranking."""

import csv
import os
import tempfile
import unicodedata
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis

from joint.errors import InputFormatError, OptionalDependencyError

_RANKING_COLUMNS = ["group", "names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
_RANKING_METHODS = frozenset({"t-test", "t-test_overestim_var", "wilcoxon"})
_COSG_RESERVED_KWARGS = frozenset({"adata", "groupby", "key_added", "n_genes_user"})
_CNMF_DEPENDENCY_MESSAGE = "cNMF support requires `python -m pip install 'joint-msi[cnmf]'`"


def _require_anndata(adata: object) -> anndata.AnnData:
    if not isinstance(adata, anndata.AnnData):
        raise InputFormatError("adata must be an AnnData object")
    if adata.n_obs == 0:
        raise InputFormatError("analysis requires at least 1 observation")
    if adata.n_vars == 0:
        raise InputFormatError("analysis requires at least 1 feature")
    if not adata.obs_names.is_unique:
        raise InputFormatError("AnnData observation identifiers must be unique")
    if not adata.var_names.is_unique:
        raise InputFormatError("AnnData feature identifiers must be unique")
    return adata


def _validate_matrix(matrix: object, *, name: str, shape: tuple[int, int]) -> Any:
    if not hasattr(matrix, "shape") or matrix.shape != shape:
        raise InputFormatError(f"{name} must have shape (n_obs, n_vars)")
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if values.dtype.kind not in "uif":
        raise InputFormatError(f"{name} must contain real numeric values")
    if not np.isfinite(values).all():
        raise InputFormatError(f"{name} must contain finite values")
    return matrix


def _variable_feature_count(matrix: Any) -> int:
    if sparse.issparse(matrix):
        minimum = np.asarray(matrix.min(axis=0).toarray()).ravel()
        maximum = np.asarray(matrix.max(axis=0).toarray()).ravel()
    else:
        values = np.asarray(matrix)
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
    return int(np.count_nonzero(minimum != maximum))


def _feature_variances(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        _, variances = mean_variance_axis(matrix, axis=0)
    else:
        variances = np.var(np.asarray(matrix, dtype=np.float64), axis=0)
    checked = np.asarray(variances, dtype=np.float64)
    if not np.isfinite(checked).all():
        raise InputFormatError("feature variances must be finite")
    return checked


def _top_variance_mask(matrix: Any, n_top_features: int) -> np.ndarray:
    variances = _feature_variances(matrix)
    order = np.argsort(-variances, kind="stable")
    mask = np.zeros(variances.size, dtype=bool)
    mask[order[:n_top_features]] = True
    return mask


def _selected_matrix(adata: anndata.AnnData, layer: object) -> tuple[str | None, Any]:
    if layer is None:
        return None, _validate_matrix(adata.X, name="adata.X", shape=adata.shape)
    if not isinstance(layer, str) or not layer.strip():
        raise InputFormatError("layer must be a non-empty string or None")
    if layer not in adata.layers:
        raise InputFormatError(f"adata.layers is missing requested layer {layer!r}")
    return layer, _validate_matrix(
        adata.layers[layer],
        name=f'adata.layers["{layer}"]',
        shape=adata.shape,
    )


def _bounded_positive_integer(value: object, *, name: str, minimum: int = 1, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InputFormatError(f"{name} must be an integer in [{minimum}, {maximum}]")
    checked = int(value)
    if checked < minimum or checked > maximum:
        raise InputFormatError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return checked


def _positive_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InputFormatError(f"{name} must be a positive finite number")
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0:
        raise InputFormatError(f"{name} must be a positive finite number")
    return checked


def _random_seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InputFormatError("random_seed must be an integer in [0, 2**32 - 1]")
    checked = int(value)
    if checked < 0 or checked > 2**32 - 1:
        raise InputFormatError("random_seed must be an integer in [0, 2**32 - 1]")
    return checked


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputFormatError(f"{name} must be a non-empty string")
    return value


def _single_feature_pca(
    result: anndata.AnnData,
    matrix: Any,
    mask: np.ndarray | None,
) -> None:
    feature_index = int(np.flatnonzero(mask)[0]) if mask is not None else 0
    column = matrix[:, feature_index]
    values = column.toarray().ravel() if sparse.issparse(column) else np.asarray(column).ravel()
    coordinates = np.asarray(values, dtype=np.float64) - float(np.mean(values, dtype=np.float64))
    result.obsm["X_pca"] = coordinates[:, None]
    loadings = np.zeros((result.n_vars, 1), dtype=np.float64)
    loadings[feature_index, 0] = 1.0
    result.varm["PCs"] = loadings
    variance = float(np.var(values, ddof=1))
    result.uns["pca"] = {
        "params": {
            "zero_center": True,
            "use_highly_variable": mask is not None,
            "mask_var": mask.copy() if mask is not None else None,
        },
        "variance": np.array([variance]),
        "variance_ratio": np.array([1.0]),
    }


def cluster_cells(
    adata: anndata.AnnData,
    *,
    layer: str | None = None,
    n_top_features: int | None = None,
    n_neighbors: int = 15,
    resolution: float = 0.6,
    random_seed: int = 0,
    key_added: str = "cluster",
) -> anndata.AnnData:
    """Cluster observations on a copied matrix with deterministic Scanpy settings."""
    checked = _require_anndata(adata)
    checked_layer, selected_matrix = _selected_matrix(checked, layer)
    if checked.n_obs < 3:
        raise InputFormatError("clustering requires at least 3 observations")
    if n_top_features is None:
        checked_top_features = None
    else:
        checked_top_features = _bounded_positive_integer(
            n_top_features,
            name="n_top_features",
            maximum=checked.n_vars,
        )
    variable_features = _variable_feature_count(selected_matrix)
    if variable_features == 0:
        raise InputFormatError("clustering requires at least 1 variable feature")
    if (
        checked_top_features is not None
        and checked_top_features < checked.n_vars
        and checked_top_features > variable_features
    ):
        raise InputFormatError("n_top_features must not exceed the number of variable features")
    checked_neighbors = _bounded_positive_integer(
        n_neighbors,
        name="n_neighbors",
        minimum=2,
        maximum=checked.n_obs - 1,
    )
    checked_resolution = _positive_finite_real(resolution, name="resolution")
    checked_seed = _random_seed(random_seed)
    checked_key = _nonempty_string(key_added, name="key_added")

    result = checked.copy()
    mask_var: np.ndarray | None = None
    if checked_top_features is not None and checked_top_features < result.n_vars:
        mask_var = _top_variance_mask(selected_matrix, checked_top_features)
        result.var["highly_variable"] = mask_var

    selected_features = checked_top_features if mask_var is not None else result.n_vars
    if selected_features == 1:
        n_components = 1
        pca_solver = "manual_single_feature"
        _single_feature_pca(result, selected_matrix, mask_var)
    else:
        if sparse.issparse(selected_matrix):
            n_components = min(50, result.n_obs - 1, selected_features - 1)
            pca_solver = "arpack"
        else:
            n_components = min(50, result.n_obs - 1, selected_features)
            pca_solver = "full"
        sc.tl.pca(
            result,
            n_comps=n_components,
            layer=checked_layer,
            mask_var=mask_var,
            svd_solver=pca_solver,
            random_state=checked_seed,
        )
    sc.pp.neighbors(
        result,
        n_neighbors=checked_neighbors,
        n_pcs=n_components,
        use_rep="X_pca",
        random_state=checked_seed,
    )
    sc.tl.leiden(
        result,
        resolution=checked_resolution,
        random_state=checked_seed,
        key_added=checked_key,
        directed=False,
        flavor="igraph",
        n_iterations=2,
    )
    joint_settings = result.uns.setdefault("joint", {})
    if not isinstance(joint_settings, dict):
        raise InputFormatError("AnnData uns['joint'] must be a mapping")
    joint_settings["clustering"] = {
        "layer": checked_layer,
        "n_top_features": checked_top_features,
        "n_neighbors": checked_neighbors,
        "resolution": checked_resolution,
        "random_seed": checked_seed,
        "key_added": checked_key,
        "n_pcs": n_components,
        "pca_solver": pca_solver,
        "leiden_flavor": "igraph",
        "leiden_iterations": 2,
    }
    return result


def _group_order(series: pd.Series) -> list[object]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.tolist())
        return [category for category in series.cat.categories if category in present]
    return pd.unique(series).tolist()


def _validated_groups(adata: anndata.AnnData, groupby: object) -> tuple[str, list[str], np.ndarray]:
    checked_groupby = _nonempty_string(groupby, name="groupby")
    if checked_groupby not in adata.obs:
        raise InputFormatError(
            f"groupby must name an existing adata.obs column: {checked_groupby!r}"
        )
    groups = adata.obs[checked_groupby]
    present = ~groups.isna()
    present_groups = groups.loc[present]
    ordered_values = _group_order(present_groups)
    if len(ordered_values) < 2:
        raise InputFormatError("ranking requires at least 2 groups")
    counts = present_groups.value_counts(sort=False, dropna=False)
    if any(int(counts.loc[value]) < 2 for value in ordered_values):
        raise InputFormatError("each group must contain at least 2 observations")
    ordered_strings = [str(value) for value in ordered_values]
    if len(set(ordered_strings)) != len(ordered_strings):
        raise InputFormatError("group labels must be unique when represented as strings")
    return checked_groupby, ordered_strings, present.to_numpy(dtype=bool)


def _validated_cosg_kwargs(kwargs: object) -> dict[str, Any]:
    if not isinstance(kwargs, Mapping):
        raise InputFormatError("COSG kwargs must be a mapping")
    if not all(isinstance(name, str) for name in kwargs):
        raise InputFormatError("COSG kwargs must use string names")
    reserved = sorted(_COSG_RESERVED_KWARGS.intersection(kwargs))
    if reserved:
        raise InputFormatError(f"COSG kwargs may not override reserved arguments: {reserved}")
    if "copy" in kwargs and (
        not isinstance(kwargs["copy"], (bool, np.bool_)) or bool(kwargs["copy"])
    ):
        raise InputFormatError("COSG kwargs may not override reserved arguments: ['copy']")
    return dict(kwargs)


def _cosg_group_selection(groups: object, *, group_order: list[str]) -> set[str]:
    if groups is None or (isinstance(groups, str) and groups == "all"):
        return set(group_order)
    if isinstance(groups, str) or not isinstance(groups, Iterable) or isinstance(groups, Mapping):
        values = [groups]
    else:
        values = list(groups)
    if not values or any(not isinstance(value, str) for value in values):
        raise InputFormatError("COSG groups must be non-empty string group labels or 'all'")
    if len(set(values)) != len(values) or not set(values).issubset(group_order):
        raise InputFormatError("COSG groups must select distinct existing group labels")
    return set(values)


def _cosg_active_feature_source(
    adata: anndata.AnnData,
    *,
    kwargs: Mapping[str, Any],
) -> tuple[int, set[str]]:
    use_raw = kwargs.get("use_raw", False)
    if not isinstance(use_raw, (bool, np.bool_)):
        raise InputFormatError("COSG use_raw must be a boolean when provided")
    layer = kwargs.get("layer")
    if layer is not None:
        if bool(use_raw):
            raise InputFormatError("COSG cannot combine layer with use_raw=True")
        _selected_matrix(adata, layer)
        return adata.n_vars, set(adata.var_names.map(str))

    if not bool(use_raw):
        _validate_matrix(adata.X, name="adata.X", shape=adata.shape)
        return adata.n_vars, set(adata.var_names.map(str))

    if adata.raw is None:
        raise InputFormatError("COSG use_raw=True requires adata.raw")
    if not adata.raw.var_names.is_unique:
        raise InputFormatError("AnnData raw feature identifiers must be unique")
    raw_shape = (adata.n_obs, adata.raw.n_vars)
    _validate_matrix(adata.raw.X, name="adata.raw.X", shape=raw_shape)
    return adata.raw.n_vars, set(adata.raw.var_names.map(str))


def _cosg_names_table(
    adata: anndata.AnnData,
    *,
    key_added: str,
    groups: set[str],
    n_genes: int,
    feature_ids: set[str],
) -> pd.DataFrame:
    try:
        result = adata.uns[key_added]
    except (AttributeError, KeyError, TypeError) as exc:
        raise InputFormatError(f"COSG result is missing adata.uns[{key_added!r}]") from exc
    if not isinstance(result, Mapping):
        raise InputFormatError(f"COSG result adata.uns[{key_added!r}] must be a mapping")
    try:
        names = result["names"]
    except (KeyError, TypeError) as exc:
        raise InputFormatError("COSG result is missing its structured 'names' table") from exc
    if not isinstance(names, np.ndarray) or names.dtype.names is None or names.ndim != 1:
        raise InputFormatError("COSG result 'names' must be a one-dimensional structured ndarray")
    if len(names) != n_genes:
        raise InputFormatError("COSG result 'names' does not contain the requested number of ranks")

    fields = list(names.dtype.names)
    if set(fields) != groups or len(fields) != len(groups):
        raise InputFormatError("COSG result 'names' contains unexpected groups")

    columns: dict[str, list[str]] = {}
    for group in fields:
        values = names[group]
        if values.ndim != 1:
            raise InputFormatError("COSG result 'names' fields must be one-dimensional")
        checked_values: list[str] = []
        for feature_id in values:
            if not isinstance(feature_id, str) or feature_id not in feature_ids:
                raise InputFormatError("COSG result 'names' must contain known string feature IDs")
            checked_values.append(feature_id)
        columns[group] = checked_values
    table = pd.DataFrame(columns)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def _cosg_working_copy(
    adata: anndata.AnnData,
    *,
    selected: np.ndarray,
    groupby: str,
    group_order: list[str],
    use_raw: bool,
) -> anndata.AnnData:
    if use_raw:
        assert adata.raw is not None
        working = anndata.AnnData(
            adata.raw.X[selected].copy(),
            obs=adata.obs.iloc[np.flatnonzero(selected)].copy(),
            var=adata.raw.var.copy(),
        )
        working.raw = working.copy()
    else:
        working = adata[selected].copy()
    working.obs[groupby] = pd.Categorical(
        working.obs[groupby].astype(object).map(str),
        categories=group_order,
        ordered=True,
    )
    return working


def _validate_ranking_table(table: pd.DataFrame, *, group_order: list[str]) -> pd.DataFrame:
    missing = [column for column in _RANKING_COLUMNS if column not in table]
    if missing:
        raise InputFormatError(f"Scanpy ranking result is missing columns: {missing}")
    tidy = table.loc[:, _RANKING_COLUMNS].copy()
    tidy["group"] = tidy["group"].map(str)
    tidy["names"] = tidy["names"].map(str)
    rank = {group: index for index, group in enumerate(group_order)}
    if not tidy["group"].isin(rank).all():
        raise InputFormatError("Scanpy ranking returned unexpected groups")
    tidy["_group_order"] = tidy["group"].map(rank)
    tidy["_row_order"] = np.arange(len(tidy))
    tidy = tidy.sort_values(["_group_order", "_row_order"], kind="stable")
    tidy = tidy.drop(columns=["_group_order", "_row_order"]).reset_index(drop=True)
    scores = pd.to_numeric(tidy["scores"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(scores).any():
        raise InputFormatError("ranking column 'scores' must contain finite or infinite values")
    tidy["scores"] = scores
    for column in ("pvals", "pvals_adj"):
        values = pd.to_numeric(tidy[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise InputFormatError(f"ranking column {column!r} must contain values in [0, 1]")
        tidy[column] = values
    return tidy


def rank_metabolites(
    adata: anndata.AnnData,
    *,
    groupby: str,
    method: str = "wilcoxon",
    layer: str | None = None,
) -> pd.DataFrame:
    """Return Scanpy differential rankings as a stable tidy table."""
    checked = _require_anndata(adata)
    checked_layer, _ = _selected_matrix(checked, layer)
    checked_groupby, group_order, present = _validated_groups(checked, groupby)
    checked_method = _nonempty_string(method, name="method")
    if checked_method not in _RANKING_METHODS:
        allowed = ", ".join(sorted(_RANKING_METHODS))
        raise InputFormatError(f"method must be one of: {allowed}")

    result = checked[present].copy()
    result.obs[checked_groupby] = pd.Categorical(
        result.obs[checked_groupby].map(str),
        categories=group_order,
        ordered=True,
    )
    rank_arguments = {
        "groupby": checked_groupby,
        "method": checked_method,
        "layer": checked_layer,
        "use_raw": False,
        "groups": group_order,
    }
    # Scanpy's score and p-value tests support raw and signed matrices, while
    # its optional log-fold-change summary can overflow or be undefined.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"(?:overflow encountered in expm1|"
                r"divide by zero encountered in (?:divide|log2)|"
                r"invalid value encountered in (?:divide|log2))"
            ),
            category=RuntimeWarning,
            module=r"scanpy\.tools\._rank_genes_groups",
        )
        sc.tl.rank_genes_groups(result, **rank_arguments)
    table = sc.get.rank_genes_groups_df(result, group=None)
    return _validate_ranking_table(table, group_order=group_order)


def run_cosg(
    adata: anndata.AnnData,
    *,
    groupby: str,
    n_genes: int = 20,
    key_added: str = "cosg",
    **kwargs: Any,
) -> pd.DataFrame:
    """Run optional COSG on a validated AnnData copy and return marker-name ranks."""
    checked = _require_anndata(adata)
    checked_groupby, group_order, present = _validated_groups(checked, groupby)
    checked_key = _nonempty_string(key_added, name="key_added")
    checked_kwargs = _validated_cosg_kwargs(kwargs)
    selected_groups = _cosg_group_selection(checked_kwargs.get("groups"), group_order=group_order)
    selected_group_order = [group for group in group_order if group in selected_groups]
    n_features, feature_ids = _cosg_active_feature_source(checked, kwargs=checked_kwargs)
    checked_n_genes = _bounded_positive_integer(
        n_genes,
        name="n_genes",
        maximum=n_features,
    )
    if checked_key in checked.uns:
        raise InputFormatError(f"adata.uns already contains output key {checked_key!r}")

    try:
        import cosg
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            "COSG support requires `python -m pip install 'joint-msi[cosg]'`"
        ) from exc
    try:
        cosg_function = cosg.cosg
    except AttributeError as exc:
        raise OptionalDependencyError(
            "COSG support requires `python -m pip install 'joint-msi[cosg]'`"
        ) from exc
    if not callable(cosg_function):
        exc = TypeError("COSG module does not provide a callable cosg function")
        raise OptionalDependencyError(
            "COSG support requires `python -m pip install 'joint-msi[cosg]'`"
        ) from exc
    selected = present & checked.obs[checked_groupby].map(str).isin(selected_groups).to_numpy()
    if not selected.any():
        raise InputFormatError("COSG groups must select at least 1 observation")
    working = _cosg_working_copy(
        checked,
        selected=selected,
        groupby=checked_groupby,
        group_order=selected_group_order,
        use_raw=bool(checked_kwargs.get("use_raw", False)),
    )
    cosg_function(
        working,
        key_added=checked_key,
        groupby=checked_groupby,
        n_genes_user=checked_n_genes,
        **checked_kwargs,
    )
    return _cosg_names_table(
        working,
        key_added=checked_key,
        groups=selected_groups,
        n_genes=checked_n_genes,
        feature_ids=feature_ids,
    )


@dataclass(frozen=True)
class PreparedCnmf:
    """Validated cNMF preparation and the immutable settings needed to run it."""

    backend: Any
    counts_path: Path
    output_dir: Path
    name: str
    task_dir: Path
    staging_dir: Path
    components: tuple[int, ...]
    seed: int
    num_highvar_genes: int


def _safe_path(value: object, *, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise InputFormatError(f"{name} must be a non-empty string or pathlib.Path")
    if isinstance(value, str) and not value.strip():
        raise InputFormatError(f"{name} must be a non-empty string or pathlib.Path")
    if "\x00" in str(value):
        raise InputFormatError(f"{name} must be a valid filesystem path")
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise InputFormatError(f"{name} must be a valid filesystem path") from exc


def _cnmf_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputFormatError("name must be a non-empty filesystem-safe name")
    if value != value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise InputFormatError("name must be one safe path component")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise InputFormatError("name must not contain control or format characters")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InputFormatError("name must be valid UTF-8") from exc
    if len(encoded) > 100:
        raise InputFormatError("name must be at most 100 UTF-8 bytes")
    return value


def _cnmf_components(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise InputFormatError("components must contain unique positive integers")
    values = tuple(value)
    if not values or any(
        isinstance(component, (bool, np.bool_))
        or not isinstance(component, Integral)
        or int(component) <= 0
        for component in values
    ):
        raise InputFormatError("components must contain unique positive integers")
    checked = tuple(int(component) for component in values)
    if len(set(checked)) != len(checked):
        raise InputFormatError("components must contain unique positive integers")
    return checked


def _cnmf_seed(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise InputFormatError("seed must be an integer in [0, 2**32 - 1]")
    checked = int(value)
    if checked < 0 or checked > 2**32 - 1:
        raise InputFormatError("seed must be an integer in [0, 2**32 - 1]")
    return checked


def _cnmf_class() -> Any:
    try:
        from cnmf import cNMF
    except ModuleNotFoundError as exc:
        if exc.name != "cnmf":
            raise
        raise OptionalDependencyError(_CNMF_DEPENDENCY_MESSAGE) from exc
    if not callable(cNMF):
        raise TypeError("cnmf module does not provide a callable cNMF class")
    return cNMF


def _write_cnmf_counts(adata: anndata.AnnData, destination: Path) -> Path:
    try:
        temporary_directory = Path(tempfile.mkdtemp(prefix=".joint-cnmf-", dir=destination.parent))
    except OSError as exc:
        raise InputFormatError(
            f"Could not create cNMF temporary directory in {destination.parent}: {exc}"
        ) from exc
    temporary = temporary_directory / "counts.h5ad"
    try:
        adata.write_h5ad(temporary)
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise InputFormatError(f"cNMF counts output already exists: {destination}") from exc
    except Exception as exc:
        raise InputFormatError(f"Could not write cNMF counts {destination}: {exc}") from exc
    return temporary_directory


def prepare_cnmf(
    adata: anndata.AnnData,
    output_dir: str | Path,
    *,
    name: str,
    components: list[int],
    seed: int,
    num_highvar_genes: int,
) -> PreparedCnmf:
    """Validate counts and prepare a lazily imported cNMF task without overwriting outputs."""
    checked = _require_anndata(adata)
    counts = _validate_matrix(checked.X, name="adata.X", shape=checked.shape)
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if np.any(values < 0):
        raise InputFormatError("cNMF counts in adata.X must be nonnegative")
    checked_name = _cnmf_name(name)
    destination = _safe_path(output_dir, name="output_dir")
    checked_components = _cnmf_components(components)
    checked_seed = _cnmf_seed(seed)
    checked_highvar = _bounded_positive_integer(
        num_highvar_genes,
        name="num_highvar_genes",
        maximum=checked.n_vars,
    )
    if destination.exists() and not destination.is_dir():
        raise InputFormatError(f"output_dir must be a directory: {destination}")
    counts_path = destination / f"{checked_name}.h5ad"
    task_output = destination / checked_name
    if counts_path.exists():
        raise InputFormatError(f"cNMF counts output already exists: {counts_path}")

    backend_class = _cnmf_class()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputFormatError(
            f"Could not create cNMF output directory {destination}: {exc}"
        ) from exc
    try:
        task_output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise InputFormatError(f"cNMF task output already exists: {task_output}") from exc
    except OSError as exc:
        raise InputFormatError(f"Could not reserve cNMF task output {task_output}: {exc}") from exc
    staging_dir = _write_cnmf_counts(checked, counts_path)
    backend = backend_class(output_dir=str(destination), name=checked_name)
    backend.prepare(
        counts_fn=str(counts_path),
        components=np.asarray(checked_components),
        seed=checked_seed,
        num_highvar_genes=checked_highvar,
    )
    return PreparedCnmf(
        backend=backend,
        counts_path=counts_path,
        output_dir=destination,
        name=checked_name,
        task_dir=task_output,
        staging_dir=staging_dir,
        components=checked_components,
        seed=checked_seed,
        num_highvar_genes=checked_highvar,
    )


def _validated_prepared_cnmf(prepared: object) -> PreparedCnmf:
    if not isinstance(prepared, PreparedCnmf):
        raise InputFormatError("prepared must be a PreparedCnmf object")
    return prepared


def _validated_cnmf_workers(worker_index: object, total_workers: object) -> tuple[int, int]:
    if (
        isinstance(total_workers, (bool, np.bool_))
        or not isinstance(total_workers, Integral)
        or int(total_workers) <= 0
    ):
        raise InputFormatError("total_workers must be a positive integer")
    checked_workers = int(total_workers)
    if (
        isinstance(worker_index, (bool, np.bool_))
        or not isinstance(worker_index, Integral)
        or int(worker_index) < 0
        or int(worker_index) >= checked_workers
    ):
        raise InputFormatError("worker_index must be a zero-based worker in total_workers")
    return int(worker_index), checked_workers


def _validated_cnmf_consensus(
    prepared: PreparedCnmf,
    selected_k: object,
    density_threshold: object,
) -> tuple[int, float]:
    if (
        isinstance(selected_k, (bool, np.bool_))
        or not isinstance(selected_k, Integral)
        or int(selected_k) not in prepared.components
    ):
        raise InputFormatError("selected_k must be one of prepared.components")
    checked_k = int(selected_k)
    if isinstance(density_threshold, (bool, np.bool_)) or not isinstance(density_threshold, Real):
        raise InputFormatError("density_threshold must be a finite real number")
    checked_density = float(density_threshold)
    if not np.isfinite(checked_density):
        raise InputFormatError("density_threshold must be a finite real number")
    return checked_k, checked_density


def factorize_cnmf(
    prepared: PreparedCnmf,
    *,
    worker_index: int,
    total_workers: int,
) -> PreparedCnmf:
    """Run one documented cNMF factorization worker."""
    checked = _validated_prepared_cnmf(prepared)
    checked_worker, checked_workers = _validated_cnmf_workers(worker_index, total_workers)
    checked.backend.factorize(worker_i=checked_worker, total_workers=checked_workers)
    return checked


def select_cnmf_k(prepared: PreparedCnmf) -> PreparedCnmf:
    """Combine completed workers and generate cNMF's k-selection plot."""
    checked = _validated_prepared_cnmf(prepared)
    checked.backend.combine()
    checked.backend.k_selection_plot()
    return checked


def consensus_cnmf(
    prepared: PreparedCnmf,
    *,
    selected_k: int,
    density_threshold: float = 0.5,
) -> PreparedCnmf:
    """Build the documented cNMF consensus for one prepared component."""
    checked = _validated_prepared_cnmf(prepared)
    checked_k, checked_density = _validated_cnmf_consensus(
        checked,
        selected_k,
        density_threshold,
    )
    checked.backend.consensus(k=checked_k, density_threshold=checked_density)
    return checked


def run_cnmf(
    prepared: PreparedCnmf,
    *,
    worker_index: int,
    total_workers: int,
    selected_k: int,
    density_threshold: float = 0.5,
) -> PreparedCnmf:
    """Run the complete single-worker cNMF factorize/select/consensus workflow."""
    checked = _validated_prepared_cnmf(prepared)
    checked_worker, checked_workers = _validated_cnmf_workers(worker_index, total_workers)
    if checked_workers != 1:
        raise InputFormatError(
            "run_cnmf supports total_workers=1 only; distributed workflows must call "
            "factorize_cnmf for each worker, then select_cnmf_k and consensus_cnmf"
        )
    _validated_cnmf_consensus(checked, selected_k, density_threshold)

    factorize_cnmf(checked, worker_index=checked_worker, total_workers=checked_workers)
    select_cnmf_k(checked)
    consensus_cnmf(
        checked,
        selected_k=selected_k,
        density_threshold=density_threshold,
    )
    return checked


def _read_cnmf_usage(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"))
        if len(header) < 2:
            raise InputFormatError("cNMF usage must contain at least one usage column")
        usage_columns = header[1:]
        if any(not column for column in usage_columns):
            raise InputFormatError("cNMF usage column names must be non-empty")
        if len(set(usage_columns)) != len(usage_columns):
            raise InputFormatError("cNMF usage columns must be unique")
        table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except InputFormatError:
        raise
    except (OSError, StopIteration, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise InputFormatError(f"Could not read cNMF usage {path}: {exc}") from exc
    if table.shape[1] != len(header) or table.columns[1:].tolist() != usage_columns:
        raise InputFormatError("cNMF usage columns could not be read exactly")
    usage = table.iloc[:, 1:].copy()
    usage.index = pd.Index(table.iloc[:, 0].tolist(), name=header[0] or None)
    for column in usage.columns:
        try:
            usage[column] = pd.to_numeric(usage[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise InputFormatError("cNMF usage values must be real numeric values") from exc
    return usage


def load_cnmf_results(
    adata: anndata.AnnData,
    usage_path: str | Path,
) -> anndata.AnnData:
    """Attach a validated cNMF usage table by exact observation identifier alignment."""
    checked = _require_anndata(adata)
    source_joint = checked.uns.get("joint")
    if source_joint is not None and not isinstance(source_joint, Mapping):
        raise InputFormatError("AnnData uns['joint'] must be a mapping")
    path = _safe_path(usage_path, name="usage_path")
    usage = _read_cnmf_usage(path)
    if usage.index.hasnans or not usage.index.is_unique:
        raise InputFormatError("cNMF usage observation identifiers must be unique")
    expected = set(checked.obs_names)
    observed = set(usage.index)
    if observed != expected or len(usage.index) != checked.n_obs:
        raise InputFormatError("cNMF usage observations must exactly match AnnData observations")
    collisions = sorted(set(usage.columns).intersection(checked.obs.columns))
    if collisions:
        raise InputFormatError(f"cNMF usage columns collide with adata.obs: {collisions}")
    for column in usage.columns:
        series = usage[column]
        if (
            not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype)
        ):
            raise InputFormatError("cNMF usage values must be real numeric values")
    usage_values = usage.to_numpy()
    if not np.isfinite(usage_values).all():
        raise InputFormatError("cNMF usage values must be finite")
    if np.any(usage_values < 0):
        raise InputFormatError("cNMF usage values must be nonnegative")

    aligned = usage.loc[checked.obs_names]
    result = checked.copy()
    for column in aligned.columns:
        result.obs[column] = aligned[column].to_numpy(copy=True)
    joint_settings = dict(result.uns.get("joint", {}))
    joint_settings["cnmf_usage_path"] = str(path)
    result.uns["joint"] = joint_settings
    return result


__all__ = [
    "PreparedCnmf",
    "cluster_cells",
    "consensus_cnmf",
    "factorize_cnmf",
    "load_cnmf_results",
    "prepare_cnmf",
    "rank_metabolites",
    "run_cnmf",
    "run_cosg",
    "select_cnmf_k",
]
