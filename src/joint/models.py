from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    regions: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrationResult:
    adata: anndata.AnnData
    mapping: pd.DataFrame
    transform: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantificationResult:
    adata: anndata.AnnData
    overlaps: pd.DataFrame
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    artifacts: dict[str, Path]
    manifest: Path
    reports: dict[str, Any] = field(default_factory=dict)
