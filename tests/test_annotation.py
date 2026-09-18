import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

from joint.annotation import annotate_hmdb, annotate_metaboscape
from joint.errors import InputFormatError


def query_adata() -> anndata.AnnData:
    return anndata.AnnData(
        np.ones((1, 2)),
        var=pd.DataFrame({"mz": [100.0, 200.0]}, index=["m1", "m2"]),
    )


def test_hmdb_annotation_retains_multiple_matches_in_reference_order_without_mutation():
    adata = query_adata()
    reference = pd.DataFrame(
        {
            "accession": ["HMDB2", "HMDB1", "HMDB3"],
            "name": ["B", "A", "C"],
            "mz": [100.0004, 100.0002, 300.0],
            "mode": [" POS ", "pos", "pos"],
        }
    )
    original = adata.copy()
    original_reference = reference.copy(deep=True)

    annotated = annotate_hmdb(adata, reference, mode=" pos ", ppm=5)

    assert annotated is not adata
    assert annotated.var_names.tolist() == ["m1", "m2"]
    assert annotated.var["mz"].equals(original.var["mz"])
    assert "hmdb_accessions" not in adata.var
    assert adata.var.equals(original.var)
    assert reference.equals(original_reference)
    assert json.loads(annotated.var.loc["m1", "hmdb_accessions"]) == ["HMDB2", "HMDB1"]
    assert json.loads(annotated.var.loc["m1", "hmdb_names"]) == ["B", "A"]
    assert json.loads(annotated.var.loc["m2", "hmdb_accessions"]) == []


def test_metaboscape_annotation_matches_exact_ppm_boundary_and_keeps_json_unicode():
    reference = pd.DataFrame(
        {
            "Measured m/z": [100.0 * (1.0 + 3.0 / 1_000_000.0)],
            "Molecular Formula": ["C6H8"],
            "Name": ["café-β"],
        }
    )

    annotated = annotate_metaboscape(query_adata(), reference, ppm=3)

    assert json.loads(annotated.var.loc["m1", "metaboscape_formulas"]) == ["C6H8"]
    assert json.loads(annotated.var.loc["m1", "metaboscape_names"]) == ["café-β"]
    assert "café-β" in annotated.var.loc["m1", "metaboscape_names"]
    assert annotated.var["metaboscape_names"].dtype == object
    assert annotated.var["metaboscape_formulas"].dtype == object


@pytest.mark.parametrize("suffix, separator", [(".csv", ","), (".tsv", "\t"), (".txt", "\t")])
def test_hmdb_accepts_csv_tsv_and_text_reference_paths(
    tmp_path: Path, suffix: str, separator: str
):
    path = tmp_path / f"hmdb{suffix}"
    pd.DataFrame(
        {"accession": ["HMDB1"], "name": ["A"], "mz": [100.0], "mode": ["pos"]}
    ).to_csv(path, index=False, sep=separator)

    annotated = annotate_hmdb(query_adata(), path, mode="POS")

    assert json.loads(annotated.var.loc["m1", "hmdb_accessions"]) == ["HMDB1"]


def test_annotation_accepts_categorical_and_numeric_string_reference_columns():
    reference = pd.DataFrame(
        {
            "accession": pd.Categorical(["HMDB1"]),
            "name": pd.Categorical(["candidate"]),
            "mz": pd.Categorical(["100.0"]),
            "mode": pd.Categorical([" POS "]),
        }
    )

    annotated = annotate_hmdb(query_adata(), reference, mode="pos")

    assert json.loads(annotated.var.loc["m1", "hmdb_accessions"]) == ["HMDB1"]


@pytest.mark.parametrize(
    ("adata", "match"),
    [
        (None, "AnnData"),
        (anndata.AnnData(np.ones((1, 1))), 'column "mz"'),
        (anndata.AnnData(np.ones((1, 1)), var={"mz": ["bad"]}), "numeric"),
        (anndata.AnnData(np.ones((1, 1)), var={"mz": [np.inf]}), "finite"),
        (anndata.AnnData(np.ones((1, 1)), var={"mz": [0.0]}), "positive"),
        (anndata.AnnData(np.ones((1, 1)), var={"mz": [True]}), "numeric"),
    ],
)
def test_annotation_rejects_invalid_query_anndata(adata: object, match: str):
    reference = pd.DataFrame(
        {"accession": ["HMDB1"], "name": ["A"], "mz": [100.0], "mode": ["pos"]}
    )

    with pytest.raises(InputFormatError, match=match):
        annotate_hmdb(adata, reference, mode="pos")


def test_annotation_rejects_nonunique_query_feature_names():
    adata = query_adata()
    adata.var.index = ["duplicate", "duplicate"]
    reference = pd.DataFrame(
        {"accession": ["HMDB1"], "name": ["A"], "mz": [100.0], "mode": ["pos"]}
    )

    with pytest.raises(InputFormatError, match="unique"):
        annotate_hmdb(adata, reference, mode="pos")


@pytest.mark.parametrize("ppm", [True, -0.1, np.inf, np.nan, "5"])
def test_annotation_rejects_invalid_ppm(ppm: object):
    reference = pd.DataFrame(
        {"accession": ["HMDB1"], "name": ["A"], "mz": [100.0], "mode": ["pos"]}
    )

    with pytest.raises(InputFormatError, match="ppm"):
        annotate_hmdb(query_adata(), reference, mode="pos", ppm=ppm)


@pytest.mark.parametrize("mode", [None, "", "   ", 1])
def test_hmdb_annotation_rejects_empty_or_non_string_mode(mode: object):
    reference = pd.DataFrame(
        {"accession": ["HMDB1"], "name": ["A"], "mz": [100.0], "mode": ["pos"]}
    )

    with pytest.raises(InputFormatError, match="mode"):
        annotate_hmdb(query_adata(), reference, mode=mode)


@pytest.mark.parametrize(
    ("reference", "match"),
    [
        (None, "DataFrame"),
        (42, "DataFrame"),
        (pd.DataFrame({"accession": ["a"]}), "columns"),
        (
            pd.DataFrame(
                {"accession": ["a"], "name": ["A"], "mz": [100.0], "mode": ["pos"], "extra": [1]}
            ),
            "columns",
        ),
        (pd.DataFrame({"accession": ["a"], "name": ["A"], "mz": ["bad"], "mode": ["pos"]}), "numeric"),
        (pd.DataFrame({"accession": ["a"], "name": ["A"], "mz": [np.inf], "mode": ["pos"]}), "finite"),
        (pd.DataFrame({"accession": ["a"], "name": ["A"], "mz": [0.0], "mode": ["pos"]}), "positive"),
        (pd.DataFrame({"accession": ["a"], "name": ["A"], "mz": [True], "mode": ["pos"]}), "numeric"),
        (pd.DataFrame({"accession": [None], "name": ["A"], "mz": [100.0], "mode": ["pos"]}), "accession"),
        (pd.DataFrame({"accession": ["a"], "name": [pd.NA], "mz": [100.0], "mode": ["pos"]}), "name"),
    ],
)
def test_hmdb_annotation_rejects_invalid_references(reference: object, match: str):
    with pytest.raises(InputFormatError, match=match):
        annotate_hmdb(query_adata(), reference, mode="pos")


def test_annotation_wraps_missing_and_malformed_reference_file_errors(tmp_path: Path):
    missing = tmp_path / "missing.tsv"
    malformed = tmp_path / "malformed.csv"
    malformed.write_text('accession,name,mz,mode\n"unterminated,A,100,pos\n')

    for path in (missing, malformed):
        with pytest.raises(InputFormatError, match="Could not read reference") as error:
            annotate_hmdb(query_adata(), path, mode="pos")
        assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    ("reference", "match"),
    [
        (pd.DataFrame({"Measured m/z": [100.0]}), "columns"),
        (
            pd.DataFrame(
                {
                    "Measured m/z": [100.0],
                    "Molecular Formula": ["C6H8"],
                    "Name": ["candidate"],
                    "extra": [1],
                }
            ),
            "columns",
        ),
        (pd.DataFrame({"Measured m/z": ["bad"], "Molecular Formula": ["C6H8"], "Name": ["A"]}), "numeric"),
        (pd.DataFrame({"Measured m/z": [100.0], "Molecular Formula": [None], "Name": ["A"]}), "Formula"),
        (pd.DataFrame({"Measured m/z": [100.0], "Molecular Formula": ["C6H8"], "Name": [None]}), "Name"),
    ],
)
def test_metaboscape_annotation_rejects_invalid_references(reference: pd.DataFrame, match: str):
    with pytest.raises(InputFormatError, match=match):
        annotate_metaboscape(query_adata(), reference)


def test_empty_hmdb_mode_filtered_reference_produces_empty_json_lists():
    reference = pd.DataFrame(
        {"accession": ["HMDB1"], "name": ["A"], "mz": [100.0], "mode": ["neg"]}
    )

    annotated = annotate_hmdb(query_adata(), reference, mode="pos")

    assert annotated.var["hmdb_accessions"].tolist() == ["[]", "[]"]
    assert annotated.var["hmdb_names"].tolist() == ["[]", "[]"]


def test_hmdb_annotation_includes_decimal_ppm_boundaries_but_rejects_outside_values():
    reference = pd.DataFrame(
        {
            "accession": ["lower", "upper", "below", "above"],
            "name": ["lower", "upper", "below", "above"],
            "mz": [99.9995, 100.0005, 99.99949, 100.00051],
            "mode": ["pos", "pos", "pos", "pos"],
        }
    )

    annotated = annotate_hmdb(query_adata(), reference, mode="pos", ppm=5)

    assert json.loads(annotated.var.loc["m1", "hmdb_accessions"]) == ["lower", "upper"]


def test_annotation_handles_finite_extreme_mz_without_overflow_warnings():
    adata = anndata.AnnData(
        np.ones((1, 1)),
        var=pd.DataFrame({"mz": [np.nextafter(0.0, 1.0)]}, index=["m1"]),
    )
    reference = pd.DataFrame(
        {
            "Measured m/z": [np.finfo(float).max],
            "Molecular Formula": ["C1"],
            "Name": ["far-away"],
        }
    )

    annotated = annotate_metaboscape(adata, reference, ppm=np.finfo(float).max)

    assert annotated.var.loc["m1", "metaboscape_names"] == "[]"


def test_annotation_requires_exact_mz_when_ppm_is_zero():
    reference = pd.DataFrame(
        {
            "Measured m/z": [np.nextafter(100.0, np.inf)],
            "Molecular Formula": ["C1"],
            "Name": ["adjacent-float"],
        }
    )

    annotated = annotate_metaboscape(query_adata(), reference, ppm=0)

    assert annotated.var.loc["m1", "metaboscape_names"] == "[]"


@pytest.mark.parametrize(
    "mz_values",
    [
        pd.Series([100.0 + 0.0j]),
        pd.Series([100.0 + 0.0j], dtype=object),
        pd.Series(pd.Categorical([100.0 + 0.0j])),
    ],
)
def test_annotation_rejects_complex_query_mz_before_float_conversion(mz_values: pd.Series):
    var = pd.DataFrame({"mz": mz_values})
    var.index = ["m1"]
    adata = anndata.AnnData(np.ones((1, 1)), var=var)
    reference = pd.DataFrame(
        {"accession": ["HMDB1"], "name": ["A"], "mz": [100.0], "mode": ["pos"]}
    )

    with pytest.raises(InputFormatError, match="real"):
        annotate_hmdb(adata, reference, mode="pos")


@pytest.mark.parametrize(
    "mz_values",
    [
        pd.Series([100.0 + 0.0j]),
        pd.Series([100.0 + 0.0j], dtype=object),
        pd.Series(pd.Categorical([100.0 + 0.0j])),
    ],
)
def test_annotation_rejects_complex_reference_mz_before_float_conversion(mz_values: pd.Series):
    reference = pd.DataFrame(
        {
            "Measured m/z": mz_values,
            "Molecular Formula": ["C1"],
            "Name": ["candidate"],
        }
    )

    with pytest.raises(InputFormatError, match="real"):
        annotate_metaboscape(query_adata(), reference)
