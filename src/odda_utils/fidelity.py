# Pure, deterministic "fidelity report" utilities for quantifying how closely an
# ODDA-reproduced omics result (proteomics / RNA-seq) matches a published result.
# Provides: identification-level overlap, quantitative agreement (Pearson/Spearman
# on log intensities), differential-expression (DEP) overlap with a three-bucket
# decomposition of the non-reproduced published hits, and a tool-version
# identification-gain/loss helper. All functions are network-free and LLM-free and
# depend only on numpy + the standard library (pandas is optional). Outputs are
# plain dataclasses containing JSON-serializable primitives.

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

try:  # pandas is optional; the module works fully without it.
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - exercised only when pandas is absent
    _pd = None


# ---------------------------------------------------------------------------
# Input containers
# ---------------------------------------------------------------------------


@dataclass
class AbundanceMatrix:
    """A feature-by-sample abundance matrix.

    Parameters
    ----------
    feature_ids : list of str
        Feature identifiers (e.g. UniProt accessions or gene ids), one per row.
    sample_names : list of str
        Sample/column names, one per column.
    values : numpy.ndarray
        2-D array of shape ``(n_features, n_samples)`` holding intensities.
        Missing values should be encoded as ``numpy.nan``.
    """

    feature_ids: list[str]
    sample_names: list[str]
    values: np.ndarray

    def __post_init__(self) -> None:
        self.feature_ids = [str(f) for f in self.feature_ids]
        self.sample_names = [str(s) for s in self.sample_names]
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2:
            raise ValueError("values must be a 2-D array (features x samples)")
        if self.values.shape != (len(self.feature_ids), len(self.sample_names)):
            raise ValueError(
                "values shape %s does not match (%d features, %d samples)"
                % (self.values.shape, len(self.feature_ids), len(self.sample_names))
            )


@dataclass
class DepRecord:
    """A single differential-expression result row.

    Parameters
    ----------
    feature_id : str
        Feature identifier.
    log2fc : float, optional
        Log2 fold change.
    pvalue : float, optional
        Raw p-value.
    padj : float, optional
        Adjusted p-value (e.g. Benjamini-Hochberg).
    significant : bool, optional
        Explicit significance flag. When ``None`` the significance is derived
        from ``padj``/``pvalue`` and the configured thresholds.
    """

    feature_id: str
    log2fc: Optional[float] = None
    pvalue: Optional[float] = None
    padj: Optional[float] = None
    significant: Optional[bool] = None


# ---------------------------------------------------------------------------
# Output containers (JSON-serializable primitives only)
# ---------------------------------------------------------------------------


@dataclass
class IdentificationComparison:
    """Identification-level (feature membership) comparison result."""

    n_reproduced: int
    n_published: int
    n_shared: int
    n_reproduced_only: int
    n_published_only: int
    jaccard: float
    shared_features: list[str] = field(default_factory=list)
    reproduced_only_features: list[str] = field(default_factory=list)
    published_only_features: list[str] = field(default_factory=list)


@dataclass
class SampleCorrelation:
    """Per-sample quantitative agreement on shared features."""

    reproduced_sample: str
    published_sample: str
    n: int
    pearson: Optional[float] = None
    spearman: Optional[float] = None


@dataclass
class QuantitativeAgreement:
    """Quantitative agreement of intensities on shared features."""

    n_shared_features: int
    log_transformed: bool
    log_base: Optional[float]
    pooled_n: int
    pooled_pearson: Optional[float] = None
    pooled_spearman: Optional[float] = None
    sample_correlations: list[SampleCorrelation] = field(default_factory=list)


@dataclass
class DepDecomposition:
    """DEP overlap and decomposition of the non-reproduced published hits.

    The four attribution counts partition every published-significant feature:
    ``n_reproduced_concordant + not_quantified + quantified_not_significant +
    significant_different_direction == n_published_significant``.
    """

    n_reproduced_significant: int
    n_published_significant: int
    n_shared_significant: int
    jaccard_significant: float
    overlap_pct_of_published: float
    n_direction_agree: int
    n_direction_disagree: int
    n_reproduced_concordant: int
    not_quantified: int
    quantified_not_significant: int
    significant_different_direction: int
    significance_threshold: float
    lfc_threshold: float
    used_padj: bool
    shared_significant_features: list[str] = field(default_factory=list)
    reproduced_concordant_features: list[str] = field(default_factory=list)
    not_quantified_features: list[str] = field(default_factory=list)
    quantified_not_significant_features: list[str] = field(default_factory=list)
    significant_different_direction_features: list[str] = field(default_factory=list)


@dataclass
class VersionComparison:
    """Identification gain/loss between two tool versions."""

    label_a: str
    label_b: str
    n_a: int
    n_b: int
    n_shared: int
    n_gained: int
    n_lost: int
    jaccard: float
    gained_features: list[str] = field(default_factory=list)
    lost_features: list[str] = field(default_factory=list)


@dataclass
class FidelityReport:
    """Top-level fidelity report bundling the requested comparison sections."""

    identification: Optional[IdentificationComparison] = None
    quantitative: Optional[QuantitativeAgreement] = None
    dep: Optional[DepDecomposition] = None
    version: Optional[VersionComparison] = None
    notes: list[str] = field(default_factory=list)
    recorded_analysis_run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Low-level numeric helpers
# ---------------------------------------------------------------------------


def _to_float(value) -> float:
    """Parse a cell into a float, mapping blanks / NA sentinels to NaN.

    Parameters
    ----------
    value : object
        Raw cell content.

    Returns
    -------
    float
        Parsed value, or ``numpy.nan`` when the value is missing or unparseable.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        v = float(value)
        return v
    s = str(value).strip()
    if s == "" or s.lower() in {
        "na",
        "nan",
        "n/a",
        "#n/a",
        "null",
        "none",
        "filtered",
        "inf",
        "-inf",
    }:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Pearson correlation of two 1-D arrays (already NaN-filtered)."""
    n = x.size
    if n < 2:
        return None
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    if denom == 0.0:
        return None
    r = float((xm * ym).sum() / denom)
    # Guard against tiny floating-point excursions beyond [-1, 1].
    return max(-1.0, min(1.0, r))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Assign ranks to data, averaging ranks of ties (like scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = a.argsort(kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # mean of 1-based ranks i+1..j+1
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _correlate(
    x: Sequence[float], y: Sequence[float]
) -> tuple[Optional[float], Optional[float], int]:
    """Compute Pearson and Spearman correlations over NaN-aligned pairs.

    Parameters
    ----------
    x, y : sequence of float
        Paired observations.

    Returns
    -------
    tuple
        ``(pearson, spearman, n)`` where ``n`` is the number of finite pairs and
        the correlations are ``None`` when undefined (fewer than two pairs or
        zero variance).
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = int(xa.size)
    pearson = _pearson(xa, ya)
    spearman = _pearson(_rankdata(xa), _rankdata(ya)) if n >= 2 else None
    return pearson, spearman, n


def _log_transform(
    arr: np.ndarray, base: float, pseudocount: float, enabled: bool
) -> np.ndarray:
    """Log-transform an array; non-positive inputs become NaN.

    Parameters
    ----------
    arr : numpy.ndarray
        Intensity values.
    base : float
        Logarithm base (e.g. 2.0).
    pseudocount : float
        Value added before taking the logarithm.
    enabled : bool
        When ``False`` the array is returned unchanged (as float).

    Returns
    -------
    numpy.ndarray
        Transformed values.
    """
    arr = np.asarray(arr, dtype=float)
    if not enabled:
        return arr
    shifted = arr + pseudocount
    out = np.full(arr.shape, np.nan, dtype=float)
    positive = np.isfinite(shifted) & (shifted > 0)
    out[positive] = np.log(shifted[positive]) / np.log(base)
    return out


def _sign(value: Optional[float]) -> Optional[int]:
    """Return the sign of a fold change (1, -1) or ``None`` when undefined/zero."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    if v > 0:
        return 1
    if v < 0:
        return -1
    return None


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


MatrixLike = Union["AbundanceMatrix", str, Path, object]


def _coerce_matrix(
    obj: MatrixLike,
    id_column: Optional[str] = None,
    intensity_columns: Optional[Sequence[str]] = None,
    sep: Optional[str] = None,
) -> AbundanceMatrix:
    """Coerce an input into an :class:`AbundanceMatrix`.

    Accepts an :class:`AbundanceMatrix`, a pandas ``DataFrame`` (features in the
    index unless ``id_column`` is given), or a filesystem path to a delimited
    table.
    """
    if isinstance(obj, AbundanceMatrix):
        return obj
    if _pd is not None and isinstance(obj, _pd.DataFrame):
        df = obj
        if id_column is not None and id_column in df.columns:
            feature_ids = df[id_column].astype(str).tolist()
            value_df = df.drop(columns=[id_column])
        else:
            feature_ids = [str(x) for x in df.index.tolist()]
            value_df = df
        if intensity_columns is not None:
            value_df = value_df[list(intensity_columns)]
        sample_names = [str(c) for c in value_df.columns.tolist()]
        values = np.asarray(value_df.to_numpy(), dtype=float)
        return AbundanceMatrix(feature_ids, sample_names, values)
    if isinstance(obj, (str, Path)):
        return load_matrix(
            obj,
            id_column=id_column,
            intensity_columns=intensity_columns,
            sep=sep,
        )
    raise TypeError(f"Cannot coerce object of type {type(obj)!r} to AbundanceMatrix")


def _as_id_list(obj) -> list[str]:
    """Extract a list of feature ids from a matrix, id iterable, or path."""
    if isinstance(obj, AbundanceMatrix):
        return list(obj.feature_ids)
    if isinstance(obj, (str, Path)):
        return list(_coerce_matrix(obj).feature_ids)
    if _pd is not None and isinstance(obj, _pd.DataFrame):
        return list(_coerce_matrix(obj).feature_ids)
    if isinstance(obj, dict):
        return [str(k) for k in obj.keys()]
    if isinstance(obj, Iterable):
        return [str(x) for x in obj]
    raise TypeError(f"Cannot interpret object of type {type(obj)!r} as feature ids")


def _as_dep_records(obj) -> list[DepRecord]:
    """Coerce an input into a list of :class:`DepRecord`.

    Accepts a list of :class:`DepRecord`, a list of dicts with canonical keys
    (``feature_id``, ``log2fc``, ``pvalue``, ``padj``, ``significant``), a pandas
    ``DataFrame`` with those columns, or a filesystem path to a delimited table.
    """
    if obj is None:
        return []
    if isinstance(obj, (str, Path)):
        return load_dep_results(obj)
    if _pd is not None and isinstance(obj, _pd.DataFrame):
        obj = obj.to_dict(orient="records")
    records: list[DepRecord] = []
    for item in obj:
        if isinstance(item, DepRecord):
            records.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(
                f"DEP records must be DepRecord or dict, got {type(item)!r}"
            )
        sig = item.get("significant", None)
        if sig is not None and not isinstance(sig, bool):
            sig = _parse_bool(sig)
        records.append(
            DepRecord(
                feature_id=str(item.get("feature_id")),
                log2fc=_optional_float(item.get("log2fc")),
                pvalue=_optional_float(item.get("pvalue")),
                padj=_optional_float(item.get("padj")),
                significant=sig,
            )
        )
    return records


def _optional_float(value) -> Optional[float]:
    """Return ``value`` as float, or ``None`` for missing/NaN values."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    v = _to_float(value)
    if math.isnan(v):
        return None
    return v


def _parse_bool(value) -> Optional[bool]:
    """Parse a truthy/falsey cell into a bool, or ``None`` when unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "y", "1", "sig", "significant", "+"}:
        return True
    if s in {"false", "f", "no", "n", "0", "ns", "nonsig", "not_significant", ""}:
        return False
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _guess_sep(path: Path, sep: Optional[str]) -> str:
    """Return the field delimiter, inferring from the extension when not given."""
    if sep is not None:
        return sep
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    return "\t"


def _read_delimited(path: Union[str, Path], sep: Optional[str]) -> tuple[list[str], list[dict]]:
    """Read a delimited text file into a header list and list of row dicts.

    Parameters
    ----------
    path : str or Path
        Path to the delimited file.
    sep : str, optional
        Field delimiter. Inferred from the file extension when ``None``.

    Returns
    -------
    tuple
        ``(headers, rows)`` where ``rows`` is a list of ``dict`` keyed by header.
    """
    path = Path(path)
    delimiter = _guess_sep(path, sep)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        return [], []
    headers = [h.strip() for h in rows[0]]
    records: list[dict] = []
    for raw in rows[1:]:
        if not raw:
            continue
        record = {headers[i]: (raw[i] if i < len(raw) else "") for i in range(len(headers))}
        records.append(record)
    return headers, records


#: Columns that DIA-NN ``report.pg_matrix.tsv`` files carry as metadata.
DIANN_METADATA_COLUMNS = (
    "Protein.Group",
    "Protein.Ids",
    "Protein.Names",
    "Genes",
    "First.Protein.Description",
)


def load_matrix(
    path: Union[str, Path],
    id_column: Optional[str] = None,
    intensity_columns: Optional[Sequence[str]] = None,
    sep: Optional[str] = None,
    metadata_columns: Optional[Sequence[str]] = None,
) -> AbundanceMatrix:
    """Load a feature-by-sample abundance matrix from a delimited text file.

    Parameters
    ----------
    path : str or Path
        Path to a CSV/TSV file.
    id_column : str, optional
        Name of the feature-id column. When ``None`` the first column is used.
    intensity_columns : sequence of str, optional
        Explicit list of sample/intensity columns. When ``None`` every column
        except the id column and any ``metadata_columns`` is treated as a sample.
    sep : str, optional
        Field delimiter. Inferred from the extension when ``None``.
    metadata_columns : sequence of str, optional
        Non-sample columns to exclude when ``intensity_columns`` is not given.

    Returns
    -------
    AbundanceMatrix
        The parsed matrix.
    """
    headers, rows = _read_delimited(path, sep)
    if not headers:
        return AbundanceMatrix([], [], np.empty((0, 0), dtype=float))

    id_col = id_column if id_column is not None else headers[0]
    if id_col not in headers:
        raise ValueError(f"id_column {id_col!r} not found in {headers}")

    excluded = {id_col}
    if metadata_columns:
        excluded.update(c for c in metadata_columns if c in headers)

    if intensity_columns is not None:
        sample_cols = [c for c in intensity_columns if c in headers]
    else:
        sample_cols = [c for c in headers if c not in excluded]

    feature_ids = [str(r.get(id_col, "")) for r in rows]
    if rows:
        values = np.array(
            [[_to_float(r.get(c)) for c in sample_cols] for r in rows],
            dtype=float,
        )
    else:
        values = np.empty((0, len(sample_cols)), dtype=float)
    return AbundanceMatrix(feature_ids, sample_cols, values)


def load_diann_pg_matrix(
    path: Union[str, Path],
    id_column: str = "Protein.Group",
    intensity_columns: Optional[Sequence[str]] = None,
    sep: Optional[str] = None,
) -> AbundanceMatrix:
    """Load a DIA-NN ``report.pg_matrix.tsv`` protein-group matrix.

    Uses ``Protein.Group`` as the feature id and treats every non-metadata
    column as a sample intensity column.

    Parameters
    ----------
    path : str or Path
        Path to the DIA-NN pg matrix file.
    id_column : str, optional
        Feature-id column name. Defaults to ``"Protein.Group"``.
    intensity_columns : sequence of str, optional
        Explicit sample columns; auto-detected when ``None``.
    sep : str, optional
        Field delimiter; defaults to tab.

    Returns
    -------
    AbundanceMatrix
        The parsed matrix.
    """
    return load_matrix(
        path,
        id_column=id_column,
        intensity_columns=intensity_columns,
        sep=sep,
        metadata_columns=DIANN_METADATA_COLUMNS,
    )


def load_maxquant_protein_groups(
    path: Union[str, Path],
    id_column: str = "Majority protein IDs",
    intensity_prefix: str = "LFQ intensity ",
    intensity_columns: Optional[Sequence[str]] = None,
    sep: Optional[str] = None,
) -> AbundanceMatrix:
    """Load a MaxQuant ``proteinGroups.txt`` matrix.

    Sample columns are those beginning with ``intensity_prefix`` (default
    ``"LFQ intensity "``); the sample name is the remainder of the header. Falls
    back to ``"Protein IDs"`` when the preferred id column is absent, and to the
    ``"Intensity "`` prefix when no LFQ columns are present.

    Parameters
    ----------
    path : str or Path
        Path to the MaxQuant proteinGroups file.
    id_column : str, optional
        Feature-id column name. Defaults to ``"Majority protein IDs"``.
    intensity_prefix : str, optional
        Prefix identifying per-sample intensity columns.
    intensity_columns : sequence of str, optional
        Explicit sample columns; auto-detected when ``None``.
    sep : str, optional
        Field delimiter; defaults to tab.

    Returns
    -------
    AbundanceMatrix
        The parsed matrix; sample names have the intensity prefix stripped.
    """
    headers, rows = _read_delimited(path, sep)
    if not headers:
        return AbundanceMatrix([], [], np.empty((0, 0), dtype=float))

    id_col = id_column
    if id_col not in headers:
        for candidate in ("Majority protein IDs", "Protein IDs", "id"):
            if candidate in headers:
                id_col = candidate
                break
    if id_col not in headers:
        raise ValueError(f"No usable id column found in {headers}")

    if intensity_columns is not None:
        sample_cols = [c for c in intensity_columns if c in headers]
        sample_names = [str(c) for c in sample_cols]
    else:
        prefix = intensity_prefix
        sample_cols = [
            c for c in headers if c.startswith(prefix) and c != prefix.strip()
        ]
        if not sample_cols:
            prefix = "Intensity "
            sample_cols = [
                c for c in headers if c.startswith(prefix) and c != prefix.strip()
            ]
        sample_names = [c[len(prefix):] for c in sample_cols]

    feature_ids = [str(r.get(id_col, "")) for r in rows]
    if rows:
        values = np.array(
            [[_to_float(r.get(c)) for c in sample_cols] for r in rows],
            dtype=float,
        )
    else:
        values = np.empty((0, len(sample_cols)), dtype=float)
    return AbundanceMatrix(feature_ids, sample_names, values)


def load_dep_results(
    path: Union[str, Path],
    id_column: str = "feature_id",
    log2fc_column: str = "log2fc",
    pvalue_column: Optional[str] = "pvalue",
    padj_column: Optional[str] = "padj",
    significant_column: Optional[str] = "significant",
    sep: Optional[str] = None,
) -> list[DepRecord]:
    """Load differential-expression results from a delimited text file.

    Only columns that are present in the file are read; configured column names
    that are absent are ignored, so partial tables load cleanly.

    Parameters
    ----------
    path : str or Path
        Path to a CSV/TSV file.
    id_column : str, optional
        Feature-id column name.
    log2fc_column : str, optional
        Log2 fold-change column name.
    pvalue_column : str, optional
        Raw p-value column name.
    padj_column : str, optional
        Adjusted p-value column name.
    significant_column : str, optional
        Explicit significance-flag column name.
    sep : str, optional
        Field delimiter; inferred from the extension when ``None``.

    Returns
    -------
    list of DepRecord
        Parsed records.
    """
    headers, rows = _read_delimited(path, sep)
    if not headers:
        return []
    if id_column not in headers:
        raise ValueError(f"id_column {id_column!r} not found in {headers}")

    def _col(name: Optional[str]) -> Optional[str]:
        return name if (name is not None and name in headers) else None

    lfc_c = _col(log2fc_column)
    p_c = _col(pvalue_column)
    padj_c = _col(padj_column)
    sig_c = _col(significant_column)

    records: list[DepRecord] = []
    for r in rows:
        records.append(
            DepRecord(
                feature_id=str(r.get(id_column, "")),
                log2fc=_optional_float(r.get(lfc_c)) if lfc_c else None,
                pvalue=_optional_float(r.get(p_c)) if p_c else None,
                padj=_optional_float(r.get(padj_c)) if padj_c else None,
                significant=_parse_bool(r.get(sig_c)) if sig_c else None,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------


def _jaccard(n_intersection: int, n_union: int) -> float:
    """Jaccard index with a zero-union guard."""
    if n_union == 0:
        return 0.0
    return n_intersection / n_union


def compare_identifications(
    reproduced,
    published,
    include_feature_lists: bool = True,
) -> IdentificationComparison:
    """Compare feature membership between reproduced and published results.

    Parameters
    ----------
    reproduced, published : AbundanceMatrix or iterable of str or path
        The two identification sets. Anything :func:`_as_id_list` understands is
        accepted (matrix, list/set of ids, or a delimited-file path).
    include_feature_lists : bool, optional
        When ``True`` (default) the sorted feature-id lists are included in the
        result; set ``False`` to return counts only.

    Returns
    -------
    IdentificationComparison
        Shared / reproduced-only / published-only counts and the Jaccard index.
    """
    rep = set(_as_id_list(reproduced))
    pub = set(_as_id_list(published))
    shared = rep & pub
    rep_only = rep - pub
    pub_only = pub - rep
    union = rep | pub
    result = IdentificationComparison(
        n_reproduced=len(rep),
        n_published=len(pub),
        n_shared=len(shared),
        n_reproduced_only=len(rep_only),
        n_published_only=len(pub_only),
        jaccard=_jaccard(len(shared), len(union)),
    )
    if include_feature_lists:
        result.shared_features = sorted(shared)
        result.reproduced_only_features = sorted(rep_only)
        result.published_only_features = sorted(pub_only)
    return result


def compare_versions(
    features_a,
    features_b,
    label_a: str = "version_a",
    label_b: str = "version_b",
    include_feature_lists: bool = True,
) -> VersionComparison:
    """Compare identification sets between two tool versions.

    Gains are features present in ``features_b`` (the "new" version) but not in
    ``features_a``; losses are the reverse. This underpins the tool-version
    protein-count-gap explanation.

    Parameters
    ----------
    features_a, features_b : AbundanceMatrix or iterable of str or path
        Identification sets for version A (baseline) and version B (comparison).
    label_a, label_b : str, optional
        Human-readable labels for the two versions.
    include_feature_lists : bool, optional
        When ``True`` (default) include the gained/lost feature-id lists.

    Returns
    -------
    VersionComparison
        Gained / lost / shared counts and the Jaccard index.
    """
    a = set(_as_id_list(features_a))
    b = set(_as_id_list(features_b))
    shared = a & b
    gained = b - a
    lost = a - b
    union = a | b
    result = VersionComparison(
        label_a=label_a,
        label_b=label_b,
        n_a=len(a),
        n_b=len(b),
        n_shared=len(shared),
        n_gained=len(gained),
        n_lost=len(lost),
        jaccard=_jaccard(len(shared), len(union)),
    )
    if include_feature_lists:
        result.gained_features = sorted(gained)
        result.lost_features = sorted(lost)
    return result


def compare_quantitative(
    reproduced,
    published,
    sample_map: Optional[dict] = None,
    log_transform: bool = True,
    log_base: float = 2.0,
    pseudocount: float = 0.0,
    id_column: Optional[str] = None,
    intensity_columns: Optional[Sequence[str]] = None,
    sep: Optional[str] = None,
) -> QuantitativeAgreement:
    """Quantitative agreement of intensities on shared features.

    Computes per-sample and pooled Pearson and Spearman correlations of the
    (optionally log-transformed) intensities restricted to shared features.

    Parameters
    ----------
    reproduced, published : AbundanceMatrix or pandas.DataFrame or path
        The two abundance matrices (features x samples).
    sample_map : dict, optional
        Mapping of reproduced sample name -> published sample name. When
        ``None`` samples present in both matrices (by identical name) are paired.
    log_transform : bool, optional
        Whether to log-transform intensities before correlating. Default True.
    log_base : float, optional
        Logarithm base used when ``log_transform`` is True. Default 2.0.
    pseudocount : float, optional
        Value added before taking the logarithm. Default 0.0.
    id_column, intensity_columns, sep : optional
        Passed through to the loader when a path is supplied.

    Returns
    -------
    QuantitativeAgreement
        Per-sample and pooled correlations on shared features.
    """
    rep = _coerce_matrix(reproduced, id_column, intensity_columns, sep)
    pub = _coerce_matrix(published, id_column, intensity_columns, sep)

    shared = sorted(set(rep.feature_ids) & set(pub.feature_ids))
    agreement = QuantitativeAgreement(
        n_shared_features=len(shared),
        log_transformed=log_transform,
        log_base=float(log_base) if log_transform else None,
        pooled_n=0,
    )
    if not shared:
        return agreement

    rep_index = {f: i for i, f in enumerate(rep.feature_ids)}
    pub_index = {f: i for i, f in enumerate(pub.feature_ids)}
    rep_pos = [rep_index[f] for f in shared]
    pub_pos = [pub_index[f] for f in shared]
    rep_sub = rep.values[rep_pos, :]
    pub_sub = pub.values[pub_pos, :]
    rep_col_index = {s: j for j, s in enumerate(rep.sample_names)}
    pub_col_index = {s: j for j, s in enumerate(pub.sample_names)}

    if sample_map:
        pairs = [
            (str(rs), str(ps))
            for rs, ps in sample_map.items()
            if str(rs) in rep_col_index and str(ps) in pub_col_index
        ]
    else:
        common = sorted(set(rep.sample_names) & set(pub.sample_names))
        pairs = [(s, s) for s in common]

    pooled_rep: list[np.ndarray] = []
    pooled_pub: list[np.ndarray] = []
    for rep_sample, pub_sample in pairs:
        rep_col = _log_transform(
            rep_sub[:, rep_col_index[rep_sample]], log_base, pseudocount, log_transform
        )
        pub_col = _log_transform(
            pub_sub[:, pub_col_index[pub_sample]], log_base, pseudocount, log_transform
        )
        pearson, spearman, n = _correlate(rep_col, pub_col)
        agreement.sample_correlations.append(
            SampleCorrelation(
                reproduced_sample=rep_sample,
                published_sample=pub_sample,
                n=n,
                pearson=pearson,
                spearman=spearman,
            )
        )
        pooled_rep.append(rep_col)
        pooled_pub.append(pub_col)

    if pooled_rep:
        pearson, spearman, n = _correlate(
            np.concatenate(pooled_rep), np.concatenate(pooled_pub)
        )
        agreement.pooled_pearson = pearson
        agreement.pooled_spearman = spearman
        agreement.pooled_n = n
    return agreement


def _effective_significance(
    rec: DepRecord,
    significance_threshold: float,
    lfc_threshold: float,
    use_padj: bool,
) -> bool:
    """Resolve a record's significance, honoring an explicit flag when present."""
    if rec.significant is not None:
        return bool(rec.significant)
    metric: Optional[float] = None
    if use_padj and rec.padj is not None and not math.isnan(rec.padj):
        metric = rec.padj
    elif rec.pvalue is not None and not math.isnan(rec.pvalue):
        metric = rec.pvalue
    if metric is None:
        return False
    passes = metric <= significance_threshold
    if lfc_threshold > 0:
        if rec.log2fc is None or math.isnan(rec.log2fc):
            return False
        passes = passes and abs(rec.log2fc) >= lfc_threshold
    return passes


def compare_deps(
    reproduced,
    published,
    reproduced_quantified_ids: Optional[Iterable[str]] = None,
    significance_threshold: float = 0.05,
    lfc_threshold: float = 0.0,
    use_padj: bool = True,
    include_feature_lists: bool = True,
) -> DepDecomposition:
    """Compare DEP result sets and decompose the non-reproduced published hits.

    The significant-set overlap (count / Jaccard / percent of published) is
    direction-agnostic, mirroring the published overlap metric. Separately,
    every published-significant feature is partitioned into exactly one of four
    buckets: concordantly reproduced (significant in both, same direction),
    ``not_quantified`` (absent from the reproduced results/matrix),
    ``quantified_not_significant`` (present but not significant), or
    ``significant_different_direction`` (significant but opposite sign).

    Parameters
    ----------
    reproduced, published : list of DepRecord or list of dict or path
        The two DEP result sets.
    reproduced_quantified_ids : iterable of str, optional
        Feature ids quantified in the reproduced analysis but possibly filtered
        out of its DEP table (e.g. the reproduced abundance-matrix features).
        Used to distinguish ``not_quantified`` from ``quantified_not_significant``.
    significance_threshold : float, optional
        Threshold applied to ``padj``/``pvalue`` when a record lacks an explicit
        significance flag. Default 0.05.
    lfc_threshold : float, optional
        Minimum absolute log2 fold change required for derived significance.
        Default 0.0 (no fold-change filter).
    use_padj : bool, optional
        Prefer ``padj`` over ``pvalue`` for derived significance. Default True.
    include_feature_lists : bool, optional
        When ``True`` (default) include per-bucket feature-id lists.

    Returns
    -------
    DepDecomposition
        Overlap metrics, direction agreement, and the four-bucket attribution.
    """
    rep_records = _as_dep_records(reproduced)
    pub_records = _as_dep_records(published)

    rep_by_id: dict[str, DepRecord] = {}
    for rec in rep_records:
        rep_by_id.setdefault(rec.feature_id, rec)
    pub_by_id: dict[str, DepRecord] = {}
    for rec in pub_records:
        pub_by_id.setdefault(rec.feature_id, rec)

    rep_sig_ids = {
        fid
        for fid, rec in rep_by_id.items()
        if _effective_significance(rec, significance_threshold, lfc_threshold, use_padj)
    }
    pub_sig_ids = {
        fid
        for fid, rec in pub_by_id.items()
        if _effective_significance(rec, significance_threshold, lfc_threshold, use_padj)
    }

    quantified_ids = set(rep_by_id.keys())
    if reproduced_quantified_ids is not None:
        quantified_ids.update(str(x) for x in reproduced_quantified_ids)

    shared_sig = rep_sig_ids & pub_sig_ids
    union_sig = rep_sig_ids | pub_sig_ids

    # Direction agreement among the (direction-agnostic) shared-significant set.
    n_agree = 0
    n_disagree = 0
    for fid in shared_sig:
        rd = _sign(rep_by_id[fid].log2fc)
        pd = _sign(pub_by_id[fid].log2fc)
        if rd is not None and pd is not None and rd == pd:
            n_agree += 1
        else:
            n_disagree += 1

    # Four-bucket partition of every published-significant feature.
    concordant: list[str] = []
    not_quantified: list[str] = []
    quant_not_sig: list[str] = []
    diff_direction: list[str] = []
    for fid in pub_sig_ids:
        pub_dir = _sign(pub_by_id[fid].log2fc)
        if fid not in quantified_ids:
            not_quantified.append(fid)
        elif fid in rep_sig_ids:
            rep_dir = _sign(rep_by_id[fid].log2fc)
            if rep_dir is not None and pub_dir is not None and rep_dir == pub_dir:
                concordant.append(fid)
            else:
                diff_direction.append(fid)
        else:
            quant_not_sig.append(fid)

    result = DepDecomposition(
        n_reproduced_significant=len(rep_sig_ids),
        n_published_significant=len(pub_sig_ids),
        n_shared_significant=len(shared_sig),
        jaccard_significant=_jaccard(len(shared_sig), len(union_sig)),
        overlap_pct_of_published=(
            100.0 * len(shared_sig) / len(pub_sig_ids) if pub_sig_ids else 0.0
        ),
        n_direction_agree=n_agree,
        n_direction_disagree=n_disagree,
        n_reproduced_concordant=len(concordant),
        not_quantified=len(not_quantified),
        quantified_not_significant=len(quant_not_sig),
        significant_different_direction=len(diff_direction),
        significance_threshold=float(significance_threshold),
        lfc_threshold=float(lfc_threshold),
        used_padj=bool(use_padj),
    )
    if include_feature_lists:
        result.shared_significant_features = sorted(shared_sig)
        result.reproduced_concordant_features = sorted(concordant)
        result.not_quantified_features = sorted(not_quantified)
        result.quantified_not_significant_features = sorted(quant_not_sig)
        result.significant_different_direction_features = sorted(diff_direction)
    return result


def assemble_report(
    identification: Optional[IdentificationComparison] = None,
    quantitative: Optional[QuantitativeAgreement] = None,
    dep: Optional[DepDecomposition] = None,
    version: Optional[VersionComparison] = None,
    notes: Optional[list[str]] = None,
) -> FidelityReport:
    """Bundle the individual comparison sections into a :class:`FidelityReport`.

    Parameters
    ----------
    identification, quantitative, dep, version : optional
        The comparison sections to include; omit any that were not computed.
    notes : list of str, optional
        Free-text notes to attach to the report.

    Returns
    -------
    FidelityReport
        The assembled report.
    """
    return FidelityReport(
        identification=identification,
        quantitative=quantitative,
        dep=dep,
        version=version,
        notes=list(notes) if notes else [],
    )
