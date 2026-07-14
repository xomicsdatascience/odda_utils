# Bounded, LLM-safe summaries of omics tables/matrices.
#
# Cost- and safety-control for the ODDA trust/context boundary: a whole omics
# quantification matrix (thousands of features x many samples) must NEVER be
# placed into a model's context -- it is expensive and unnecessary. This module
# is the sanctioned way to let an agent understand a table's STRUCTURE and
# content (shape, columns, dtypes, per-column numeric statistics or top
# categorical values, and a few truncated example rows) without ever emitting
# the full matrix. Python (pandas/numpy) does all of the table work here; the
# returned object is small, JSON-serializable, and hard-capped along the row,
# column, and cell dimensions so the output size is bounded regardless of input
# size. Actual quantitative computation on matrices (QC, differential
# expression, meta-analysis) is done elsewhere in Python (the sandboxed
# ``run_analysis`` container and ``meta_analysis``); only these compact
# summaries -- not raw matrices -- should ever reach an LLM.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Hard caps that guarantee the output is a SUMMARY, never the whole matrix.
DEFAULT_MAX_COLUMNS_DETAILED = 100
DEFAULT_MAX_EXAMPLE_ROWS = 5
DEFAULT_MAX_CELL_CHARS = 80
DEFAULT_MAX_TOP_VALUES = 5
# Cap on rows pandas scans, to bound host memory/time on pathological inputs.
DEFAULT_MAX_SCAN_ROWS = 2_000_000
# Columns with no more than this many distinct values are summarized by their
# top values rather than treated as free text.
_CATEGORICAL_MAX_UNIQUE = 50


@dataclass
class ColumnSummary:
    """Compact summary of a single table column.

    Attributes
    ----------
    name : str
        Column name (truncated to the cell-char cap).
    dtype : str
        Pandas dtype string.
    non_null : int
        Number of non-null values.
    null_count : int
        Number of null values.
    n_unique : int
        Number of distinct values.
    is_numeric : bool
        Whether the column is numeric.
    min, max, mean, median, std : float or None
        Numeric statistics (None for non-numeric columns or when undefined).
    top_values : list
        For non-numeric / low-cardinality columns, up to ``max_top_values``
        ``[value, count]`` pairs (value truncated). Empty otherwise.
    """

    name: str
    dtype: str
    non_null: int
    null_count: int
    n_unique: int
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    top_values: list = field(default_factory=list)


@dataclass
class TableSummary:
    """Bounded, JSON-serializable summary of a table/matrix.

    The row, column, and cell dimensions are all hard-capped so the summary can
    never reproduce the full matrix, regardless of input size.

    Attributes
    ----------
    source : str
        The path (or label) that was summarized.
    file_type : str
        Detected file type (e.g. ``"csv"``, ``"tsv"``, ``"excel"``,
        ``"parquet"``).
    n_rows : int
        Number of rows scanned (see ``rows_truncated``).
    n_cols : int
        Number of columns in the table.
    rows_truncated : bool
        True when the table had more rows than ``max_scan_rows`` and only the
        leading window was scanned (``n_rows`` is then the scanned count).
    n_columns_described : int
        Number of columns detailed in ``columns`` (capped by
        ``max_columns_detailed``).
    columns : list of ColumnSummary
        Per-column summaries (capped).
    example_rows : list of dict
        A few example rows (capped), with each cell coerced to a string and
        truncated. Only described columns are included.
    sheet : str or None
        Excel sheet name, if applicable.
    delimiter : str or None
        Detected delimiter for delimited text files.
    file_size_bytes : int or None
        Size of the source file on disk.
    notes : list of str
        Free-text notes (caps applied, truncation, etc.).
    error : str or None
        Error message if the table could not be summarized.
    """

    source: str
    file_type: str = "unknown"
    n_rows: int = 0
    n_cols: int = 0
    rows_truncated: bool = False
    n_columns_described: int = 0
    columns: list[ColumnSummary] = field(default_factory=list)
    example_rows: list[dict] = field(default_factory=list)
    sheet: Optional[str] = None
    delimiter: Optional[str] = None
    file_size_bytes: Optional[int] = None
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None


def _truncate(value: Any, max_chars: int) -> str:
    """Coerce a cell value to a string and truncate it to ``max_chars``."""
    text = "" if value is None else str(value)
    text = " ".join(text.split())  # collapse whitespace/newlines
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def _finite_or_none(value: Any) -> Optional[float]:
    """Return a plain float if finite, else None (JSON-safe)."""
    import math

    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(fval) or math.isinf(fval):
        return None
    return fval


def _detect_file_type(path: Path, delimiter: Optional[str]) -> tuple[str, Optional[str]]:
    """Detect file type and (for delimited text) delimiter from the suffix."""
    suffixes = [s.lower() for s in path.suffixes]
    flat = "".join(suffixes)
    if any(s in (".xlsx", ".xls", ".xlsm") for s in suffixes):
        return "excel", None
    if ".parquet" in suffixes:
        return "parquet", None
    if ".feather" in suffixes:
        return "feather", None
    if ".tsv" in flat or ".tab" in flat:
        return "tsv", delimiter or "\t"
    if ".csv" in flat:
        return "csv", delimiter or ","
    # Unknown text: let pandas sniff the delimiter.
    return "delimited", delimiter


def _read_table(
    path: Path,
    file_type: str,
    delimiter: Optional[str],
    sheet: Optional[str],
    max_scan_rows: int,
):
    """Read a bounded number of rows of a table into a pandas DataFrame.

    Returns
    -------
    tuple
        ``(dataframe, used_delimiter, used_sheet)``.
    """
    import pandas as pd

    if file_type == "excel":
        frame = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
        used_sheet = sheet if sheet is not None else (
            frame.attrs.get("sheet_name") if hasattr(frame, "attrs") else None
        )
        if len(frame) > max_scan_rows:
            frame = frame.iloc[:max_scan_rows]
        return frame, None, (str(sheet) if sheet is not None else None)

    if file_type == "parquet":
        frame = pd.read_parquet(path)
        if len(frame) > max_scan_rows:
            frame = frame.iloc[:max_scan_rows]
        return frame, None, None

    if file_type == "feather":
        frame = pd.read_feather(path)
        if len(frame) > max_scan_rows:
            frame = frame.iloc[:max_scan_rows]
        return frame, None, None

    # Delimited text (csv/tsv/unknown).
    read_kwargs: dict[str, Any] = {"nrows": max_scan_rows}
    if delimiter:
        read_kwargs["sep"] = delimiter
    else:
        read_kwargs["sep"] = None
        read_kwargs["engine"] = "python"
    frame = pd.read_csv(path, **read_kwargs)
    used_delim = delimiter
    return frame, used_delim, None


def summarize_table(
    path: str | Path,
    sheet: Optional[str] = None,
    delimiter: Optional[str] = None,
    max_columns_detailed: int = DEFAULT_MAX_COLUMNS_DETAILED,
    max_example_rows: int = DEFAULT_MAX_EXAMPLE_ROWS,
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
    max_top_values: int = DEFAULT_MAX_TOP_VALUES,
    max_scan_rows: int = DEFAULT_MAX_SCAN_ROWS,
) -> TableSummary:
    """Summarize a table/matrix into a bounded, LLM-safe description.

    Reads the table with pandas (Python does all the table work) and returns a
    compact summary: shape, per-column dtype/null/uniqueness and either numeric
    statistics or top categorical values, plus a few truncated example rows. The
    output is hard-capped along the row, column, and cell dimensions so it can
    never reproduce the full matrix -- use this instead of ever loading a whole
    omics matrix into a model's context.

    Parameters
    ----------
    path : str or Path
        Path to the table file (CSV/TSV/other delimited text, Excel, Parquet,
        or Feather).
    sheet : str, optional
        Excel sheet name (defaults to the first sheet).
    delimiter : str, optional
        Field delimiter for delimited text; auto-sniffed when omitted.
    max_columns_detailed : int, optional
        Cap on the number of columns detailed in the summary.
    max_example_rows : int, optional
        Cap on the number of example rows returned.
    max_cell_chars : int, optional
        Cap on the length of any single cell/value string in the output.
    max_top_values : int, optional
        Cap on the number of top values reported per categorical column.
    max_scan_rows : int, optional
        Cap on the number of rows pandas scans (bounds host memory/time).

    Returns
    -------
    TableSummary
        The bounded summary. On a read/parse failure the ``error`` field is set
        (and the rest is left at defaults) rather than raising, so the tool is
        robust at the MCP boundary.
    """
    source = str(path)
    file_path = Path(path)
    summary = TableSummary(source=source)

    if not file_path.exists():
        summary.error = f"File not found: {source}"
        return summary

    try:
        summary.file_size_bytes = file_path.stat().st_size
    except OSError:
        summary.file_size_bytes = None

    file_type, detected_delim = _detect_file_type(file_path, delimiter)
    summary.file_type = file_type

    try:
        import pandas as pd  # noqa: F401  (ensure available; used in helpers)

        frame, used_delim, used_sheet = _read_table(
            file_path, file_type, detected_delim, sheet, max_scan_rows
        )
    except Exception as exc:  # noqa: BLE001 - report, do not crash the server
        summary.error = f"Could not read table: {exc}"
        return summary

    summary.delimiter = used_delim or detected_delim
    summary.sheet = used_sheet
    summary.n_rows = int(len(frame))
    summary.n_cols = int(frame.shape[1])
    if summary.n_rows >= max_scan_rows:
        summary.rows_truncated = True
        summary.notes.append(
            f"Only the leading {max_scan_rows} rows were scanned; the file may "
            "have more."
        )

    import pandas.api.types as ptypes

    described_columns = list(frame.columns[:max_columns_detailed])
    if summary.n_cols > max_columns_detailed:
        summary.notes.append(
            f"Described the first {max_columns_detailed} of {summary.n_cols} "
            "columns."
        )

    for col in described_columns:
        series = frame[col]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        try:
            n_unique = int(series.nunique(dropna=True))
        except TypeError:  # unhashable cell types
            n_unique = -1

        is_numeric = bool(ptypes.is_numeric_dtype(series))
        col_summary = ColumnSummary(
            name=_truncate(col, max_cell_chars),
            dtype=str(series.dtype),
            non_null=non_null,
            null_count=null_count,
            n_unique=n_unique,
            is_numeric=is_numeric,
        )

        if is_numeric and non_null > 0:
            col_summary.min = _finite_or_none(series.min())
            col_summary.max = _finite_or_none(series.max())
            col_summary.mean = _finite_or_none(series.mean())
            col_summary.median = _finite_or_none(series.median())
            col_summary.std = _finite_or_none(series.std())
        elif not is_numeric and 0 <= n_unique <= _CATEGORICAL_MAX_UNIQUE:
            try:
                counts = series.value_counts(dropna=True).head(max_top_values)
                col_summary.top_values = [
                    [_truncate(idx, max_cell_chars), int(cnt)]
                    for idx, cnt in counts.items()
                ]
            except TypeError:
                col_summary.top_values = []

        summary.columns.append(col_summary)

    summary.n_columns_described = len(summary.columns)

    # Example rows: bounded rows x described columns, every cell truncated.
    head = frame[described_columns].head(max_example_rows)
    for _, row in head.iterrows():
        summary.example_rows.append(
            {
                _truncate(col, max_cell_chars): _truncate(row[col], max_cell_chars)
                for col in described_columns
            }
        )

    return summary
