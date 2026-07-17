# Unit tests for odda_utils.table_summary. Verifies that summarize_table emits a
# bounded, JSON-serializable summary of a table/matrix and never reproduces the
# full matrix: the row, column, and cell dimensions are all hard-capped, numeric
# columns get statistics, low-cardinality columns get top values, column/row caps
# are honoured, and read failures are reported (not raised). Depends on
# pandas + numpy; no network, no model, no code execution.

import json
import os
import tempfile
import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

from odda_utils.table_summary import summarize_table


def _make_matrix(path, n_rows=5000, n_samples=8):
    df = pd.DataFrame({"protein_id": [f"P{i:05d}" for i in range(n_rows)]})
    rng = np.random.default_rng(0)
    for s in range(n_samples):
        df[f"sample_{s}"] = rng.normal(20, 3, n_rows)
    df["group"] = ["treated" if i % 2 else "control" for i in range(n_rows)]
    df.to_csv(path, index=False)
    return df


class TestTableSummary(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.csv = os.path.join(self.dir, "matrix.csv")
        self.df = _make_matrix(self.csv)

    def test_shape_and_bounded_output(self):
        s = summarize_table(self.csv)
        self.assertIsNone(s.error)
        self.assertEqual(s.n_rows, 5000)
        self.assertEqual(s.n_cols, 10)
        self.assertLessEqual(len(s.example_rows), 5)
        js = json.dumps(asdict(s))
        # Summary is far smaller than the raw matrix on disk.
        self.assertLess(len(js), os.path.getsize(self.csv) / 20)

    def test_cells_truncated(self):
        s = summarize_table(self.csv, max_cell_chars=10)
        for row in s.example_rows:
            for value in row.values():
                self.assertLessEqual(len(value), 10)

    def test_numeric_stats_and_top_values(self):
        s = summarize_table(self.csv)
        by = {c.name: c for c in s.columns}
        self.assertTrue(by["sample_0"].is_numeric)
        self.assertIsNotNone(by["sample_0"].mean)
        self.assertEqual(
            sorted(v[0] for v in by["group"].top_values), ["control", "treated"]
        )

    def test_column_cap(self):
        s = summarize_table(self.csv, max_columns_detailed=3)
        self.assertEqual(s.n_columns_described, 3)
        self.assertEqual(s.n_cols, 10)
        self.assertTrue(any("first 3 of 10" in n for n in s.notes))

    def test_row_scan_cap_flagged(self):
        s = summarize_table(self.csv, max_scan_rows=100)
        self.assertTrue(s.rows_truncated)
        self.assertEqual(s.n_rows, 100)

    def test_tsv_detection(self):
        tsv = os.path.join(self.dir, "m.tsv")
        self.df.head(50).to_csv(tsv, sep="\t", index=False)
        s = summarize_table(tsv)
        self.assertIsNone(s.error)
        self.assertEqual(s.file_type, "tsv")
        self.assertEqual(s.n_rows, 50)

    def test_missing_file_reports_error(self):
        s = summarize_table(os.path.join(self.dir, "nope.csv"))
        self.assertIsNotNone(s.error)
        self.assertIn("not found", s.error.lower())

    def test_json_serializable(self):
        s = summarize_table(self.csv)
        # Must not raise (no numpy scalars / NaN / inf leaking through).
        json.dumps(asdict(s))


if __name__ == "__main__":
    unittest.main()
