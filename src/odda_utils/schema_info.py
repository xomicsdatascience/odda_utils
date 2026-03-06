# Database schema introspection utilities.
# Provides functions to query SQLite database schema information including
# tables, columns, types, and constraints.

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    data_type: str
    nullable: bool
    default_value: Optional[str]
    is_primary_key: bool


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    columns: list[ColumnInfo] = field(default_factory=list)


@dataclass
class SchemaInfoResult:
    """Result of querying database schema information.

    Attributes
    ----------
    tables : list[TableInfo]
        List of tables with their column information.
    table_count : int
        Total number of tables in the database.
    error : str, optional
        Error message if the operation failed, None otherwise.
    """

    tables: list[TableInfo] = field(default_factory=list)
    table_count: int = 0
    error: Optional[str] = None


def get_schema_info(
    db_path: str | Path,
    table_name: Optional[str] = None,
) -> SchemaInfoResult:
    """Get schema information for tables in a SQLite database.

    Queries the SQLite database to retrieve information about tables and their
    columns, including column names, data types, nullability, default values,
    and primary key status.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    table_name : str, optional
        If provided, only return information for this specific table.
        If None, return information for all tables.

    Returns
    -------
    SchemaInfoResult
        Result containing:
        - tables: List of TableInfo objects with column details
        - table_count: Number of tables returned
        - error: Error message if the operation failed
    """
    result = SchemaInfoResult()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of tables
        if table_name:
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name = ?
                ORDER BY name
                """,
                (table_name,),
            )
        else:
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
                """
            )

        tables = cursor.fetchall()

        for (tbl_name,) in tables:
            # Get column info using PRAGMA table_info
            cursor.execute(f"PRAGMA table_info({tbl_name})")
            columns_raw = cursor.fetchall()

            columns = []
            for col in columns_raw:
                # PRAGMA table_info returns:
                # cid, name, type, notnull, dflt_value, pk
                cid, col_name, col_type, notnull, dflt_value, pk = col
                columns.append(
                    ColumnInfo(
                        name=col_name,
                        data_type=col_type if col_type else "BLOB",
                        nullable=not notnull,
                        default_value=dflt_value,
                        is_primary_key=bool(pk),
                    )
                )

            result.tables.append(TableInfo(name=tbl_name, columns=columns))

        result.table_count = len(result.tables)
        conn.close()

    except sqlite3.Error as e:
        result.error = f"Database error: {e}"
    except Exception as e:
        result.error = f"Unexpected error: {e}"

    return result
