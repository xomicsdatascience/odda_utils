# Download FASTA files from UniProt based on proteome information stored in the database.
#
# This module provides functionality to:
# - Query the uniprot_fasta table to find matching proteome entries
# - Construct UniProt FTP URLs from proteome metadata
# - Download and decompress FASTA files to a specified location
# - Compute SHA256 checksums for downloaded files
# - Gzip uncompressed files after SHA256 computation to ensure all stored files are compressed
# - Update the database with the local file path and sha256sum after successful download
# - Check for existing local copies before downloading (via database and filesystem verification)

import gzip
import hashlib
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("./articles.sqlite")
DEFAULT_FASTA_DIR = Path("/data/supporting/fasta/")

# UniProt FTP base URL for reference proteomes
UNIPROT_FTP_BASE = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes"

# Valid columns in the uniprot_fasta table for querying
VALID_QUERY_COLUMNS = frozenset([
    "proteome_id",
    "tax_id",
    "oscode",
    "superregnum",
    "species_name",
])


@dataclass
class UniProtFastaEntry:
    """Represents a UniProt proteome entry from the database."""

    proteome_id: str
    tax_id: int
    oscode: Optional[str]
    superregnum: str
    count_1: int
    count_2: int
    count_3: int
    species_name: str
    local_filepath: Optional[str] = None
    sha256sum: Optional[str] = None


@dataclass
class UniProtFastaDownloadResult:
    """Result of a UniProt FASTA download operation."""

    success: bool
    proteome_id: Optional[str] = None
    tax_id: Optional[int] = None
    species_name: Optional[str] = None
    ftp_url: Optional[str] = None
    local_filepath: Optional[str] = None
    file_size_bytes: Optional[int] = None
    sha256sum: Optional[str] = None
    error: Optional[str] = None


def _get_table_columns(conn: sqlite3.Connection) -> set[str]:
    """Get the column names from the uniprot_fasta table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    set[str]
        Set of column names in the uniprot_fasta table.
    """
    cursor = conn.execute("PRAGMA table_info(uniprot_fasta)")
    return {row[1] for row in cursor.fetchall()}


def _build_ftp_url(entry: UniProtFastaEntry) -> str:
    """Construct the UniProt FTP URL for a proteome FASTA file.

    The URL structure for UniProt reference proteomes is:
    {base}/{Superregnum}/{Proteome_ID}/{Proteome_ID}_{Tax_ID}.fasta.gz

    Parameters
    ----------
    entry : UniProtFastaEntry
        The proteome entry containing metadata needed for URL construction.

    Returns
    -------
    str
        The complete FTP URL for the FASTA file.
    """
    # Capitalize the first letter of superregnum for the directory name
    # e.g., "eukaryota" -> "Eukaryota", "bacteria" -> "Bacteria"
    superregnum_dir = entry.superregnum.capitalize()

    # Construct the URL
    filename = f"{entry.proteome_id}_{entry.tax_id}.fasta.gz"
    url = f"{UNIPROT_FTP_BASE}/{superregnum_dir}/{entry.proteome_id}/{filename}"

    return url


def _compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 checksum of a file.

    Parameters
    ----------
    file_path : Path
        Path to the file to compute the checksum for.

    Returns
    -------
    str
        The hexadecimal SHA256 checksum of the file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _gzip_file(source_path: Path, dest_path: Optional[Path] = None) -> Path:
    """Gzip a file and optionally remove the original.

    Compresses the source file using gzip and writes to the destination path.
    If no destination path is provided, appends '.gz' to the source path.
    After successful compression, the original uncompressed file is removed.

    Parameters
    ----------
    source_path : Path
        Path to the uncompressed file to gzip.
    dest_path : Path, optional
        Path where the gzipped file should be written.
        If not provided, uses source_path with '.gz' appended.

    Returns
    -------
    Path
        Path to the gzipped file.

    Raises
    ------
    IOError
        If reading the source file or writing the gzipped file fails.
    """
    if dest_path is None:
        dest_path = source_path.with_suffix(source_path.suffix + ".gz")

    logger.info(f"Gzipping {source_path} to {dest_path}")

    with open(source_path, "rb") as f_in:
        with gzip.open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the original uncompressed file
    source_path.unlink()
    logger.info(f"Removed original uncompressed file {source_path}")

    return dest_path


def _download_file(url: str, dest_path: Path, decompress: bool = True) -> int:
    """Download a file from a URL to a local path.

    Parameters
    ----------
    url : str
        The URL to download from.
    dest_path : Path
        The local path to save the file to.
    decompress : bool
        If True and the file is gzipped, decompress it.

    Returns
    -------
    int
        The size of the downloaded file in bytes.

    Raises
    ------
    URLError
        If the download fails due to network issues.
    HTTPError
        If the server returns an error status code.
    IOError
        If writing to the local file fails.
    """
    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up request with a user agent
    request = Request(
        url,
        headers={"User-Agent": "MCP-UniProt-Downloader/1.0"}
    )

    # Download the file
    logger.info(f"Downloading {url} to {dest_path}")

    if decompress and url.endswith(".gz"):
        # Download and decompress in one step
        with urlopen(request, timeout=300) as response:
            with gzip.GzipFile(fileobj=response) as gz_file:
                with open(dest_path, "wb") as out_file:
                    shutil.copyfileobj(gz_file, out_file)
    else:
        # Download as-is
        with urlopen(request, timeout=300) as response:
            with open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

    return dest_path.stat().st_size


def query_uniprot_fasta(
    conn: sqlite3.Connection,
    **kwargs,
) -> list[UniProtFastaEntry]:
    """Query the uniprot_fasta table with the given criteria.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    **kwargs
        Column name and value pairs to filter by. Valid columns are:
        proteome_id, tax_id, oscode, superregnum, species_name.

    Returns
    -------
    list[UniProtFastaEntry]
        List of matching entries.

    Raises
    ------
    ValueError
        If an invalid column name is provided.
    """
    if not kwargs:
        raise ValueError("At least one query parameter must be provided")

    # Validate column names
    invalid_columns = set(kwargs.keys()) - VALID_QUERY_COLUMNS
    if invalid_columns:
        raise ValueError(
            f"Invalid query columns: {invalid_columns}. "
            f"Valid columns are: {VALID_QUERY_COLUMNS}"
        )

    # Build the WHERE clause
    conditions = []
    params = []
    for column, value in kwargs.items():
        if value is not None:
            # Use LIKE for species_name to allow partial matching
            if column == "species_name":
                conditions.append(f"{column} LIKE ?")
                params.append(f"%{value}%")
            else:
                conditions.append(f"{column} = ?")
                params.append(value)

    if not conditions:
        raise ValueError("At least one non-None query parameter must be provided")

    where_clause = " AND ".join(conditions)

    # Use quoted column names for the special #(1), #(2), #(3) columns
    query = f"""
        SELECT proteome_id, tax_id, oscode, superregnum,
               "#(1)", "#(2)", "#(3)", species_name, local_filepath, sha256sum
        FROM uniprot_fasta
        WHERE {where_clause}
    """

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    entries = []
    for row in rows:
        entries.append(UniProtFastaEntry(
            proteome_id=row[0],
            tax_id=row[1],
            oscode=row[2] if row[2] != "None" else None,
            superregnum=row[3],
            count_1=row[4],
            count_2=row[5],
            count_3=row[6],
            species_name=row[7],
            local_filepath=row[8],
            sha256sum=row[9],
        ))

    return entries


def update_local_filepath(
    conn: sqlite3.Connection,
    proteome_id: str,
    local_filepath: str,
    sha256sum: Optional[str] = None,
) -> None:
    """Update the local_filepath and sha256sum for a proteome entry after download.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    proteome_id : str
        The proteome ID to update.
    local_filepath : str
        The local file path where the FASTA was saved.
    sha256sum : str, optional
        The SHA256 checksum of the downloaded file.
    """
    conn.execute(
        """
        UPDATE uniprot_fasta
        SET local_filepath = ?, sha256sum = ?
        WHERE proteome_id = ?
        """,
        (local_filepath, sha256sum, proteome_id),
    )
    conn.commit()


def download_uniprot_fasta(
    db_path: Path | str = DEFAULT_DB_PATH,
    download_dir: Path | str = DEFAULT_FASTA_DIR,
    decompress: bool = True,
    overwrite: bool = False,
    proteome_id: Optional[str] = None,
    tax_id: Optional[int] = None,
    oscode: Optional[str] = None,
    superregnum: Optional[str] = None,
    species_name: Optional[str] = None,
) -> UniProtFastaDownloadResult:
    """Download a UniProt FASTA file based on database query criteria.

    Queries the uniprot_fasta table to find a matching entry, constructs the
    UniProt FTP URL, downloads the FASTA file, computes its SHA256 checksum,
    and updates the database with the local file path and checksum.

    After computing the SHA256 checksum, if the downloaded file is not compressed
    (i.e., does not have a .gz extension), the file is gzipped to ensure all
    stored files are compressed. The database is updated with the final .gz path.

    Before downloading, checks if there is already a local copy by verifying:
    1. The database has a local_filepath defined for the proteome
    2. The file exists on disk at that path

    If both conditions are met and overwrite is False, returns early with
    the existing file information.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite database file.
    download_dir : Path or str
        Directory to save downloaded FASTA files.
    decompress : bool
        If True, decompress the gzipped FASTA file after download.
        Note: After SHA256 computation, uncompressed files are re-gzipped
        to ensure all stored files are compressed.
    overwrite : bool
        If True, download even if the file already exists locally.
    proteome_id : str, optional
        UniProt proteome ID (e.g., "UP000005640").
    tax_id : int, optional
        NCBI taxonomy ID.
    oscode : str, optional
        UniProt organism code (e.g., "HUMAN").
    superregnum : str, optional
        Domain/kingdom (e.g., "eukaryota", "bacteria", "viruses", "archaea").
    species_name : str, optional
        Species name (partial match supported).

    Returns
    -------
    UniProtFastaDownloadResult
        Result containing download status, file paths, sha256sum, and any errors.

    Notes
    -----
    At least one query parameter must be provided. If the query matches
    multiple entries, an error is returned. Use more specific criteria
    to narrow down to a single entry.

    The SHA256 checksum is computed on the downloaded file before any
    post-download gzip compression. This ensures the checksum reflects
    the original decompressed content when decompress=True.
    """
    db_path = Path(db_path)
    download_dir = Path(download_dir)

    # Build query kwargs from provided parameters
    query_kwargs = {}
    if proteome_id is not None:
        query_kwargs["proteome_id"] = proteome_id
    if tax_id is not None:
        query_kwargs["tax_id"] = tax_id
    if oscode is not None:
        query_kwargs["oscode"] = oscode
    if superregnum is not None:
        query_kwargs["superregnum"] = superregnum
    if species_name is not None:
        query_kwargs["species_name"] = species_name

    if not query_kwargs:
        return UniProtFastaDownloadResult(
            success=False,
            error="At least one query parameter must be provided",
        )

    # Connect to database and query
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        return UniProtFastaDownloadResult(
            success=False,
            error=f"Database connection failed: {e}",
        )

    try:
        # Check if the table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='uniprot_fasta'"
        )
        if cursor.fetchone() is None:
            return UniProtFastaDownloadResult(
                success=False,
                error="Table 'uniprot_fasta' does not exist in the database",
            )

        # Query for matching entries
        try:
            entries = query_uniprot_fasta(conn, **query_kwargs)
        except ValueError as e:
            return UniProtFastaDownloadResult(
                success=False,
                error=str(e),
            )

        # Check for no matches
        if not entries:
            return UniProtFastaDownloadResult(
                success=False,
                error=f"No matching entries found for query: {query_kwargs}",
            )

        # Check for multiple matches
        if len(entries) > 1:
            proteome_ids = [e.proteome_id for e in entries[:10]]
            more = f" (and {len(entries) - 10} more)" if len(entries) > 10 else ""
            return UniProtFastaDownloadResult(
                success=False,
                error=(
                    f"Query matched {len(entries)} entries. "
                    f"Please provide more specific criteria. "
                    f"Matching proteome IDs: {proteome_ids}{more}"
                ),
            )

        # Single match found
        entry = entries[0]

        # Check if already downloaded by verifying database has local_path
        # and the file exists on disk
        if entry.local_filepath and not overwrite:
            local_path = Path(entry.local_filepath)
            if local_path.exists():
                logger.info(
                    f"FASTA file already exists at {local_path} for proteome "
                    f"{entry.proteome_id} (use overwrite=True to re-download)"
                )
                return UniProtFastaDownloadResult(
                    success=True,
                    proteome_id=entry.proteome_id,
                    tax_id=entry.tax_id,
                    species_name=entry.species_name,
                    ftp_url=_build_ftp_url(entry),
                    local_filepath=str(local_path),
                    file_size_bytes=local_path.stat().st_size,
                    sha256sum=entry.sha256sum,
                )
            else:
                # Database says file exists but it's not on disk
                # Log this discrepancy and proceed with download
                logger.warning(
                    f"Database indicates local_filepath={entry.local_filepath} for "
                    f"proteome {entry.proteome_id}, but file does not exist on disk. "
                    "Proceeding with download."
                )

        # Build the FTP URL
        ftp_url = _build_ftp_url(entry)

        # Determine local file path for initial download
        # If decompress is True, download to .fasta first, then gzip after SHA256
        if decompress:
            filename = f"{entry.proteome_id}_{entry.tax_id}.fasta"
        else:
            filename = f"{entry.proteome_id}_{entry.tax_id}.fasta.gz"
        local_path = download_dir / filename

        # Download the file
        try:
            file_size = _download_file(ftp_url, local_path, decompress=decompress)
        except HTTPError as e:
            return UniProtFastaDownloadResult(
                success=False,
                proteome_id=entry.proteome_id,
                tax_id=entry.tax_id,
                species_name=entry.species_name,
                ftp_url=ftp_url,
                error=f"HTTP error during download: {e.code} {e.reason}",
            )
        except URLError as e:
            return UniProtFastaDownloadResult(
                success=False,
                proteome_id=entry.proteome_id,
                tax_id=entry.tax_id,
                species_name=entry.species_name,
                ftp_url=ftp_url,
                error=f"URL error during download: {e.reason}",
            )
        except IOError as e:
            return UniProtFastaDownloadResult(
                success=False,
                proteome_id=entry.proteome_id,
                tax_id=entry.tax_id,
                species_name=entry.species_name,
                ftp_url=ftp_url,
                error=f"IO error during download: {e}",
            )

        # Compute SHA256 checksum of the downloaded file (before any gzip compression)
        logger.info(f"Computing SHA256 checksum for {local_path}")
        sha256sum = _compute_sha256(local_path)
        logger.info(f"SHA256 checksum: {sha256sum}")

        # If the file is not compressed, gzip it to ensure all stored files are compressed
        if not str(local_path).endswith(".gz"):
            try:
                local_path = _gzip_file(local_path)
                file_size = local_path.stat().st_size
                logger.info(f"File gzipped to {local_path}, new size: {file_size} bytes")
            except IOError as e:
                return UniProtFastaDownloadResult(
                    success=False,
                    proteome_id=entry.proteome_id,
                    tax_id=entry.tax_id,
                    species_name=entry.species_name,
                    ftp_url=ftp_url,
                    error=f"IO error during gzip compression: {e}",
                )

        # Update the database with the local path and sha256sum
        update_local_filepath(conn, entry.proteome_id, str(local_path), sha256sum)

        return UniProtFastaDownloadResult(
            success=True,
            proteome_id=entry.proteome_id,
            tax_id=entry.tax_id,
            species_name=entry.species_name,
            ftp_url=ftp_url,
            local_filepath=str(local_path),
            file_size_bytes=file_size,
            sha256sum=sha256sum,
        )

    finally:
        conn.close()
