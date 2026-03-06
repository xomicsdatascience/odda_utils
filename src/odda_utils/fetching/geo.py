"""Gene Expression Omnibus (GEO) dataset fetching via FTP and E-utilities.

This module provides functionality to download datasets from GEO (Gene Expression
Omnibus), an NCBI database that stores gene expression and other functional
genomics data. Datasets are identified by GSE IDs (e.g., GSE12345).
"""

import ftplib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)

GEO_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
GEO_FTP_BASE = "/geo/series"
GEO_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
GEO_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


@dataclass
class GEOFileInfo:
    """Information about a single file in a GEO dataset."""

    filename: str
    path: str
    size_bytes: int


@dataclass
class GEOFileSizeResult:
    """Result of querying file sizes for a GEO dataset."""

    gse_id: str
    title: str | None = None
    files: list[GEOFileInfo] = field(default_factory=list)
    total_size_bytes: int = 0
    file_count: int = 0
    error: str | None = None


@dataclass
class GEODownloadResult:
    """Result of a GEO dataset download."""

    gse_id: str
    title: str | None = None
    summary: str | None = None
    downloaded_files: list[Path] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class GSEMetadata:
    """Metadata for a GEO Series dataset from NCBI GEO.

    Mirrors the PXDMetadata structure for consistency across dataset types.
    """

    dataset_id: str
    title: str | None = None
    description: str | None = None
    species: list[str] = field(default_factory=list)
    platform: str | None = None
    repository: str = "GEO"
    submission_date: str | None = None
    publication_date: str | None = None
    n_samples: int | None = None
    linked_doi: str | None = None
    linked_pmid: str | None = None
    linked_pmcid: str | None = None
    error: str | None = None


GEO_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"


def _validate_gse_id(gse_id: str) -> str:
    """Validate and normalize GEO Series identifier.

    Parameters
    ----------
    gse_id : str
        GEO Series identifier (e.g., "GSE12345").

    Returns
    -------
    str
        Normalized identifier.

    Raises
    ------
    ValueError
        If identifier format is invalid.
    """
    gse_id = gse_id.strip().upper()
    if not re.match(r"^GSE\d+$", gse_id):
        raise ValueError(
            f"Invalid GEO Series ID format: {gse_id}. "
            "Expected format: GSE followed by digits (e.g., GSE12345)"
        )
    return gse_id


def _get_ftp_directory(gse_id: str) -> str:
    """Get the FTP directory path for a GSE dataset.

    GEO organizes datasets into range directories by replacing the last 3 digits
    with 'nnn'. For example:
    - GSE1 -> /geo/series/GSEnnn/GSE1/
    - GSE1234 -> /geo/series/GSE1nnn/GSE1234/
    - GSE12345 -> /geo/series/GSE12nnn/GSE12345/

    Parameters
    ----------
    gse_id : str
        GEO Series identifier (e.g., "GSE12345").

    Returns
    -------
    str
        FTP directory path.
    """
    # Extract the numeric part
    numeric_part = gse_id[3:]

    # Create the range directory name
    if len(numeric_part) <= 3:
        range_dir = "GSEnnn"
    else:
        range_dir = f"GSE{numeric_part[:-3]}nnn"

    return f"{GEO_FTP_BASE}/{range_dir}/{gse_id}"


def _get_gse_metadata(gse_id: str, timeout: float) -> dict:
    """Fetch GSE metadata from NCBI E-utilities.

    Parameters
    ----------
    gse_id : str
        GEO Series identifier.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    dict
        Metadata dictionary with title, summary, etc.
    """
    # First search for the GEO ID to get the UID
    search_params = {
        "db": "gds",
        "term": f"{gse_id}[Accession]",
        "retmode": "json",
    }

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        search_response = client.get(GEO_ESEARCH_URL, params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {}

        # Get summary for the first result
        summary_params = {
            "db": "gds",
            "id": id_list[0],
            "retmode": "json",
        }
        summary_response = client.get(GEO_ESUMMARY_URL, params=summary_params)
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        result = summary_data.get("result", {})
        if id_list[0] in result:
            doc = result[id_list[0]]
            return {
                "title": doc.get("title"),
                "summary": doc.get("summary"),
                "gpl": doc.get("gpl"),
                "gse": doc.get("gse"),
                "taxon": doc.get("taxon"),
                "n_samples": doc.get("n_samples"),
            }

    return {}


def _get_linked_pubmed_ids(gds_uid: str, timeout: float) -> dict:
    """Fetch linked PubMed IDs for a GDS record.

    Parameters
    ----------
    gds_uid : str
        GDS database UID (not the GSE accession).
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    dict
        Dictionary with 'pmid' and optionally 'doi' keys.
    """
    # Use elink to find PubMed links
    link_params = {
        "dbfrom": "gds",
        "db": "pubmed",
        "id": gds_uid,
        "retmode": "json",
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            link_response = client.get(GEO_ELINK_URL, params=link_params)
            link_response.raise_for_status()
            link_data = link_response.json()

            linksets = link_data.get("linksets", [])
            if linksets:
                linksetdbs = linksets[0].get("linksetdbs", [])
                for linksetdb in linksetdbs:
                    if linksetdb.get("dbto") == "pubmed":
                        links = linksetdb.get("links", [])
                        if links:
                            return {"pmid": str(links[0])}
    except Exception as e:
        logger.debug("Could not fetch linked PubMed IDs: %s", e)

    return {}


def fetch_gse_metadata(
    gse_id: str,
    timeout: float = 30.0,
) -> GSEMetadata:
    """Fetch dataset metadata from NCBI GEO E-utilities API.

    Retrieves project metadata including title, description, species,
    submission/publication dates, and linked publications.

    Parameters
    ----------
    gse_id : str
        GEO Series identifier (e.g., "GSE12345").
    timeout : float, optional
        Request timeout in seconds, by default 30.0.

    Returns
    -------
    GSEMetadata
        Dataset metadata from NCBI GEO.

    Examples
    --------
    >>> metadata = fetch_gse_metadata("GSE12345")
    >>> print(metadata.title)
    >>> print(metadata.species)
    """
    # Validate and normalize GSE ID
    try:
        gse_id = _validate_gse_id(gse_id)
    except ValueError as e:
        return GSEMetadata(dataset_id=gse_id, error=str(e))

    result = GSEMetadata(dataset_id=gse_id)

    # Search for the GEO ID to get the UID
    search_params = {
        "db": "gds",
        "term": f"{gse_id}[Accession]",
        "retmode": "json",
    }

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            search_response = client.get(GEO_ESEARCH_URL, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                result.error = f"Dataset {gse_id} not found in GEO"
                return result

            gds_uid = id_list[0]

            # Get summary for the first result
            summary_params = {
                "db": "gds",
                "id": gds_uid,
                "retmode": "json",
            }
            summary_response = client.get(GEO_ESUMMARY_URL, params=summary_params)
            summary_response.raise_for_status()
            summary_data = summary_response.json()

            doc = summary_data.get("result", {}).get(gds_uid, {})

            if doc:
                result.title = doc.get("title")
                result.description = doc.get("summary")
                result.platform = doc.get("gpl")
                result.n_samples = doc.get("n_samples")

                # Parse taxon - can be a string like "Homo sapiens"
                taxon = doc.get("taxon")
                if taxon:
                    if isinstance(taxon, str):
                        result.species = [taxon]
                    elif isinstance(taxon, list):
                        result.species = taxon

                # Parse dates - GEO esummary returns pdat in YYYY/MM/DD format
                pdat = doc.get("pdat")
                if pdat and isinstance(pdat, str):
                    # Convert YYYY/MM/DD to YYYY-MM-DD
                    result.publication_date = pdat.replace("/", "-")

            # Try to get linked PubMed IDs
            pubmed_info = _get_linked_pubmed_ids(gds_uid, timeout)
            if pubmed_info.get("pmid"):
                result.linked_pmid = pubmed_info["pmid"]

    except httpx.HTTPStatusError as e:
        logger.error("Failed to fetch GEO metadata for %s: %s", gse_id, e)
        result.error = f"Failed to fetch metadata: {e}"
    except Exception as e:
        logger.error("Error fetching GEO metadata for %s: %s", gse_id, e)
        result.error = f"Error fetching metadata: {e}"

    return result


def _list_ftp_files(
    ftp: ftplib.FTP,
    path: str,
    base_path: str,
) -> list[GEOFileInfo]:
    """List all files in an FTP directory (non-recursive for GEO).

    Parameters
    ----------
    ftp : ftplib.FTP
        Connected FTP instance.
    path : str
        Current directory path to list.
    base_path : str
        Base path for calculating relative paths.

    Returns
    -------
    list[GEOFileInfo]
        List of file information objects.
    """
    files = []

    try:
        ftp.cwd(path)
    except ftplib.error_perm as e:
        logger.warning("Cannot access directory %s: %s", path, e)
        return files

    # Get directory listing with MLSD (machine-readable)
    try:
        entries = list(ftp.mlsd())
    except ftplib.error_perm:
        # Fall back to NLST if MLSD is not supported
        try:
            names = ftp.nlst()
            for name in names:
                if name in (".", ".."):
                    continue
                full_path = f"{path}/{name}"
                # Try to get size, if fails assume directory
                try:
                    size = ftp.size(full_path)
                    if size is not None:
                        rel_path = full_path[len(base_path) :].lstrip("/")
                        files.append(
                            GEOFileInfo(
                                filename=name,
                                path=rel_path,
                                size_bytes=size,
                            )
                        )
                except ftplib.error_perm:
                    # Likely a directory, list it
                    subfiles = _list_ftp_files(ftp, full_path, base_path)
                    files.extend(subfiles)
            return files
        except ftplib.error_perm as e:
            logger.warning("Cannot list directory %s: %s", path, e)
            return files

    for name, facts in entries:
        if name in (".", ".."):
            continue

        full_path = f"{path}/{name}"
        entry_type = facts.get("type", "").lower()

        if entry_type == "file":
            size = int(facts.get("size", 0))
            rel_path = full_path[len(base_path) :].lstrip("/")
            files.append(
                GEOFileInfo(
                    filename=name,
                    path=rel_path,
                    size_bytes=size,
                )
            )
        elif entry_type == "dir":
            # List subdirectory (for suppl/ and other subdirs)
            subfiles = _list_ftp_files(ftp, full_path, base_path)
            files.extend(subfiles)

    return files


def get_gse_file_sizes(
    gse_id: str,
    timeout: float = 30.0,
) -> GEOFileSizeResult:
    """Get file names and sizes for a GEO dataset without downloading.

    Connects to the NCBI FTP server and lists all files in the dataset
    directory, returning file information including sizes. This allows
    checking dataset size before committing to a download.

    Parameters
    ----------
    gse_id : str
        GEO Series identifier (e.g., "GSE12345").
    timeout : float, optional
        Maximum time to wait for FTP operations in seconds, by default 30.0.

    Returns
    -------
    GEOFileSizeResult
        Result with file information and total size.

    Examples
    --------
    >>> result = get_gse_file_sizes("GSE12345")
    >>> print(f"Total size: {result.total_size_bytes / 1024**3:.2f} GB")
    >>> for f in result.files:
    ...     print(f"{f.filename}: {f.size_bytes / 1024**2:.2f} MB")
    """
    # Validate identifier
    try:
        gse_id = _validate_gse_id(gse_id)
    except ValueError as e:
        return GEOFileSizeResult(gse_id=gse_id, error=str(e))

    result = GEOFileSizeResult(gse_id=gse_id)

    # Try to fetch metadata from E-utilities
    try:
        metadata = _get_gse_metadata(gse_id, timeout)
        result.title = metadata.get("title")
    except httpx.HTTPStatusError as e:
        logger.debug("Could not fetch E-utilities metadata for %s: %s", gse_id, e)
    except Exception as e:
        logger.debug("Could not fetch E-utilities metadata for %s: %s", gse_id, e)

    # Connect to FTP and list files
    try:
        ftp = ftplib.FTP(GEO_FTP_HOST, timeout=timeout)
        ftp.login()  # Anonymous login
    except Exception as e:
        logger.error("Failed to connect to GEO FTP: %s", e)
        result.error = f"Failed to connect to FTP server: {e}"
        return result

    # Get the FTP directory path
    base_path = _get_ftp_directory(gse_id)

    # Check if directory exists
    try:
        ftp.cwd(base_path)
        ftp.cwd("/")
    except ftplib.error_perm:
        logger.error("Dataset %s not found on FTP server at %s", gse_id, base_path)
        result.error = f"Dataset {gse_id} not found on FTP server"
        ftp.quit()
        return result

    logger.info("Listing files for %s from FTP path %s", gse_id, base_path)

    try:
        files = _list_ftp_files(ftp, base_path, base_path)
    except Exception as e:
        logger.error("Failed to list files for %s: %s", gse_id, e)
        result.error = f"Failed to list files: {e}"
        ftp.quit()
        return result

    ftp.quit()

    if not files:
        logger.warning("No files found for dataset %s", gse_id)
        result.error = "No files found in dataset"
        return result

    # Sort by path for consistent output
    files.sort(key=lambda x: x.path)

    total_size = sum(f.size_bytes for f in files)
    result.files = files
    result.total_size_bytes = total_size
    result.file_count = len(files)

    logger.info(
        "Dataset %s: %d files, %.2f GB total",
        gse_id,
        len(files),
        total_size / 1024**3,
    )

    return result


def _download_ftp_file(
    ftp: ftplib.FTP,
    remote_path: str,
    local_path: Path,
    size: int,
    silent: bool,
) -> int:
    """Download a single file from FTP.

    Parameters
    ----------
    ftp : ftplib.FTP
        Connected FTP instance.
    remote_path : str
        Remote file path on FTP server.
    local_path : Path
        Local destination path.
    size : int
        Expected file size for progress bar.
    silent : bool
        If True, hide progress bar.

    Returns
    -------
    int
        Number of bytes downloaded.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_downloaded = 0

    with open(local_path, "wb") as f:
        if silent or size == 0:

            def callback(data: bytes) -> None:
                nonlocal bytes_downloaded
                f.write(data)
                bytes_downloaded += len(data)

            ftp.retrbinary(f"RETR {remote_path}", callback)
        else:
            with tqdm(
                total=size,
                unit="B",
                unit_scale=True,
                desc=local_path.name,
                leave=False,
            ) as pbar:

                def callback_with_progress(data: bytes) -> None:
                    nonlocal bytes_downloaded
                    f.write(data)
                    bytes_downloaded += len(data)
                    pbar.update(len(data))

                ftp.retrbinary(f"RETR {remote_path}", callback_with_progress)

    return bytes_downloaded


def download_gse_dataset(
    gse_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = False,
    timeout: float = 30.0,
) -> GEODownloadResult:
    """Download all files from a GEO dataset.

    Parameters
    ----------
    gse_id : str
        GEO Series identifier (e.g., "GSE12345").
    output_dir : str or Path
        Directory to save downloaded files. Files will be stored in a
        subdirectory named after the GSE ID, preserving the original
        directory structure.
    force : bool, optional
        If True, re-download files even if they already exist, by default False.
    silent : bool, optional
        If True, hide download progress bars, by default False.
    timeout : float, optional
        Maximum time to wait for server responses, by default 30.0.

    Returns
    -------
    GEODownloadResult
        Result with information about the download.

    Examples
    --------
    >>> result = download_gse_dataset("GSE12345", "/data/geo")
    >>> print(f"Downloaded {len(result.downloaded_files)} files")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate identifier
    try:
        gse_id = _validate_gse_id(gse_id)
    except ValueError as e:
        return GEODownloadResult(gse_id=gse_id, error=str(e))

    result = GEODownloadResult(gse_id=gse_id)

    # Try to fetch metadata from E-utilities
    try:
        metadata = _get_gse_metadata(gse_id, timeout)
        result.title = metadata.get("title")
        result.summary = metadata.get("summary")
    except httpx.HTTPStatusError as e:
        logger.debug("Could not fetch E-utilities metadata for %s: %s", gse_id, e)
    except Exception as e:
        logger.debug("Could not fetch E-utilities metadata for %s: %s", gse_id, e)

    # Get file list
    file_result = get_gse_file_sizes(gse_id, timeout)
    if file_result.error:
        result.error = file_result.error
        return result

    if not file_result.files:
        result.error = "No files found in dataset"
        return result

    # Update title if we got it from file listing
    if file_result.title and not result.title:
        result.title = file_result.title

    logger.info(
        "Downloading %d files for dataset %s: %s",
        len(file_result.files),
        gse_id,
        result.title or "(no title)",
    )

    # Create output directory
    dataset_dir = output_dir / gse_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Connect to FTP for download
    try:
        ftp = ftplib.FTP(GEO_FTP_HOST, timeout=timeout)
        ftp.login()  # Anonymous login
    except Exception as e:
        logger.error("Failed to connect to GEO FTP: %s", e)
        result.error = f"Failed to connect to FTP server: {e}"
        return result

    # Get the FTP directory path
    base_path = _get_ftp_directory(gse_id)

    try:
        for file_info in file_result.files:
            remote_path = f"{base_path}/{file_info.path}"
            local_path = dataset_dir / file_info.path

            # Skip if file exists and not forcing
            if local_path.exists() and not force:
                existing_size = local_path.stat().st_size
                if existing_size == file_info.size_bytes:
                    logger.debug("Skipping existing file: %s", file_info.path)
                    result.skipped_files.append(file_info.path)
                    continue
                else:
                    logger.info(
                        "Re-downloading %s (size mismatch: %d vs %d)",
                        file_info.path,
                        existing_size,
                        file_info.size_bytes,
                    )

            try:
                _download_ftp_file(
                    ftp,
                    remote_path,
                    local_path,
                    file_info.size_bytes,
                    silent,
                )
                result.downloaded_files.append(local_path)
                logger.debug("Downloaded: %s", file_info.path)
            except Exception as e:
                logger.warning("Failed to download %s: %s", file_info.path, e)
                result.failed_files.append(file_info.path)
                # Clean up partial download
                if local_path.exists():
                    local_path.unlink()

    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    logger.info(
        "Download complete for %s: %d downloaded, %d skipped, %d failed",
        gse_id,
        len(result.downloaded_files),
        len(result.skipped_files),
        len(result.failed_files),
    )

    return result
