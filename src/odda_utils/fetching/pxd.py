"""ProteomeXchange dataset fetching and metadata retrieval.

This module provides functionality to download proteomics datasets from
ProteomeXchange, query dataset metadata from the PRIDE Archive API, and
retrieve file sizes before downloading. Also includes direct URL download
for datasets hosted on repositories not supported by ppx (e.g., iProX).

File classification for cataloged files uses the shared classification
functions from odda_utils.ingestion.analyze_directory to maintain
consistency across all file classification in the codebase.

The catalog functions support nested archive inspection, examining files
inside compressed archives (.zip, .tar.gz, .tar, .tgz) and creating
database entries with paths like 'archive_name/internal_path'.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

import ppx
import requests
from tqdm import tqdm

from odda_utils.ingestion.analyze_directory import (
    ArchiveFileInfo,
    FileCategory,
    classify_file_by_heuristics,
    classify_files_shallow_llm,
    extract_file_from_archive,
    get_file_header_from_archive,
    is_supported_archive,
    list_archive_contents,
)

# Maximum depth for nested archive inspection (to prevent infinite recursion)
MAX_ARCHIVE_DEPTH = 3

logger = logging.getLogger(__name__)

PROTEOMEXCHANGE_API = "http://proteomecentral.proteomexchange.org/cgi/GetDataset"


@dataclass
class PXDFileInfo:
    """Information about a single file in a ProteomeXchange dataset."""

    filename: str
    url: str
    size_bytes: int


@dataclass
class PXDFileSizeResult:
    """Result of querying file sizes for a ProteomeXchange dataset."""

    pxd_id: str
    title: str | None = None
    files: list[PXDFileInfo] = field(default_factory=list)
    total_size_bytes: int = 0
    file_count: int = 0
    error: str | None = None


@dataclass
class PXDDownloadResult:
    """Result of a ProteomeXchange dataset download."""

    pxd_id: str
    title: str | None = None
    doi: str | None = None
    downloaded_files: list[Path] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    error: str | None = None


def _get_file_size(url: str, timeout: float) -> tuple[str, int]:
    """Get file size via HTTP HEAD request or FTP SIZE command.

    Args:
        url: URL of the file (supports http, https, and ftp protocols).
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (url, size_bytes). Size is 0 if request fails.
    """
    if url.startswith("ftp://"):
        return _get_ftp_file_size(url, timeout)

    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            return url, int(response.headers.get("content-length", 0))
    except requests.RequestException as e:
        logger.debug("Failed to get size for %s: %s", url, e)
    return url, 0


def _get_ftp_file_size(url: str, timeout: float) -> tuple[str, int]:
    """Get file size via FTP SIZE command.

    Args:
        url: FTP URL of the file.
        timeout: Connection timeout in seconds.

    Returns:
        Tuple of (url, size_bytes). Size is 0 if request fails.
    """
    import ftplib
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        path = parsed.path

        if not host:
            return url, 0

        ftp = ftplib.FTP(timeout=timeout)
        ftp.connect(host)
        ftp.login()  # Anonymous login
        size = ftp.size(path)
        ftp.quit()

        return url, size if size is not None else 0
    except Exception as e:
        logger.debug("Failed to get FTP size for %s: %s", url, e)
    return url, 0


def get_pxd_file_sizes(
    pxd_id: str,
    timeout: float = 30.0,
    max_workers: int = 5,
) -> PXDFileSizeResult:
    """Get file names and sizes for a ProteomeXchange dataset without downloading.

    Queries the ProteomeXchange API for the dataset file list, then uses HTTP
    HEAD requests to determine file sizes. This allows checking dataset size
    before committing to a download.

    Args:
        pxd_id: ProteomeXchange dataset identifier (e.g., "PXD021040").
        timeout: Maximum time to wait for HTTP requests in seconds.
        max_workers: Number of parallel workers for fetching file sizes.

    Returns:
        PXDFileSizeResult with file information and total size.

    Example:
        >>> result = get_pxd_file_sizes("PXD021040")
        >>> print(f"Total size: {result.total_size_bytes / 1024**3:.2f} GB")
        >>> for f in result.files:
        ...     print(f"{f.filename}: {f.size_bytes / 1024**2:.2f} MB")
    """
    pxd_id = pxd_id.strip().upper()
    result = PXDFileSizeResult(pxd_id=pxd_id)

    # Fetch dataset metadata from ProteomeXchange
    try:
        params = {"ID": pxd_id, "outputMode": "JSON", "test": "no"}
        response = requests.get(PROTEOMEXCHANGE_API, params=params, timeout=timeout)
        response.raise_for_status()
        px_data = response.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch ProteomeXchange metadata for %s: %s", pxd_id, e)
        result.error = f"Failed to fetch dataset metadata: {e}"
        return result
    except ValueError as e:
        logger.error("Invalid JSON response for %s: %s", pxd_id, e)
        result.error = f"Invalid response from ProteomeXchange: {e}"
        return result

    result.title = px_data.get("title")

    # Extract file URLs (both HTTP and FTP protocols)
    file_urls = [
        f.get("value")
        for f in px_data.get("datasetFiles", [])
        if f.get("value") and f.get("value").startswith(("http", "ftp"))
    ]

    if not file_urls:
        logger.warning("No files found for dataset %s", pxd_id)
        result.error = "No files found in dataset"
        return result

    logger.info("Fetching sizes for %d files in %s", len(file_urls), pxd_id)

    # Fetch file sizes in parallel
    files = []
    total_size = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_get_file_size, url, timeout): url for url in file_urls
        }
        for future in as_completed(futures):
            url, size = future.result()
            filename = url.split("/")[-1]
            files.append(PXDFileInfo(filename=filename, url=url, size_bytes=size))
            total_size += size

    # Sort by filename for consistent output
    files.sort(key=lambda x: x.filename)

    result.files = files
    result.total_size_bytes = total_size
    result.file_count = len(files)

    logger.info(
        "Dataset %s: %d files, %.2f GB total",
        pxd_id,
        len(files),
        total_size / 1024**3,
    )

    return result


def download_pxd_dataset(
    pxd_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = False,
    timeout: float = 10.0,
) -> PXDDownloadResult:
    """Download all files from a ProteomeXchange dataset.

    Args:
        pxd_id: ProteomeXchange dataset identifier (e.g., "PXD021040").
        output_dir: Directory to save downloaded files. Files will be stored
            in a subdirectory named after the PXD ID.
        force: If True, re-download files even if they already exist.
        silent: If True, hide download progress bars.
        timeout: Maximum time to wait for server responses when fetching
            project metadata.

    Returns:
        PXDDownloadResult with information about the download.

    Example:
        >>> result = download_pxd_dataset("PXD021040", "/data/proteomics")
        >>> print(f"Downloaded {len(result.downloaded_files)} files")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = PXDDownloadResult(pxd_id=pxd_id)

    try:
        project = ppx.find_project(
            identifier=pxd_id,
            local=output_dir / pxd_id,
            timeout=timeout,
        )
    except Exception as e:
        logger.error("Failed to find project %s: %s", pxd_id, e)
        result.error = f"Failed to find project: {e}"
        return result

    result.title = project.title
    result.doi = project.doi

    try:
        remote_files = project.remote_files()
    except Exception as e:
        logger.error("Failed to list files for %s: %s", pxd_id, e)
        result.error = f"Failed to list remote files: {e}"
        return result

    if not remote_files:
        logger.warning("No files found for project %s", pxd_id)
        result.error = "No files found in project"
        return result

    logger.info(
        "Downloading %d files for project %s: %s",
        len(remote_files),
        pxd_id,
        result.title,
    )

    for filename in remote_files:
        try:
            downloaded_paths = project.download(
                files=filename,
                force_=force,
                silent=silent,
            )
            result.downloaded_files.extend(downloaded_paths)
            logger.debug("Downloaded: %s", filename)
        except Exception as e:
            logger.warning("Failed to download %s: %s", filename, e)
            result.failed_files.append(filename)

    logger.info(
        "Download complete for %s: %d succeeded, %d failed",
        pxd_id,
        len(result.downloaded_files),
        len(result.failed_files),
    )

    return result


class FileInfoDict(TypedDict):
    """Input file information for URL downloads."""

    filename: str
    url: str
    size_bytes: int


@dataclass
class URLDownloadResult:
    """Result of a direct URL download operation."""

    output_dir: str
    downloaded_files: list[Path] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    total_bytes_downloaded: int = 0
    error: str | None = None


def download_files_from_urls(
    files: list[FileInfoDict],
    output_dir: str | Path,
    force: bool = False,
    silent: bool = False,
    timeout: float = 30.0,
    chunk_size: int = 8192,
) -> URLDownloadResult:
    """Download files directly from URLs.

    This function downloads files from a list of URLs, useful for datasets
    hosted on repositories that aren't fully supported by other download tools
    (e.g., iProX datasets accessed via ProteomeXchange).

    Args:
        files: List of file info dictionaries, each containing:
            - filename: Name to save the file as
            - url: URL to download from
            - size_bytes: Expected file size (for progress bar, 0 if unknown)
        output_dir: Directory to save downloaded files.
        force: If True, re-download files even if they already exist.
        silent: If True, hide download progress bars.
        timeout: Timeout for HTTP requests in seconds.
        chunk_size: Size of chunks to download at a time.

    Returns:
        URLDownloadResult with information about the download.

    Example:
        >>> files = [
        ...     {"filename": "data.raw", "url": "http://example.com/data.raw", "size_bytes": 1000000},
        ... ]
        >>> result = download_files_from_urls(files, "/data/downloads")
        >>> print(f"Downloaded {len(result.downloaded_files)} files")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = URLDownloadResult(output_dir=str(output_path))

    if not files:
        logger.warning("No files provided for download")
        result.error = "No files provided"
        return result

    logger.info("Downloading %d files to %s", len(files), output_path)

    for file_info in files:
        filename = file_info["filename"]
        url = file_info["url"]
        expected_size = file_info.get("size_bytes", 0)
        dest_path = output_path / filename

        # Skip if file exists and force is False
        if dest_path.exists() and not force:
            # Check if existing file matches expected size
            existing_size = dest_path.stat().st_size
            if expected_size == 0 or existing_size == expected_size:
                logger.debug("Skipping existing file: %s", filename)
                result.skipped_files.append(filename)
                continue
            else:
                logger.info(
                    "Re-downloading %s (size mismatch: %d vs %d)",
                    filename,
                    existing_size,
                    expected_size,
                )

        # Download the file
        try:
            logger.info("Downloading: %s", filename)
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Get actual content length from response if available
            content_length = int(response.headers.get("content-length", expected_size))

            # Download with progress bar
            bytes_downloaded = 0
            with open(dest_path, "wb") as f:
                if silent:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                else:
                    with tqdm(
                        total=content_length,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=filename,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                bytes_downloaded += len(chunk)
                                pbar.update(len(chunk))

            result.downloaded_files.append(dest_path)
            result.total_bytes_downloaded += bytes_downloaded
            logger.debug("Downloaded: %s (%d bytes)", filename, bytes_downloaded)

        except requests.RequestException as e:
            logger.warning("Failed to download %s: %s", filename, e)
            result.failed_files.append(filename)
            # Clean up partial download
            if dest_path.exists():
                dest_path.unlink()

    logger.info(
        "Download complete: %d downloaded, %d skipped, %d failed",
        len(result.downloaded_files),
        len(result.skipped_files),
        len(result.failed_files),
    )

    return result


# PRIDE Archive REST API endpoints
PRIDE_API_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v2"
PRIDE_API_V3_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v3"


@dataclass
class PXDPublicationRef:
    """Publication reference linked to a dataset."""

    pubmed_id: str | None = None
    doi: str | None = None
    reference_line: str | None = None


@dataclass
class PXDMetadata:
    """Metadata for a ProteomeXchange dataset from PRIDE Archive."""

    dataset_id: str
    title: str | None = None
    description: str | None = None
    species: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    repository: str = "PRIDE"
    submission_date: str | None = None
    publication_date: str | None = None
    publications: list[PXDPublicationRef] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    project_tags: list[str] = field(default_factory=list)
    sample_processing: list[str] = field(default_factory=list)
    data_processing: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class PXDFileMetadata:
    """Metadata for a single file in a PRIDE dataset."""

    filename: str
    file_type: str | None = None
    file_type_reason: str | None = None
    method: Literal["heuristic", "shallow_llm", "llm"] | None = None
    model: str | None = None
    size_bytes: int = 0
    checksum: str | None = None
    checksum_type: str | None = None


@dataclass
class PXDFilesMetadataResult:
    """Result of fetching file metadata from PRIDE Archive."""

    dataset_id: str
    files: list[PXDFileMetadata] = field(default_factory=list)
    total_files: int = 0
    error: str | None = None


def _parse_pride_date(date_value: str | int | None) -> str | None:
    """Parse date from PRIDE API response.

    PRIDE API returns dates in various formats including Unix timestamps
    and ISO date strings.

    Parameters
    ----------
    date_value : str, int, or None
        Date value from API response.

    Returns
    -------
    str or None
        Date in YYYY-MM-DD format, or None if parsing fails.
    """
    import re
    from datetime import datetime

    if date_value is None:
        return None

    # Handle Unix timestamp (milliseconds)
    if isinstance(date_value, int):
        try:
            # PRIDE uses milliseconds
            dt = datetime.fromtimestamp(date_value / 1000)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, OSError):
            return None

    # Handle string date
    if isinstance(date_value, str):
        # Try to extract YYYY-MM-DD from various formats
        # Match ISO date format
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_value)
        if match:
            return date_value[:10]
        # Match date with slashes
        match = re.match(r"(\d{4})/(\d{2})/(\d{2})", date_value)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return None


def fetch_pxd_metadata(
    pxd_id: str,
    timeout: float = 30.0,
) -> PXDMetadata:
    """Fetch dataset metadata from PRIDE Archive API.

    Retrieves project metadata including title, description, species, instruments,
    submission/publication dates, and linked publications.

    Parameters
    ----------
    pxd_id : str
        ProteomeXchange dataset identifier (e.g., "PXD012345").
    timeout : float, optional
        Request timeout in seconds, by default 30.0.

    Returns
    -------
    PXDMetadata
        Dataset metadata from PRIDE Archive.

    Examples
    --------
    >>> metadata = fetch_pxd_metadata("PXD012345")
    >>> print(metadata.title)
    >>> print(metadata.species)
    """
    pxd_id = pxd_id.strip().upper()
    result = PXDMetadata(dataset_id=pxd_id)

    # Try v3 API first, fall back to v2 if needed
    url = f"{PRIDE_API_V3_BASE}/projects/{pxd_id}"

    try:
        response = requests.get(url, timeout=timeout)

        # Fall back to v2 API if v3 returns 404
        if response.status_code == 404:
            url = f"{PRIDE_API_BASE}/projects/{pxd_id}"
            response = requests.get(url, timeout=timeout)

        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error("Failed to fetch PRIDE metadata for %s: %s", pxd_id, e)
        result.error = f"Failed to fetch metadata: {e}"
        return result
    except ValueError as e:
        logger.error("Invalid JSON response for %s: %s", pxd_id, e)
        result.error = f"Invalid response from PRIDE: {e}"
        return result

    # Parse the response - handle both v2 and v3 response formats
    result.title = data.get("title")
    result.description = data.get("projectDescription") or data.get("description")

    # Parse species - can be in different formats
    organisms = data.get("organisms") or data.get("species") or []
    if isinstance(organisms, list):
        for org in organisms:
            if isinstance(org, dict):
                name = org.get("name") or org.get("scientificName")
                if name:
                    result.species.append(name)
            elif isinstance(org, str):
                result.species.append(org)

    # Parse instruments
    instruments = data.get("instruments") or data.get("instrumentNames") or []
    if isinstance(instruments, list):
        for inst in instruments:
            if isinstance(inst, dict):
                name = inst.get("name") or inst.get("cvLabel")
                if name:
                    result.instruments.append(name)
            elif isinstance(inst, str):
                result.instruments.append(inst)

    # Parse dates
    result.submission_date = _parse_pride_date(data.get("submissionDate"))
    result.publication_date = _parse_pride_date(data.get("publicationDate"))

    # Parse publications
    references = data.get("references") or data.get("publications") or []
    for ref in references:
        if isinstance(ref, dict):
            pub_ref = PXDPublicationRef(
                pubmed_id=str(ref.get("pubmedId")) if ref.get("pubmedId") else None,
                doi=ref.get("doi"),
                reference_line=ref.get("referenceLine"),
            )
            result.publications.append(pub_ref)

    # Parse keywords and tags
    result.keywords = data.get("keywords") or []
    result.project_tags = data.get("projectTags") or []

    # Parse sample and data processing
    sample_proc = data.get("sampleProcessingProtocol") or data.get("sampleProcessing")
    if sample_proc:
        if isinstance(sample_proc, list):
            result.sample_processing = sample_proc
        else:
            result.sample_processing = [sample_proc]

    data_proc = data.get("dataProcessingProtocol") or data.get("dataProcessing")
    if data_proc:
        if isinstance(data_proc, list):
            result.data_processing = data_proc
        else:
            result.data_processing = [data_proc]

    logger.info(
        "Fetched metadata for %s: %s (%d species, %d instruments)",
        pxd_id,
        result.title[:50] + "..." if result.title and len(result.title) > 50 else result.title,
        len(result.species),
        len(result.instruments),
    )

    return result


def fetch_pxd_files_metadata(
    pxd_id: str,
    timeout: float = 30.0,
    page_size: int = 100,
) -> PXDFilesMetadataResult:
    """Fetch file metadata from PRIDE Archive API.

    Retrieves information about files in a dataset including filenames,
    types, and sizes.

    Parameters
    ----------
    pxd_id : str
        ProteomeXchange dataset identifier (e.g., "PXD012345").
    timeout : float, optional
        Request timeout in seconds, by default 30.0.
    page_size : int, optional
        Number of files per API request, by default 100.

    Returns
    -------
    PXDFilesMetadataResult
        File metadata for the dataset.

    Examples
    --------
    >>> files_result = fetch_pxd_files_metadata("PXD012345")
    >>> for f in files_result.files:
    ...     print(f"{f.filename}: {f.size_bytes} bytes")
    """
    pxd_id = pxd_id.strip().upper()
    result = PXDFilesMetadataResult(dataset_id=pxd_id)

    # PRIDE Archive files endpoint
    url = f"{PRIDE_API_BASE}/files/byProject"

    all_files = []
    page = 0

    try:
        while True:
            params = {
                "accession": pxd_id,
                "pageSize": page_size,
                "page": page,
            }

            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            # Handle paginated response
            files_data = data if isinstance(data, list) else data.get("_embedded", {}).get("files", [])

            if not files_data:
                break

            for file_info in files_data:
                file_meta = PXDFileMetadata(
                    filename=file_info.get("fileName", ""),
                    file_type=file_info.get("fileCategory") or file_info.get("fileType"),
                    size_bytes=file_info.get("fileSizeBytes", 0),
                    checksum=file_info.get("checksum"),
                    checksum_type=file_info.get("checksumType"),
                )
                all_files.append(file_meta)

            # Check if there are more pages
            if len(files_data) < page_size:
                break

            page += 1

    except requests.RequestException as e:
        logger.error("Failed to fetch file metadata for %s: %s", pxd_id, e)
        result.error = f"Failed to fetch file metadata: {e}"
        return result
    except ValueError as e:
        logger.error("Invalid JSON response for file metadata %s: %s", pxd_id, e)
        result.error = f"Invalid response from PRIDE: {e}"
        return result

    result.files = all_files
    result.total_files = len(all_files)

    logger.info(
        "Fetched file metadata for %s: %d files",
        pxd_id,
        len(all_files),
    )

    return result


@dataclass
class CatalogResult:
    """Result of cataloging local dataset files."""

    files: list["PXDFileMetadata"] = field(default_factory=list)
    heuristic_classified: int = 0
    shallow_llm_classified: int = 0
    unknown_files: list[str] = field(default_factory=list)
    archive_files_cataloged: int = 0


def _catalog_archive_contents(
    archive_path: Path,
    base_filename: str,
) -> list[tuple[str, int]]:
    """List contents of an archive file for classification.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file on disk.
    base_filename : str
        The filename to use as prefix for internal paths (e.g., "results.zip").

    Returns
    -------
    list[tuple[str, int]]
        List of (virtual_path, size_bytes) for files inside the archive.
    """
    files = []

    contents = list_archive_contents(archive_path)
    if not contents:
        return files

    for file_info in contents:
        virtual_path = f"{base_filename}/{file_info.internal_path}"
        files.append((virtual_path, file_info.size_bytes))

    return files


def catalog_local_dataset_files(
    dataset_id: str,
    local_path: str | Path,
) -> list[PXDFileMetadata]:
    """Catalog files in a locally downloaded dataset directory.

    Scans a local directory for dataset files and returns their metadata.
    Uses the shared classification functions from analyze_directory module.

    For archive files (.zip, .tar.gz, .tar, .tgz), this function also lists
    the contents inside the archive and creates entries for each file using
    the path format: archive_name/internal_path.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (for logging purposes).
    local_path : str or Path
        Path to the local dataset directory.

    Returns
    -------
    list[PXDFileMetadata]
        List of file metadata for files in the directory and inside archives.
        Categories: raw_data, quantitative_data, summary, processing_parameters,
        instrument_parameters, unknown.

    Examples
    --------
    >>> files = catalog_local_dataset_files("PXD012345", "/data/datasets/PXD012345")
    >>> for f in files:
    ...     print(f"{f.filename}: {f.file_type} - {f.size_bytes} bytes")
    """
    local_path = Path(local_path)
    files = []

    if not local_path.exists():
        logger.warning("Local path does not exist: %s", local_path)
        return files

    if not local_path.is_dir():
        logger.warning("Local path is not a directory: %s", local_path)
        return files

    heuristic_count = 0
    unknown_count = 0
    archive_count = 0

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_path)
            filename = str(rel_path)

            classification_result = classify_file_by_heuristics(filename)

            if classification_result is not None:
                file_type, file_type_reason = classification_result
                heuristic_count += 1
            else:
                file_type = "unknown"
                file_type_reason = "Could not classify by heuristics"
                unknown_count += 1

            files.append(PXDFileMetadata(
                filename=filename,
                file_type=file_type,
                file_type_reason=file_type_reason,
                method="heuristic",
                model=None,
                size_bytes=file_path.stat().st_size,
            ))

            # If it's an archive, also catalog its contents
            if is_supported_archive(filename):
                archive_count += 1
                try:
                    archive_files = _catalog_archive_contents(
                        archive_path=file_path,
                        base_filename=filename,
                    )
                    for virtual_path, size_bytes in archive_files:
                        classification_result = classify_file_by_heuristics(virtual_path)
                        if classification_result is not None:
                            file_type, file_type_reason = classification_result
                            heuristic_count += 1
                        else:
                            file_type = "unknown"
                            file_type_reason = "Could not classify by heuristics"
                            unknown_count += 1

                        files.append(PXDFileMetadata(
                            filename=virtual_path,
                            file_type=file_type,
                            file_type_reason=file_type_reason,
                            method="heuristic",
                            model=None,
                            size_bytes=size_bytes,
                        ))
                except Exception as e:
                    logger.warning("Failed to catalog archive %s: %s", filename, e)

    logger.info(
        "Cataloged %d local files for %s in %s (%d heuristic, %d unknown, %d archives inspected)",
        len(files),
        dataset_id,
        local_path,
        heuristic_count,
        unknown_count,
        archive_count,
    )

    return files


def catalog_local_dataset_files_with_llm(
    dataset_id: str,
    local_path: str | Path,
    endpoint: str,
    api_key: str,
    llm_model: str = "gpt-5",
    article_abstract: str | None = None,
) -> CatalogResult:
    """Catalog files with LLM classification for ambiguous files.

    Similar to catalog_local_dataset_files but uses shallow LLM classification
    (batched, filename-based) for files that cannot be classified by heuristics.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier (for logging purposes).
    local_path : str or Path
        Path to the local dataset directory.
    endpoint : str
        Azure OpenAI endpoint URL.
    api_key : str
        Azure OpenAI API key.
    llm_model : str, optional
        Name of the LLM model deployment, by default "gpt-5".
    article_abstract : str, optional
        Article abstract for context in LLM classification.

    Returns
    -------
    CatalogResult
        Result containing file metadata, classification counts, and unknown files.
    """
    local_path = Path(local_path)
    result = CatalogResult()

    if not local_path.exists():
        logger.warning("Local path does not exist: %s", local_path)
        return result

    if not local_path.is_dir():
        logger.warning("Local path is not a directory: %s", local_path)
        return result

    # Collect all files (including archive contents)
    all_files: list[tuple[str, int]] = []  # (filename, size_bytes)

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_path)
            filename = str(rel_path)
            all_files.append((filename, file_path.stat().st_size))

            if is_supported_archive(filename):
                result.archive_files_cataloged += 1
                try:
                    archive_files = _catalog_archive_contents(
                        archive_path=file_path,
                        base_filename=filename,
                    )
                    all_files.extend(archive_files)
                except Exception as e:
                    logger.warning("Failed to catalog archive %s: %s", filename, e)

    # First pass: heuristic classification
    heuristic_results = {}
    unknown_files = []

    for filename, size_bytes in all_files:
        classification_result = classify_file_by_heuristics(filename)
        if classification_result is not None:
            file_type, file_type_reason = classification_result
            heuristic_results[filename] = (file_type, file_type_reason, size_bytes)
            result.heuristic_classified += 1
        else:
            unknown_files.append((filename, size_bytes))

    # Second pass: batch LLM classification for unknown files
    llm_results = {}
    if unknown_files:
        unknown_filenames = [fn for fn, _ in unknown_files]
        llm_classifications = classify_files_shallow_llm(
            filenames=unknown_filenames,
            article_abstract=article_abstract,
            endpoint=endpoint,
            api_key=api_key,
            model=llm_model,
        )
        for (filename, size_bytes), classification in zip(unknown_files, llm_classifications):
            llm_results[filename] = (
                classification.category,
                classification.reason,
                size_bytes,
                classification.model,
            )
            if classification.category != "unknown":
                result.shallow_llm_classified += 1
            else:
                result.unknown_files.append(filename)

    # Combine results
    for filename, size_bytes in all_files:
        if filename in heuristic_results:
            file_type, file_type_reason, size_bytes = heuristic_results[filename]
            result.files.append(PXDFileMetadata(
                filename=filename,
                file_type=file_type,
                file_type_reason=file_type_reason,
                method="heuristic",
                model=None,
                size_bytes=size_bytes,
            ))
        elif filename in llm_results:
            file_type, file_type_reason, size_bytes, model = llm_results[filename]
            result.files.append(PXDFileMetadata(
                filename=filename,
                file_type=file_type,
                file_type_reason=file_type_reason,
                method="shallow_llm",
                model=model,
                size_bytes=size_bytes,
            ))

    logger.info(
        "Cataloged %d local files for %s with LLM (%d heuristic, %d shallow_llm, %d unknown)",
        len(result.files),
        dataset_id,
        result.heuristic_classified,
        result.shallow_llm_classified,
        len(result.unknown_files),
    )

    return result
