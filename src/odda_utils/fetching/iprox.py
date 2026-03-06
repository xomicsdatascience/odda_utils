"""iProX dataset fetching via REST API.

This module provides functionality to download proteomics datasets from iProX
(Integrated Proteome Resources), a member of the ProteomeXchange consortium.
Datasets are identified by IPX IDs (e.g., IPX0001234000).
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)

IPROX_API_BASE = "https://www.iprox.cn/"


@dataclass
class IPXDownloadResult:
    """Result of an iProX dataset download."""

    ipx_id: str
    title: str | None = None
    doi: str | None = None
    downloaded_files: list[Path] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class IPXMetadata:
    """Metadata for an iProX dataset from the iProX PROXI API.

    Mirrors the PXDMetadata structure for consistency across dataset types.
    """

    dataset_id: str
    title: str | None = None
    description: str | None = None
    species: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    repository: str = "iProX"
    submission_date: str | None = None
    publication_date: str | None = None
    linked_doi: str | None = None
    linked_pmid: str | None = None
    pxd_id: str | None = None
    keywords: list[str] = field(default_factory=list)
    error: str | None = None


def _validate_ipx_id(ipx_id: str) -> str:
    """Validate and normalize iProX identifier.

    Args:
        ipx_id: iProX identifier (e.g., "IPX0001234000").

    Returns:
        Normalized identifier.

    Raises:
        ValueError: If identifier format is invalid.
    """
    ipx_id = ipx_id.strip().upper()
    if not re.match(r"^IPX\d{10}$", ipx_id):
        raise ValueError(
            f"Invalid iProX ID format: {ipx_id}. "
            "Expected format: IPX followed by 10 digits (e.g., IPX0001234000)"
        )
    return ipx_id


def _get_project_metadata(ipx_id: str, timeout: float) -> dict:
    """Fetch project metadata from iProX API.

    Args:
        ipx_id: iProX dataset identifier.
        timeout: Request timeout in seconds.

    Returns:
        Project metadata dictionary.
    """
    url = f"{IPROX_API_BASE}proxi/rest/datasets/{ipx_id}"

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _get_file_list(ipx_id: str, timeout: float) -> list[dict]:
    """Fetch list of files for a dataset.

    Args:
        ipx_id: iProX dataset identifier.
        timeout: Request timeout in seconds.

    Returns:
        List of file metadata dictionaries with 'fileName' and 'downloadUrl' keys.
    """
    url = f"{IPROX_API_BASE}page/api/file.html"
    params = {"projectId": ipx_id}

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    # iProX returns files in a nested structure under "data"
    # Each item in data may have a "files" list or be a file itself
    files = []
    if isinstance(data, dict):
        data_list = data.get("data", [])
        if isinstance(data_list, list):
            for item in data_list:
                if isinstance(item, dict):
                    if "files" in item:
                        files.extend(item["files"])
                    elif "fileName" in item or "downloadUrl" in item:
                        files.append(item)
    elif isinstance(data, list):
        files = data

    return files


def fetch_ipx_metadata(
    ipx_id: str,
    timeout: float = 30.0,
) -> IPXMetadata:
    """Fetch dataset metadata for an iProX dataset.

    Note: iProX does not provide a public metadata API. This function validates
    the IPX ID format and returns an IPXMetadata object with empty metadata
    fields. The dataset can still be ingested into the database with the
    dataset_id and repository fields populated.

    Parameters
    ----------
    ipx_id : str
        iProX dataset identifier (e.g., "IPX0001234000").
    timeout : float, optional
        Request timeout in seconds (unused), by default 30.0.

    Returns
    -------
    IPXMetadata
        Dataset metadata object with dataset_id and repository populated.
        Other metadata fields will be empty since iProX does not provide
        a public metadata API.

    Examples
    --------
    >>> metadata = fetch_ipx_metadata("IPX0001234000")
    >>> print(metadata.dataset_id)  # IPX0001234000
    >>> print(metadata.repository)  # iProX
    """
    # Validate and normalize IPX ID
    try:
        ipx_id = _validate_ipx_id(ipx_id)
    except ValueError as e:
        return IPXMetadata(dataset_id=ipx_id, error=str(e))

    # Warn about lack of metadata API (both stdout and logger)
    warning_msg = (
        f"Warning: iProX does not provide a public metadata API. "
        f"Dataset {ipx_id} will be ingested with minimal metadata (dataset_id and repository only). "
        f"Metadata fields can be populated manually or from linked publications."
    )
    print(warning_msg)
    logger.warning(warning_msg)

    # Return result with only the ID and repository set
    # Other fields remain None/empty since iProX API requires authentication
    return IPXMetadata(dataset_id=ipx_id)


def _download_file(
    url: str,
    dest_path: Path,
    force: bool,
    silent: bool,
    timeout: float,
) -> bool:
    """Download a single file from iProX.

    Args:
        url: Download URL.
        dest_path: Destination file path.
        force: Re-download even if file exists.
        silent: Hide progress bar.
        timeout: Request timeout.

    Returns:
        True if download succeeded.

    Raises:
        httpx.HTTPError: If download fails.
    """
    if dest_path.exists() and not force:
        logger.debug("File already exists, skipping: %s", dest_path)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a longer timeout for actual file downloads
    download_timeout = httpx.Timeout(timeout, read=300.0)

    with httpx.Client(timeout=download_timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as f:
                if silent or total == 0:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                else:
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=dest_path.name,
                    ) as pbar:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
    return True


def download_ipx_dataset(
    ipx_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = False,
    timeout: float = 30.0,
) -> IPXDownloadResult:
    """Download all files from an iProX dataset.

    Args:
        ipx_id: iProX dataset identifier (e.g., "IPX0001234000").
        output_dir: Directory to save downloaded files. Files will be stored
            in a subdirectory named after the IPX ID.
        force: If True, re-download files even if they already exist.
        silent: If True, hide download progress bars.
        timeout: Maximum time to wait for server responses when fetching
            project metadata.

    Returns:
        IPXDownloadResult with information about the download.

    Example:
        >>> result = download_ipx_dataset("IPX0001234000", "/data/proteomics")
        >>> print(f"Downloaded {len(result.downloaded_files)} files")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate identifier
    try:
        ipx_id = _validate_ipx_id(ipx_id)
    except ValueError as e:
        return IPXDownloadResult(ipx_id=ipx_id, error=str(e))

    result = IPXDownloadResult(ipx_id=ipx_id)

    # Fetch project metadata
    try:
        metadata = _get_project_metadata(ipx_id, timeout)
        result.title = metadata.get("title")
        result.doi = metadata.get("doi")
    except httpx.HTTPStatusError as e:
        logger.error("Failed to fetch metadata for %s: %s", ipx_id, e)
        if e.response.status_code == 404:
            result.error = f"Project not found: {ipx_id}"
        else:
            result.error = f"Failed to fetch project metadata: HTTP {e.response.status_code}"
        return result
    except Exception as e:
        logger.error("Failed to fetch metadata for %s: %s", ipx_id, e)
        result.error = f"Failed to fetch project metadata: {e}"
        return result

    # Fetch file list
    try:
        files = _get_file_list(ipx_id, timeout)
    except Exception as e:
        logger.error("Failed to list files for %s: %s", ipx_id, e)
        result.error = f"Failed to list files: {e}"
        return result

    if not files:
        logger.warning("No files found for project %s", ipx_id)
        result.error = "No files found in project"
        return result

    logger.info(
        "Downloading %d files for project %s: %s",
        len(files),
        ipx_id,
        result.title,
    )

    # Create output directory
    dataset_dir = output_dir / ipx_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Download each file
    for file_info in files:
        filename = file_info.get("fileName") or file_info.get("name")
        download_url = file_info.get("downloadUrl") or file_info.get("downloadLink")

        if not filename or not download_url:
            logger.warning("Skipping file with missing name or URL: %s", file_info)
            continue

        dest_path = dataset_dir / filename

        try:
            _download_file(download_url, dest_path, force, silent, timeout)
            result.downloaded_files.append(dest_path)
            logger.debug("Downloaded: %s", filename)
        except Exception as e:
            logger.warning("Failed to download %s: %s", filename, e)
            result.failed_files.append(filename)

    logger.info(
        "Download complete for %s: %d succeeded, %d failed",
        ipx_id,
        len(result.downloaded_files),
        len(result.failed_files),
    )

    return result
