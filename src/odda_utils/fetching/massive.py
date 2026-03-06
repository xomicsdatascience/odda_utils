"""MassIVE dataset fetching via FTP and PROXI API.

This module provides functionality to download proteomics datasets from MassIVE
(Mass spectrometry Interactive Virtual Environment), a mass spectrometry data
repository at UCSD. Datasets are identified by MSV IDs (e.g., MSV000092832).
"""

import ftplib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)

MASSIVE_FTP_HOST = "massive-ftp.ucsd.edu"
MASSIVE_PROXI_API = "https://massive.ucsd.edu/ProteoSAFe/proxi/v0.1/datasets"


@dataclass
class MSVFileInfo:
    """Information about a single file in a MassIVE dataset."""

    filename: str
    path: str
    size_bytes: int


@dataclass
class MSVFileSizeResult:
    """Result of querying file sizes for a MassIVE dataset."""

    msv_id: str
    title: str | None = None
    files: list[MSVFileInfo] = field(default_factory=list)
    total_size_bytes: int = 0
    file_count: int = 0
    error: str | None = None


@dataclass
class MSVDownloadResult:
    """Result of a MassIVE dataset download."""

    msv_id: str
    title: str | None = None
    doi: str | None = None
    downloaded_files: list[Path] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class MSVMetadata:
    """Metadata for a MassIVE dataset from the PROXI API.

    Mirrors the PXDMetadata structure for consistency across dataset types.
    """

    dataset_id: str
    title: str | None = None
    description: str | None = None
    species: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    repository: str = "MassIVE"
    submission_date: str | None = None
    publication_date: str | None = None
    linked_doi: str | None = None
    linked_pmid: str | None = None
    pxd_id: str | None = None
    error: str | None = None


def _validate_msv_id(msv_id: str) -> str:
    """Validate and normalize MassIVE identifier.

    Parameters
    ----------
    msv_id : str
        MassIVE identifier (e.g., "MSV000092832").

    Returns
    -------
    str
        Normalized identifier.

    Raises
    ------
    ValueError
        If identifier format is invalid.
    """
    msv_id = msv_id.strip().upper()
    if not re.match(r"^MSV\d{9}$", msv_id):
        raise ValueError(
            f"Invalid MassIVE ID format: {msv_id}. "
            "Expected format: MSV followed by 9 digits (e.g., MSV000092832)"
        )
    return msv_id


def _find_dataset_path(ftp: ftplib.FTP, msv_id: str) -> str | None:
    """Find the FTP path for a dataset by searching all version directories.

    MassIVE FTP distributes datasets across multiple directories (v01-v12,
    x01, z01). This function searches each directory to find where the
    dataset is located.

    Parameters
    ----------
    ftp : ftplib.FTP
        Connected FTP instance.
    msv_id : str
        MassIVE dataset identifier.

    Returns
    -------
    str or None
        FTP path if found, None otherwise.
    """
    # List all directories at the root
    ftp.cwd("/")
    try:
        root_entries = list(ftp.mlsd())
    except ftplib.error_perm:
        # Fall back to nlst
        root_entries = [(name, {"type": "dir"}) for name in ftp.nlst()]

    # Search in each version directory
    for name, facts in root_entries:
        if name in (".", ".."):
            continue
        # Only search in directories that look like version folders
        if not (name.startswith("v") or name.startswith("x") or name.startswith("z")):
            continue

        try:
            path = f"/{name}/{msv_id}"
            ftp.cwd(path)
            # If we get here, the directory exists
            ftp.cwd("/")  # Go back to root
            return path
        except ftplib.error_perm:
            continue

    return None


def _get_project_metadata(msv_id: str, timeout: float) -> dict:
    """Fetch project metadata from MassIVE PROXI API.

    Parameters
    ----------
    msv_id : str
        MassIVE dataset identifier.
    timeout : float
        Request timeout in seconds.

    Returns
    -------
    dict
        Project metadata dictionary.
    """
    url = f"{MASSIVE_PROXI_API}/{msv_id}"

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def fetch_msv_metadata(
    msv_id: str,
    timeout: float = 30.0,
) -> MSVMetadata:
    """Fetch dataset metadata from MassIVE PROXI API.

    Retrieves project metadata including title, description, species, instruments,
    submission/publication dates, and linked publications.

    Parameters
    ----------
    msv_id : str
        MassIVE dataset identifier (e.g., "MSV000092832").
    timeout : float, optional
        Request timeout in seconds, by default 30.0.

    Returns
    -------
    MSVMetadata
        Dataset metadata from MassIVE.

    Examples
    --------
    >>> metadata = fetch_msv_metadata("MSV000092832")
    >>> print(metadata.title)
    >>> print(metadata.species)
    """
    # Validate and normalize MSV ID
    try:
        msv_id = _validate_msv_id(msv_id)
    except ValueError as e:
        return MSVMetadata(dataset_id=msv_id, error=str(e))

    result = MSVMetadata(dataset_id=msv_id)

    try:
        data = _get_project_metadata(msv_id, timeout)

        result.title = data.get("title")
        result.description = data.get("summary")

        # Extract species from species list
        # PROXI API returns species as a nested list of cvParam dicts
        species_list = data.get("species", [])
        if species_list:
            extracted_species = []
            # Flatten if nested (species can be [[{...}, {...}], ...])
            flat_species = []
            for item in species_list:
                if isinstance(item, list):
                    flat_species.extend(item)
                else:
                    flat_species.append(item)

            for s in flat_species:
                if isinstance(s, dict):
                    # Look for taxonomy: scientific name
                    if s.get("name") == "taxonomy: scientific name":
                        val = s.get("value")
                        if val and val not in extracted_species:
                            extracted_species.append(val)
                elif isinstance(s, str) and s not in extracted_species:
                    extracted_species.append(s)
            result.species = extracted_species

        # Extract instruments
        instruments_list = data.get("instruments", [])
        if instruments_list:
            result.instruments = [
                i.get("name", i.get("accession", str(i)))
                for i in instruments_list
                if isinstance(i, dict)
            ]
            if not result.instruments and instruments_list:
                result.instruments = [str(i) for i in instruments_list if i]

        # Extract PXD identifier if this dataset is also on ProteomeXchange
        identifiers = data.get("identifiers", [])
        for identifier in identifiers:
            if isinstance(identifier, dict):
                acc = identifier.get("accession", "")
                if acc.startswith("PXD"):
                    result.pxd_id = acc
                    break

        # Extract publication links
        publications = data.get("publications", [])
        for pub in publications:
            if isinstance(pub, dict):
                # Get PMID
                pmid = pub.get("pubmedId") or pub.get("pmid")
                if pmid and not result.linked_pmid:
                    result.linked_pmid = str(pmid)
                # Get DOI
                doi = pub.get("doi")
                if doi and not result.linked_doi:
                    result.linked_doi = doi

        # Extract dates - PROXI API may have dates in various formats
        # Try common field names
        submission_date = data.get("submissionDate") or data.get("submission_date")
        if submission_date:
            # Try to parse and normalize date
            if isinstance(submission_date, str):
                result.submission_date = submission_date[:10]  # Take just YYYY-MM-DD

        publication_date = data.get("publicationDate") or data.get("publication_date")
        if publication_date:
            if isinstance(publication_date, str):
                result.publication_date = publication_date[:10]

    except httpx.HTTPStatusError as e:
        logger.error("Failed to fetch MassIVE metadata for %s: %s", msv_id, e)
        result.error = f"Failed to fetch metadata: {e}"
    except Exception as e:
        logger.error("Error fetching MassIVE metadata for %s: %s", msv_id, e)
        result.error = f"Error fetching metadata: {e}"

    return result


def _list_ftp_files_recursive(
    ftp: ftplib.FTP,
    path: str,
    base_path: str,
) -> list[MSVFileInfo]:
    """Recursively list all files in an FTP directory.

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
    list[MSVFileInfo]
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
                            MSVFileInfo(
                                filename=name,
                                path=rel_path,
                                size_bytes=size,
                            )
                        )
                except ftplib.error_perm:
                    # Likely a directory, recurse
                    files.extend(_list_ftp_files_recursive(ftp, full_path, base_path))
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
                MSVFileInfo(
                    filename=name,
                    path=rel_path,
                    size_bytes=size,
                )
            )
        elif entry_type == "dir":
            # Recurse into subdirectory
            files.extend(_list_ftp_files_recursive(ftp, full_path, base_path))

    return files


def get_msv_file_sizes(
    msv_id: str,
    timeout: float = 30.0,
) -> MSVFileSizeResult:
    """Get file names and sizes for a MassIVE dataset without downloading.

    Connects to the MassIVE FTP server and lists all files in the dataset
    directory, returning file information including sizes. This allows
    checking dataset size before committing to a download.

    Parameters
    ----------
    msv_id : str
        MassIVE dataset identifier (e.g., "MSV000092832").
    timeout : float, optional
        Maximum time to wait for FTP operations in seconds, by default 30.0.

    Returns
    -------
    MSVFileSizeResult
        Result with file information and total size.

    Examples
    --------
    >>> result = get_msv_file_sizes("MSV000092832")
    >>> print(f"Total size: {result.total_size_bytes / 1024**3:.2f} GB")
    >>> for f in result.files:
    ...     print(f"{f.filename}: {f.size_bytes / 1024**2:.2f} MB")
    """
    # Validate identifier
    try:
        msv_id = _validate_msv_id(msv_id)
    except ValueError as e:
        return MSVFileSizeResult(msv_id=msv_id, error=str(e))

    result = MSVFileSizeResult(msv_id=msv_id)

    # Try to fetch metadata from PROXI API
    try:
        metadata = _get_project_metadata(msv_id, timeout)
        result.title = metadata.get("title")
    except httpx.HTTPStatusError as e:
        logger.debug("Could not fetch PROXI metadata for %s: %s", msv_id, e)
    except Exception as e:
        logger.debug("Could not fetch PROXI metadata for %s: %s", msv_id, e)

    # Connect to FTP and list files
    try:
        ftp = ftplib.FTP(MASSIVE_FTP_HOST, timeout=timeout)
        ftp.login()  # Anonymous login
    except Exception as e:
        logger.error("Failed to connect to MassIVE FTP: %s", e)
        result.error = f"Failed to connect to FTP server: {e}"
        return result

    # Find the dataset path
    base_path = _find_dataset_path(ftp, msv_id)
    if base_path is None:
        logger.error("Dataset %s not found on FTP server", msv_id)
        result.error = f"Dataset {msv_id} not found on FTP server"
        ftp.quit()
        return result

    logger.info("Listing files for %s from FTP path %s", msv_id, base_path)

    try:
        files = _list_ftp_files_recursive(ftp, base_path, base_path)
    except Exception as e:
        logger.error("Failed to list files for %s: %s", msv_id, e)
        result.error = f"Failed to list files: {e}"
        ftp.quit()
        return result

    ftp.quit()

    if not files:
        logger.warning("No files found for dataset %s", msv_id)
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
        msv_id,
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


def download_msv_dataset(
    msv_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = False,
    timeout: float = 30.0,
) -> MSVDownloadResult:
    """Download all files from a MassIVE dataset.

    Parameters
    ----------
    msv_id : str
        MassIVE dataset identifier (e.g., "MSV000092832").
    output_dir : str or Path
        Directory to save downloaded files. Files will be stored in a
        subdirectory named after the MSV ID, preserving the original
        directory structure.
    force : bool, optional
        If True, re-download files even if they already exist, by default False.
    silent : bool, optional
        If True, hide download progress bars, by default False.
    timeout : float, optional
        Maximum time to wait for server responses, by default 30.0.

    Returns
    -------
    MSVDownloadResult
        Result with information about the download.

    Examples
    --------
    >>> result = download_msv_dataset("MSV000092832", "/data/proteomics")
    >>> print(f"Downloaded {len(result.downloaded_files)} files")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate identifier
    try:
        msv_id = _validate_msv_id(msv_id)
    except ValueError as e:
        return MSVDownloadResult(msv_id=msv_id, error=str(e))

    result = MSVDownloadResult(msv_id=msv_id)

    # Try to fetch metadata from PROXI API
    try:
        metadata = _get_project_metadata(msv_id, timeout)
        result.title = metadata.get("title")
        # DOI might be in contacts or other fields
        contacts = metadata.get("contacts", [])
        for contact in contacts:
            if contact.get("contactType") == "dataset_submitter":
                result.doi = contact.get("affiliation")
                break
    except httpx.HTTPStatusError as e:
        logger.debug("Could not fetch PROXI metadata for %s: %s", msv_id, e)
    except Exception as e:
        logger.debug("Could not fetch PROXI metadata for %s: %s", msv_id, e)

    # Get file list
    file_result = get_msv_file_sizes(msv_id, timeout)
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
        msv_id,
        result.title or "(no title)",
    )

    # Create output directory
    dataset_dir = output_dir / msv_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Connect to FTP for download
    try:
        ftp = ftplib.FTP(MASSIVE_FTP_HOST, timeout=timeout)
        ftp.login()  # Anonymous login
    except Exception as e:
        logger.error("Failed to connect to MassIVE FTP: %s", e)
        result.error = f"Failed to connect to FTP server: {e}"
        return result

    # Find the dataset path
    base_path = _find_dataset_path(ftp, msv_id)
    if base_path is None:
        logger.error("Dataset %s not found on FTP server", msv_id)
        result.error = f"Dataset {msv_id} not found on FTP server"
        ftp.quit()
        return result

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
        msv_id,
        len(result.downloaded_files),
        len(result.skipped_files),
        len(result.failed_files),
    )

    return result
