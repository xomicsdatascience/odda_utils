"""Gene Expression Omnibus (GEO) dataset fetching."""

import logging
import re
import gzip
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

@dataclass
class GSEDownloadResult:
    """Result of a GSE dataset download."""

    gse_id: str
    downloaded_files: list[Path] = field(default_factory=list)
    error: str | None = None


def download_gse_dataset(
    gse_id: str,
    output_dir: str | Path,
    force: bool = False,
    timeout: float = 30.0,
) -> GSEDownloadResult:
    """Download SOFT format file from NCBI GEO for a given GSE ID.

    Args:
        gse_id: GEO Series identifier (e.g., "GSE12345").
        output_dir: Directory to save downloaded files.
        force: If True, re-download files even if they already exist.
        timeout: Maximum time to wait for server responses.

    Returns:
        GSEDownloadResult with information about the download.
    """
    raise NotImplementedError("GEO dataset fetching is not working.")
    if not re.match(r"^GSE\d+$", gse_id, re.IGNORECASE):
        return GSEDownloadResult(gse_id=gse_id, error=f"Invalid GSE ID format: {gse_id}")

    gse_id = gse_id.upper()
    dataset_dir = Path(output_dir) / gse_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    result = GSEDownloadResult(gse_id=gse_id)

    # SOFT file URL construction
    # Format: https://ftp.ncbi.nlm.nih.gov/geo/series/GSExxx/GSEnnn/soft/GSEnnn_family.soft.gz
    # where GSExxx is GSE followed by the ID truncated to the thousands.
    nnn = gse_id[3:]
    if len(nnn) <= 3:
        nnn_prefix = "GSEnnn"
    else:
        nnn_prefix = f"GSE{nnn[:-3]}nnn"

    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{nnn_prefix}/{gse_id}/soft/{gse_id}_family.soft.gz"
    
    dest_path_gz = dataset_dir / f"{gse_id}_family.soft.gz"
    dest_path_soft = dataset_dir / f"{gse_id}_family.soft"

    if dest_path_soft.exists() and not force:
        logger.info("File %s already exists, skipping download.", dest_path_soft)
        result.downloaded_files.append(dest_path_soft)
        # Even if series file exists, we should try to fetch samples (they might be missing)
        sample_ids = _parse_sample_ids(dest_path_soft)
        if sample_ids:
            for gsm_id in sample_ids:
                gsm_path = _download_gsm_file(gsm_id, dataset_dir, force=force, timeout=timeout)
                if gsm_path:
                    result.downloaded_files.append(gsm_path)
        return result

    logger.info("Downloading %s from %s", gse_id, url)

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 404:
            # Try without prefix if it fails? No, the prefix is mandatory for GEO FTP.
            # Some very old GSEs might have different structures, but this is the standard.
            result.error = f"Dataset {gse_id} not found on GEO FTP (404)."
            return result
        
        response.raise_for_status()

        with open(dest_path_gz, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Decompress
        with gzip.open(dest_path_gz, "rb") as f_in:
            with open(dest_path_soft, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file
        dest_path_gz.unlink()
        
        result.downloaded_files.append(dest_path_soft)
        logger.info("Successfully downloaded and decompressed %s", dest_path_soft)

        # Fetch samples
        sample_ids = _parse_sample_ids(dest_path_soft)
        if sample_ids:
            logger.info("Found %d samples in %s. Downloading...", len(sample_ids), gse_id)
            for gsm_id in sample_ids:
                gsm_path = _download_gsm_file(gsm_id, dataset_dir, force=force, timeout=timeout)
                if gsm_path:
                    result.downloaded_files.append(gsm_path)

    except requests.exceptions.RequestException as e:
        logger.error("Failed to download %s: %s", gse_id, e)
        result.error = f"Download failed: {str(e)}"
    except Exception as e:
        logger.error("Error processing %s: %s", gse_id, e)
        result.error = f"Processing error: {str(e)}"

    return result


def _parse_sample_ids(soft_file_path: Path) -> list[str]:
    """Parse SOFT file to extract sample IDs (!Series_sample_id)."""
    sample_ids = []
    try:
        with open(soft_file_path) as f:
            for line in f:
                if line.startswith("!Series_sample_id"):
                    # Format: !Series_sample_id = GSM12345
                    match = re.search(r"=\s*(GSM\d+)", line)
                    if match:
                        sample_ids.append(match.group(1))
    except Exception as e:
        logger.error("Error parsing SOFT file %s: %s", soft_file_path, e)
    return sample_ids


def _download_gsm_file(
    gsm_id: str,
    output_dir: Path,
    force: bool = False,
    timeout: float = 30.0,
) -> Path | None:
    """Download a single GSM SOFT file."""
    # Format: https://ftp.ncbi.nlm.nih.gov/geo/samples/GSMxxx/GSMnnn/soft/GSMnnn.soft.gz
    # where GSMxxx is GSM followed by the ID truncated to the thousands.
    nnn = gsm_id[3:]
    if len(nnn) <= 3:
        nnn_prefix = "GSMnnn"
    else:
        nnn_prefix = f"GSM{nnn[:-3]}nnn"

    url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{nnn_prefix}/{gsm_id}/soft/{gsm_id}.soft.gz"
    dest_path_soft = output_dir / f"{gsm_id}.soft"
    dest_path_gz = output_dir / f"{gsm_id}.soft.gz"

    if dest_path_soft.exists() and not force:
        return dest_path_soft

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(dest_path_gz, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with gzip.open(dest_path_gz, "rb") as f_in:
            with open(dest_path_soft, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        dest_path_gz.unlink()
        return dest_path_soft
    except Exception as e:
        logger.error("Failed to download sample %s: %s", gsm_id, e)
        if dest_path_gz.exists():
            dest_path_gz.unlink()
        return None

