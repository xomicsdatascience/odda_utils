# Directory analysis and file classification for omics datasets.
#
# This module examines directory contents and classifies files into categories
# based on their role in omics data analysis workflows. Uses a three-pass
# approach: heuristics, shallow LLM (filename-based), and deep LLM (content-based).
# Supports inspection of archive contents (zip, tar.gz, tar, tgz) with files
# represented as "archive.zip/internal/path/file.txt".

import io
import json
import logging
import os
import sqlite3
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from odda_utils import llm

from odda_utils.database import (
    get_dataset,
    get_article,
    get_article_by_pmid,
    get_article_by_pmcid,
    init_db,
    insert_dataset_file,
)
from odda_utils.utils import AzureCredentialsError, get_azure_credentials

logger = logging.getLogger(__name__)

# System prompts for LLM-based file classification.
_CLASSIFY_SHALLOW_SYSTEM_PROMPT = (
    "You are a scientific data classification assistant. Classify files from "
    "omics datasets into categories. Return results as JSON."
)
_CLASSIFY_DEEP_SYSTEM_PROMPT = (
    "You are a scientific data classification assistant. Classify files from "
    "omics datasets into categories."
)

# File classification categories for omics datasets
FileCategory = Literal[
    "raw_data",
    "instrument_parameters",
    "quantitative_data",
    "summary",
    "processing_parameters",
    "unknown",
]

# =============================================================================
# Heuristic Classification Rules
# =============================================================================

# Raw data file extensions (mass spec, sequencing, etc.)
RAW_DATA_EXTENSIONS = {
    # Mass spectrometry
    ".raw",  # Thermo raw files
    ".mzml",  # Standard format
    ".mzxml",  # Standard format
    ".wiff",  # AB SCIEX
    ".wiff2",  # AB SCIEX newer format
    ".d",  # Agilent/Bruker
    ".mgf",  # Mascot generic format
    ".mzid",  # mzIdentML
    # Sequencing
    ".fastq",  # Sequencing reads
    ".fq",  # Sequencing reads (alias)
    ".fastq.gz",  # Compressed FASTQ
    ".fq.gz",  # Compressed FASTQ
    ".bam",  # Alignment files
    ".sam",  # Alignment files
    ".cram",  # Compressed alignment
    ".bcf",  # Variant files
    ".vcf",  # Variant files
    ".vcf.gz",  # Compressed VCF
    # Other raw formats
    ".sra",  # SRA archive
    ".ab1",  # Sanger sequencing
}

# Instrument parameter file extensions
INSTRUMENT_PARAMETERS_EXTENSIONS = {
    ".method",  # Thermo instrument methods
    ".m",  # Agilent MassHunter methods
    ".wiff.scan",  # AB SCIEX scan settings
    ".instmethod",  # Instrument method files
    ".acqmethod",  # Acquisition method files
    ".tune",  # Instrument tune files
    ".cal",  # Calibration files
    ".meth",  # Method files (generic)
    ".prm",  # Parameter files (instrument)
    ".acq",  # Acquisition files
    ".qmethod",  # Waters method files
    ".exp",  # Experiment setup files
}

# Quantitative data file extensions
QUANTITATIVE_DATA_EXTENSIONS = {
    ".xlsx",  # Excel spreadsheets
    ".xls",  # Excel spreadsheets (legacy)
    ".csv",  # Tabular data
    ".tsv",  # Tabular data
    ".txt",  # Text data (needs context)
    ".parquet",  # Columnar data
    ".h5",  # HDF5 data files
    ".hdf5",  # HDF5 data files
}

# Summary file extensions (documents, figures)
SUMMARY_EXTENSIONS = {
    ".pdf",  # Documents
    ".doc",  # Word documents
    ".docx",  # Word documents
    ".pptx",  # Presentations
    ".ppt",  # Presentations
    ".png",  # Images/figures
    ".jpg",  # Images/figures
    ".jpeg",  # Images/figures
    ".tif",  # Images/figures
    ".tiff",  # Images/figures
    ".svg",  # Vector graphics
    ".eps",  # Vector graphics
    ".html",  # HTML reports
}

# Processing/analysis parameter file extensions
PROCESSING_PARAMETERS_EXTENSIONS = {
    ".params",  # Parameter files
    ".config",  # Configuration files
    ".ini",  # Configuration files
    ".cfg",  # Configuration files
    ".json",  # Configuration files (often)
    ".yaml",  # Configuration files
    ".yml",  # Configuration files
    ".xml",  # Configuration files (often)
    ".toml",  # Configuration files
    # Supporting reference files
    ".fasta",  # Reference sequences
    ".fa",  # Reference sequences
    ".sptxt",  # Spectral libraries
    ".msp",  # Spectral libraries
    ".blib",  # BiblioSpec spectral library
    ".dlib",  # DIA-NN spectral library
    ".pepxml",  # Search results
    ".protxml",  # Search results
    ".pep.xml",  # Search results
    ".prot.xml",  # Search results
}

# Filename patterns indicating specific categories
RAW_DATA_PATTERNS = [
    "raw",
    "_raw",
    "instrument",
    ".d/",  # Agilent .d directories
]

INSTRUMENT_PARAMETERS_PATTERNS = [
    "method",
    "acquisition",
    "tune",
    "calibration",
    "acqmethod",
    "instmethod",
    "_settings",
    "instrument_",
]

QUANTITATIVE_DATA_PATTERNS = [
    "abundance",
    "quantification",
    "expression",
    "counts",
    "intensity",
    "ratio",
    "protein",
    "peptide",
    "gene",
    "transcript",
    "metabolite",
    "peak_area",
    "normalized",
    "matrix",
    "quant",
    "fpkm",
    "tpm",
    "rpkm",
    "lfq",
]

SUMMARY_PATTERNS = [
    "supplementary",
    "figure",
    "table",
    "result",
    "summary",
    "report",
    "visualization",
]

PROCESSING_PARAMETERS_PATTERNS = [
    "parameters",
    "config",
    "library",
    "reference",
    "database",
    "fasta",
    "spectral",
    "search_",
    "_search",
    "pipeline",
    "workflow",
    "analysis_",
]

# Supported archive extensions
ARCHIVE_EXTENSIONS = {".zip", ".tar.gz", ".tar", ".tgz"}


# =============================================================================
# Archive Inspection Utilities
# =============================================================================


@dataclass
class ArchiveFileInfo:
    """Information about a file inside an archive."""

    internal_path: str
    size_bytes: int
    is_archive: bool = False


def is_supported_archive(filename: str) -> bool:
    """Check if a file is a supported archive format.

    Parameters
    ----------
    filename : str
        The filename to check.

    Returns
    -------
    bool
        True if the file has a supported archive extension.
    """
    name_lower = filename.lower()
    for ext in ARCHIVE_EXTENSIONS:
        if name_lower.endswith(ext):
            return True
    return False


def get_archive_type(filename: str) -> str | None:
    """Get the archive type for a filename.

    Parameters
    ----------
    filename : str
        The filename to check.

    Returns
    -------
    str | None
        The archive type ('zip', 'tar.gz', 'tar', 'tgz') or None if not an archive.
    """
    name_lower = filename.lower()
    if name_lower.endswith(".zip"):
        return "zip"
    elif name_lower.endswith(".tar.gz"):
        return "tar.gz"
    elif name_lower.endswith(".tgz"):
        return "tgz"
    elif name_lower.endswith(".tar"):
        return "tar"
    return None


def list_archive_contents(archive_path: Path) -> list[ArchiveFileInfo]:
    """List the contents of an archive without extracting.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file.

    Returns
    -------
    list[ArchiveFileInfo]
        List of files in the archive.
    """
    archive_type = get_archive_type(str(archive_path))
    if archive_type is None:
        return []

    try:
        if archive_type == "zip":
            return _list_zip_contents(archive_path)
        else:
            return _list_tar_contents(archive_path, archive_type)
    except Exception as e:
        logger.warning("Failed to list contents of %s: %s", archive_path, e)
        return []


def _list_zip_contents(archive_path: Path) -> list[ArchiveFileInfo]:
    """List contents of a ZIP archive."""
    files = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            files.append(
                ArchiveFileInfo(
                    internal_path=info.filename,
                    size_bytes=info.file_size,
                    is_archive=is_supported_archive(info.filename),
                )
            )
    return files


def _list_tar_contents(archive_path: Path, archive_type: str) -> list[ArchiveFileInfo]:
    """List contents of a tar archive."""
    files = []
    mode = "r:gz" if archive_type in ("tar.gz", "tgz") else "r"

    with tarfile.open(archive_path, mode) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            files.append(
                ArchiveFileInfo(
                    internal_path=member.name,
                    size_bytes=member.size,
                    is_archive=is_supported_archive(member.name),
                )
            )
    return files


def extract_file_from_archive(
    archive_path: Path,
    internal_path: str,
) -> tuple[bytes | None, str | None]:
    """Extract a single file from an archive without fully extracting.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file.
    internal_path : str
        Path to the file inside the archive.

    Returns
    -------
    tuple[bytes | None, str | None]
        Tuple of (file_data, error). If successful, file_data contains the
        extracted bytes and error is None.
    """
    archive_type = get_archive_type(str(archive_path))
    if archive_type is None:
        return None, f"Unsupported archive format: {archive_path}"

    try:
        if archive_type == "zip":
            return _extract_from_zip(archive_path, internal_path)
        else:
            return _extract_from_tar(archive_path, internal_path, archive_type)
    except Exception as e:
        return None, f"Failed to extract file: {e}"


def _extract_from_zip(
    archive_path: Path, internal_path: str
) -> tuple[bytes | None, str | None]:
    """Extract a file from a ZIP archive."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        try:
            data = zf.read(internal_path)
            return data, None
        except KeyError:
            return None, f"File not found in archive: {internal_path}"


def _extract_from_tar(
    archive_path: Path,
    internal_path: str,
    archive_type: str,
) -> tuple[bytes | None, str | None]:
    """Extract a file from a tar archive."""
    mode = "r:gz" if archive_type in ("tar.gz", "tgz") else "r"

    with tarfile.open(archive_path, mode) as tf:
        try:
            member = tf.getmember(internal_path)
            f = tf.extractfile(member)
            if f is None:
                return None, f"Could not extract file: {internal_path}"
            data = f.read()
            return data, None
        except KeyError:
            return None, f"File not found in archive: {internal_path}"


def get_file_header_from_archive(
    archive_path: Path,
    internal_path: str,
    max_bytes: int = 8192,
) -> str:
    """Extract file header from a file inside an archive.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file.
    internal_path : str
        Path to the file inside the archive.
    max_bytes : int
        Maximum bytes to read.

    Returns
    -------
    str
        File header content as text.
    """
    data, error = extract_file_from_archive(archive_path, internal_path)
    if error or data is None:
        return f"[Could not read file from archive: {error}]"

    # Limit data size
    data = data[:max_bytes]

    # Try to decode as text
    try:
        content = data.decode("utf-8", errors="replace")
        lines = content.split("\n")[:50]
        return "\n".join(lines)
    except Exception:
        return f"[Binary file, first bytes: {data[:64].hex()}]"


@dataclass
class FileClassification:
    """Classification result for a single file."""

    filename: str
    category: FileCategory
    reason: str
    method: Literal["heuristic", "shallow_llm", "llm"]
    model: str | None = None


@dataclass
class DirectoryAnalysisResult:
    """Result of analyzing a directory."""

    directory_path: str
    classifications: list[FileClassification] = field(default_factory=list)
    total_files: int = 0
    heuristic_classified: int = 0
    shallow_llm_classified: int = 0
    llm_classified: int = 0
    unknown: int = 0
    error: str | None = None


def classify_file_by_heuristics(filename: str) -> tuple[FileCategory, str] | None:
    """Classify a file based on extension and filename patterns.

    Parameters
    ----------
    filename : str
        The filename to classify (can include path components).

    Returns
    -------
    tuple[FileCategory, str] | None
        A tuple of (category, reason) if classification succeeded,
        or None if the file cannot be confidently classified by heuristics.
    """
    name_lower = filename.lower()
    basename = Path(filename).name.lower()

    # Get all extensions (handles .tar.gz, .fastq.gz, etc.)
    suffixes = Path(basename).suffixes
    ext = "".join(suffixes).lower() if suffixes else ""
    single_ext = suffixes[-1].lower() if suffixes else ""

    # Check for raw data files (high priority)
    if ext in RAW_DATA_EXTENSIONS or single_ext in RAW_DATA_EXTENSIONS:
        return ("raw_data", f"Extension {ext or single_ext} indicates raw instrument data")

    # Check for .d directories (Agilent/Bruker raw data)
    if ".d/" in name_lower or name_lower.endswith(".d"):
        return ("raw_data", "Agilent/Bruker .d directory indicates raw instrument data")

    # Check for instrument parameter files
    if ext in INSTRUMENT_PARAMETERS_EXTENSIONS or single_ext in INSTRUMENT_PARAMETERS_EXTENSIONS:
        return (
            "instrument_parameters",
            f"Extension {ext or single_ext} indicates instrument parameters",
        )

    # Check for instrument parameter patterns in filename
    for pattern in INSTRUMENT_PARAMETERS_PATTERNS:
        if pattern in name_lower:
            if single_ext not in (QUANTITATIVE_DATA_EXTENSIONS | SUMMARY_EXTENSIONS):
                return (
                    "instrument_parameters",
                    f"Filename contains '{pattern}' indicating instrument parameters",
                )

    # Check for summary files by extension
    if single_ext in SUMMARY_EXTENSIONS:
        return ("summary", f"Extension {single_ext} indicates summary/figure file")

    # Check for processing parameter files
    if single_ext in PROCESSING_PARAMETERS_EXTENSIONS:
        # Be careful with common extensions that could be quantitative data
        if single_ext in {".json", ".xml", ".yaml", ".yml"}:
            for pattern in PROCESSING_PARAMETERS_PATTERNS:
                if pattern in name_lower:
                    return (
                        "processing_parameters",
                        f"Extension {single_ext} with '{pattern}' indicates processing parameters",
                    )
        else:
            return (
                "processing_parameters",
                f"Extension {single_ext} indicates processing parameters",
            )

    # Check for quantitative data by extension + name patterns
    if single_ext in QUANTITATIVE_DATA_EXTENSIONS:
        for pattern in QUANTITATIVE_DATA_PATTERNS:
            if pattern in name_lower:
                return (
                    "quantitative_data",
                    f"Extension {single_ext} with '{pattern}' indicates quantitative data",
                )

        # Check for summary patterns in spreadsheets
        for pattern in SUMMARY_PATTERNS:
            if pattern in name_lower:
                return (
                    "summary",
                    f"Extension {single_ext} with '{pattern}' indicates summary data",
                )

    # Check for raw data patterns in any file
    for pattern in RAW_DATA_PATTERNS:
        if pattern in name_lower and single_ext not in SUMMARY_EXTENSIONS:
            return ("raw_data", f"Filename contains '{pattern}' indicating raw data")

    # Check for quantitative patterns
    for pattern in QUANTITATIVE_DATA_PATTERNS:
        if pattern in name_lower and single_ext in QUANTITATIVE_DATA_EXTENSIONS:
            return ("quantitative_data", f"Filename contains '{pattern}'")

    # Check for summary patterns
    for pattern in SUMMARY_PATTERNS:
        if pattern in name_lower:
            return ("summary", f"Filename contains '{pattern}'")

    # Cannot confidently classify
    return None


def get_file_header(file_path: Path, max_bytes: int = 8192) -> str:
    """Extract file header for deep LLM classification.

    Parameters
    ----------
    file_path : Path
        Path to the file.
    max_bytes : int
        Maximum bytes to read from the file.

    Returns
    -------
    str
        File header content as text, or error message if unreadable.
    """
    try:
        # Try to read as text first
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)
            # Limit to first N lines for readability
            lines = content.split("\n")[:50]
            return "\n".join(lines)
    except Exception:
        # Try binary inspection
        try:
            with open(file_path, "rb") as f:
                header = f.read(min(512, max_bytes))
                # Try to decode, show hex for binary
                try:
                    return header.decode("utf-8", errors="replace")
                except Exception:
                    return f"[Binary file, first bytes: {header[:64].hex()}]"
        except Exception as e:
            return f"[Could not read file: {e}]"


def _build_shallow_llm_prompt(
    filenames: list[str],
    article_abstract: str | None,
) -> str:
    """Build prompt for shallow LLM classification.

    Parameters
    ----------
    filenames : list[str]
        List of filenames to classify.
    article_abstract : str | None
        Article abstract for context.

    Returns
    -------
    str
        The prompt string.
    """
    filenames_formatted = "\n".join(f"- {fn}" for fn in filenames)

    prompt = f"""Classify the following files from an omics dataset into one of six categories:

1. **raw_data**: Raw instrument data from data collection
   - Mass spectrometry: .raw, .mzML, .wiff files
   - Sequencing: .fastq, .bam, .sam files
   - Any unprocessed data directly from instruments

2. **instrument_parameters**: Instrument acquisition parameters and settings
   - Mass spectrometer method files (.method, .m)
   - Acquisition parameters, tune files, calibration files
   - Files describing how instruments were configured

3. **quantitative_data**: Processed omic quantities derived from raw data
   - Protein abundances, gene expression levels, metabolite concentrations
   - Typically in spreadsheets (Excel, CSV) with numerical measurements
   - Contains processed measurements ready for analysis

4. **summary**: Summary tables, figures, and documents presenting findings
   - Supplementary PDFs, Word documents
   - Tables summarizing results, figures and visualizations

5. **processing_parameters**: Analysis tool settings and reference files
   - Configuration files for analysis software
   - Reference databases (FASTA files), spectral libraries
   - Pipeline parameters, search engine settings

6. **unknown**: Files that cannot be confidently classified

## Files to Classify

{filenames_formatted}

## Article Context

{article_abstract if article_abstract else "No article context available."}

## Instructions

For EACH file listed above, provide a classification. Return your answer as a JSON object with a "classifications" array containing objects with:
- "filename": the exact filename as given
- "category": one of "raw_data", "instrument_parameters", "quantitative_data", "summary", "processing_parameters", or "unknown"
- "reason": a brief explanation (under 15 words)

Example response:
{{"classifications": [
  {{"filename": "protein_abundances.xlsx", "category": "quantitative_data", "reason": "Excel file with protein abundance measurements"}},
  {{"filename": "method.m", "category": "instrument_parameters", "reason": "Agilent instrument method file"}}
]}}
"""
    return prompt


def _build_deep_llm_prompt(
    filename: str,
    file_header: str,
    article_abstract: str | None,
) -> str:
    """Build prompt for deep LLM classification with file content.

    Parameters
    ----------
    filename : str
        The filename.
    file_header : str
        Content preview from the file.
    article_abstract : str | None
        Article abstract for context.

    Returns
    -------
    str
        The prompt string.
    """
    prompt = f"""Classify this file from an omics dataset into one of six categories:

1. **raw_data**: Raw instrument data from data collection
2. **instrument_parameters**: Instrument acquisition parameters and settings
3. **quantitative_data**: Processed omic quantities (abundances, counts, intensities)
4. **summary**: Summary tables, figures, documents presenting findings
5. **processing_parameters**: Analysis tool settings and reference files
6. **unknown**: Cannot be confidently classified

## File Information

**Filename:** {filename}

**File Content Preview:**
```
{file_header[:4000]}
```

## Article Context

{article_abstract[:4000] if article_abstract else "No article context available."}

## Instructions

Based on the filename, content preview, and article context, classify this file.
Return your answer as JSON with two fields:
- "category": one of the six categories above
- "reason": a brief explanation (under 15 words)

Example response:
{{"category": "quantitative_data", "reason": "Contains protein abundance values with sample columns"}}
"""
    return prompt


def classify_files_shallow_llm(
    filenames: list[str],
    article_abstract: str | None,
    endpoint: str,
    api_key: str,
    model: str,
    api_version: str = "2024-02-01",
    batch_size: int = 50,
) -> list[FileClassification]:
    """Classify multiple files using LLM based on filenames only.

    Files are processed in batches to avoid overwhelming the LLM with
    too many files at once. Each batch makes a single LLM call.

    Parameters
    ----------
    filenames : list[str]
        List of filenames to classify.
    article_abstract : str | None
        Article abstract for context.
    endpoint : str
        Azure OpenAI endpoint URL.
    api_key : str
        Azure OpenAI API key.
    model : str
        Name of the model deployment.
    api_version : str
        Azure OpenAI API version.
    batch_size : int
        Maximum number of files to classify in a single LLM call.

    Returns
    -------
    list[FileClassification]
        List of classification results, one per input filename.
    """
    if not filenames:
        return []

    # Process in batches to avoid overwhelming the LLM
    all_classifications = []
    for i in range(0, len(filenames), batch_size):
        batch = filenames[i : i + batch_size]
        batch_results = _classify_batch_shallow_llm(
            filenames=batch,
            article_abstract=article_abstract,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            api_version=api_version,
        )
        all_classifications.extend(batch_results)

    return all_classifications


def _classify_batch_shallow_llm(
    filenames: list[str],
    article_abstract: str | None,
    endpoint: str,
    api_key: str,
    model: str,
    api_version: str = "2024-02-01",
) -> list[FileClassification]:
    """Classify a batch of files using LLM based on filenames only.

    Parameters
    ----------
    filenames : list[str]
        List of filenames to classify (should be <= batch_size).
    article_abstract : str | None
        Article abstract for context.
    endpoint : str
        Azure OpenAI endpoint URL.
    api_key : str
        Azure OpenAI API key.
    model : str
        Name of the model deployment.
    api_version : str
        Azure OpenAI API version.

    Returns
    -------
    list[FileClassification]
        List of classification results, one per input filename.
    """
    prompt = _build_shallow_llm_prompt(filenames, article_abstract)

    classifications = []

    try:
        completion = llm.complete_json(
            prompt,
            system=_CLASSIFY_SHALLOW_SYSTEM_PROMPT,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            api_version=api_version,
            max_tokens=10000,
        )
        model = completion.model or model
        result = completion.data
        if result is None:
            result = json.loads(completion.text)

        # Handle response format
        if isinstance(result, dict):
            items = result.get("classifications", [])
        elif isinstance(result, list):
            items = result
        else:
            items = []

        valid_categories = {
            "raw_data",
            "instrument_parameters",
            "quantitative_data",
            "summary",
            "processing_parameters",
            "unknown",
        }

        # Build map for lookup
        result_map = {}
        for item in items:
            if isinstance(item, dict):
                fn = item.get("filename", "")
                category = item.get("category", "unknown")
                reason = item.get("reason", "LLM classification")

                if category not in valid_categories:
                    category = "unknown"

                result_map[fn] = FileClassification(
                    filename=fn,
                    category=category,
                    reason=reason[:100] if reason else "LLM classification",
                    method="shallow_llm",
                    model=model,
                )

        # Return in same order as input
        for fn in filenames:
            if fn in result_map:
                classifications.append(result_map[fn])
            else:
                classifications.append(
                    FileClassification(
                        filename=fn,
                        category="unknown",
                        reason="File not classified by LLM",
                        method="shallow_llm",
                        model=model,
                    )
                )

    except Exception as e:
        logger.warning("Shallow LLM classification failed: %s", e)
        for fn in filenames:
            classifications.append(
                FileClassification(
                    filename=fn,
                    category="unknown",
                    reason=f"LLM classification failed: {e}",
                    method="shallow_llm",
                    model=model,
                )
            )

    return classifications


def classify_file_deep_llm(
    file_path: Path,
    filename: str,
    article_abstract: str | None,
    endpoint: str,
    api_key: str,
    model: str,
    api_version: str = "2024-02-01",
) -> FileClassification:
    """Classify a file using LLM with file content examination.

    Parameters
    ----------
    file_path : Path
        Path to the file.
    filename : str
        Display filename.
    article_abstract : str | None
        Article abstract for context.
    endpoint : str
        Azure OpenAI endpoint URL.
    api_key : str
        Azure OpenAI API key.
    model : str
        Name of the model deployment.
    api_version : str
        Azure OpenAI API version.

    Returns
    -------
    FileClassification
        Classification result.
    """
    file_header = get_file_header(file_path)
    prompt = _build_deep_llm_prompt(filename, file_header, article_abstract)

    try:
        completion = llm.complete_json(
            prompt,
            system=_CLASSIFY_DEEP_SYSTEM_PROMPT,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            api_version=api_version,
            max_tokens=1024,
        )
        model = completion.model or model
        result = completion.data
        if result is None:
            result = json.loads(completion.text)

        category = result.get("category", "unknown")
        reason = result.get("reason", "LLM classification")

        valid_categories = {
            "raw_data",
            "instrument_parameters",
            "quantitative_data",
            "summary",
            "processing_parameters",
            "unknown",
        }

        if category not in valid_categories:
            category = "unknown"

        return FileClassification(
            filename=filename,
            category=category,
            reason=reason[:100] if reason else "LLM classification",
            method="llm",
            model=model,
        )

    except Exception as e:
        logger.warning("Deep LLM classification failed for %s: %s", filename, e)
        return FileClassification(
            filename=filename,
            category="unknown",
            reason=f"LLM classification failed: {e}",
            method="llm",
            model=model,
        )


def _classify_file_deep_llm_with_header(
    filename: str,
    file_header: str,
    article_abstract: str | None,
    endpoint: str,
    api_key: str,
    model: str,
    api_version: str = "2024-02-01",
) -> FileClassification:
    """Classify a file using LLM with a pre-extracted file header.

    This is used when the file header has already been extracted (e.g., from
    an archive), so we don't need to read the file directly.

    Parameters
    ----------
    filename : str
        Display filename.
    file_header : str
        Pre-extracted content preview from the file.
    article_abstract : str | None
        Article abstract for context.
    endpoint : str
        Azure OpenAI endpoint URL.
    api_key : str
        Azure OpenAI API key.
    model : str
        Name of the model deployment.
    api_version : str
        Azure OpenAI API version.

    Returns
    -------
    FileClassification
        Classification result.
    """
    prompt = _build_deep_llm_prompt(filename, file_header, article_abstract)

    try:
        completion = llm.complete_json(
            prompt,
            system=_CLASSIFY_DEEP_SYSTEM_PROMPT,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            api_version=api_version,
            max_tokens=1024,
        )
        model = completion.model or model
        result = completion.data
        if result is None:
            result = json.loads(completion.text)

        category = result.get("category", "unknown")
        reason = result.get("reason", "LLM classification")

        valid_categories = {
            "raw_data",
            "instrument_parameters",
            "quantitative_data",
            "summary",
            "processing_parameters",
            "unknown",
        }

        if category not in valid_categories:
            category = "unknown"

        return FileClassification(
            filename=filename,
            category=category,
            reason=reason[:100] if reason else "LLM classification",
            method="llm",
            model=model,
        )

    except Exception as e:
        logger.warning("Deep LLM classification failed for %s: %s", filename, e)
        return FileClassification(
            filename=filename,
            category="unknown",
            reason=f"LLM classification failed: {e}",
            method="llm",
            model=model,
        )


def analyze_directory(
    directory_path: str | Path,
    dataset_id: str | None = None,
    db_path: str | Path | None = None,
    use_shallow_llm: bool = True,
    use_deep_llm: bool = False,
    inspect_archives: bool = True,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    store_results: bool = True,
) -> DirectoryAnalysisResult:
    """Analyze directory contents and classify files.

    Uses a three-pass approach:
    1. Heuristics: Classify files by extension and filename patterns
    2. Shallow LLM: For unknown files, use LLM with filenames + article abstract
    3. Deep LLM: When enabled, examine file headers for remaining unknowns

    When inspect_archives is True, files inside archives (zip, tar.gz, etc.)
    are also classified. Their paths are represented as "archive.zip/internal/path.txt".

    Parameters
    ----------
    directory_path : str | Path
        Path to the directory to analyze.
    dataset_id : str, optional
        Dataset identifier for database linking and article context.
    db_path : str | Path, optional
        Path to the SQLite database. If provided with dataset_id, will
        fetch article abstract for context and store results.
    use_shallow_llm : bool
        Whether to use shallow LLM for files not classified by heuristics.
    use_deep_llm : bool
        Whether to use deep LLM (with file content) for remaining unknowns.
    inspect_archives : bool
        Whether to inspect contents of archive files (zip, tar.gz, etc.).
    llm_model : str
        Name of the Azure OpenAI model deployment.
    endpoint_file : str | Path, optional
        Path to file containing Azure OpenAI endpoint.
    api_key_file : str | Path, optional
        Path to file containing Azure OpenAI API key.
    store_results : bool
        Whether to store classification results in the database.

    Returns
    -------
    DirectoryAnalysisResult
        Result containing all file classifications and statistics.
    """
    directory_path = Path(directory_path)
    result = DirectoryAnalysisResult(directory_path=str(directory_path))

    if not directory_path.exists():
        result.error = f"Directory not found: {directory_path}"
        return result

    if not directory_path.is_dir():
        result.error = f"Path is not a directory: {directory_path}"
        return result

    # Resolve credentials if LLM is enabled. Azure OpenAI credentials are treated
    # as optional legacy hints; the actual chat provider is resolved by
    # odda_utils.llm. LLM passes are only disabled if no provider is configured.
    endpoint = None
    api_key = None
    llm_available = False
    if use_shallow_llm or use_deep_llm:
        try:
            endpoint, api_key = get_azure_credentials(endpoint_file, api_key_file)
        except AzureCredentialsError:
            endpoint, api_key = None, None
        try:
            llm.resolve_chat_config(endpoint=endpoint, api_key=api_key)
            llm_available = True
        except llm.ModelConfigError as e:
            logger.warning("LLM classification disabled: %s", e)
            use_shallow_llm = False
            use_deep_llm = False

    # Get article abstract for context
    article_abstract = None
    conn = None
    if db_path and dataset_id:
        conn = init_db(db_path)
        dataset = get_dataset(conn, dataset_id)
        if dataset:
            # Try to get article abstract using dataset's article links
            article = None
            if dataset["doi"]:
                article = get_article(conn, dataset["doi"])
            elif dataset["pmid"]:
                article = get_article_by_pmid(conn, dataset["pmid"])
            elif dataset["pmcid"]:
                article = get_article_by_pmcid(conn, dataset["pmcid"])

            if article and article["abstract"]:
                article_abstract = article["abstract"]

    # Collect all files in directory (including archive contents)
    # Each entry is (file_path, rel_path, archive_path, internal_path, size_bytes)
    # For regular files: archive_path and internal_path are None
    # For archived files: archive_path is the archive, internal_path is path within
    all_files: list[tuple[Path, str, Path | None, str | None, int | None]] = []

    for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path = Path(root) / filename
            rel_path = str(file_path.relative_to(directory_path))

            # Get file size
            try:
                size_bytes = file_path.stat().st_size
            except Exception:
                size_bytes = None

            all_files.append((file_path, rel_path, None, None, size_bytes))

            # Check if this is an archive and inspect its contents
            if inspect_archives and is_supported_archive(filename):
                archive_contents = list_archive_contents(file_path)
                for archive_file in archive_contents:
                    # Create virtual path: archive.zip/internal/path.txt
                    virtual_path = f"{rel_path}/{archive_file.internal_path}"
                    all_files.append(
                        (
                            file_path,  # The archive file path
                            virtual_path,
                            file_path,  # Archive path
                            archive_file.internal_path,
                            archive_file.size_bytes,
                        )
                    )

    result.total_files = len(all_files)

    # First pass: heuristic classification
    heuristic_classifications = {}
    unknown_files = []

    for file_path, rel_path, archive_path, internal_path, size_bytes in all_files:
        heuristic_result = classify_file_by_heuristics(rel_path)

        if heuristic_result is not None:
            category, reason = heuristic_result
            classification = FileClassification(
                filename=rel_path,
                category=category,
                reason=reason,
                method="heuristic",
            )
            heuristic_classifications[rel_path] = classification
            result.heuristic_classified += 1
        else:
            unknown_files.append((file_path, rel_path, archive_path, internal_path, size_bytes))

    # Second pass: shallow LLM classification
    shallow_llm_classifications = {}
    still_unknown_files = []

    if unknown_files and use_shallow_llm and llm_available:
        unknown_filenames = [rel_path for _, rel_path, _, _, _ in unknown_files]
        llm_results = classify_files_shallow_llm(
            filenames=unknown_filenames,
            article_abstract=article_abstract,
            endpoint=endpoint,
            api_key=api_key,
            model=llm_model,
        )

        for file_info, classification in zip(unknown_files, llm_results):
            file_path, rel_path, archive_path, internal_path, size_bytes = file_info
            if classification.category != "unknown":
                shallow_llm_classifications[rel_path] = classification
                result.shallow_llm_classified += 1
            else:
                still_unknown_files.append(file_info)
    else:
        still_unknown_files = unknown_files

    # Third pass: deep LLM classification (when enabled)
    deep_llm_classifications = {}

    if still_unknown_files and use_deep_llm and llm_available:
        for file_path, rel_path, archive_path, internal_path, size_bytes in still_unknown_files:
            # Get file header - either from archive or directly
            if archive_path is not None and internal_path is not None:
                file_header = get_file_header_from_archive(archive_path, internal_path)
            else:
                file_header = get_file_header(file_path)

            classification = _classify_file_deep_llm_with_header(
                filename=rel_path,
                file_header=file_header,
                article_abstract=article_abstract,
                endpoint=endpoint,
                api_key=api_key,
                model=llm_model,
            )
            deep_llm_classifications[rel_path] = classification
            if classification.category != "unknown":
                result.llm_classified += 1
            else:
                result.unknown += 1
    else:
        # Mark remaining as unknown
        for _, rel_path, _, _, _ in still_unknown_files:
            if rel_path not in shallow_llm_classifications:
                deep_llm_classifications[rel_path] = FileClassification(
                    filename=rel_path,
                    category="unknown",
                    reason="Could not classify by heuristics or LLM",
                    method="heuristic",
                )
                result.unknown += 1

    # Combine results in original file order
    for file_path, rel_path, archive_path, internal_path, size_bytes in all_files:
        if rel_path in heuristic_classifications:
            classification = heuristic_classifications[rel_path]
        elif rel_path in shallow_llm_classifications:
            classification = shallow_llm_classifications[rel_path]
        elif rel_path in deep_llm_classifications:
            classification = deep_llm_classifications[rel_path]
        else:
            classification = FileClassification(
                filename=rel_path,
                category="unknown",
                reason="Classification error",
                method="heuristic",
            )
            result.unknown += 1

        result.classifications.append(classification)

        # Store in database if requested
        if store_results and conn and dataset_id:
            # Determine local path for storage
            if archive_path is not None:
                local_path = str(archive_path)
            else:
                local_path = str(file_path)

            insert_dataset_file(
                conn=conn,
                dataset_id=dataset_id,
                filename=rel_path,
                file_type=classification.category,
                file_type_reason=classification.reason,
                method=classification.method,
                model=classification.model,
                size_bytes=size_bytes,
                local_path=local_path,
            )

    if conn:
        conn.close()

    return result


def get_classification_summary(result: DirectoryAnalysisResult) -> dict:
    """Get a summary of classification results by category.

    Parameters
    ----------
    result : DirectoryAnalysisResult
        The analysis result.

    Returns
    -------
    dict
        Summary with counts and file lists by category.
    """
    summary = {
        "total_files": result.total_files,
        "by_method": {
            "heuristic": result.heuristic_classified,
            "shallow_llm": result.shallow_llm_classified,
            "llm": result.llm_classified,
            "unknown": result.unknown,
        },
        "by_category": {},
    }

    for classification in result.classifications:
        category = classification.category
        if category not in summary["by_category"]:
            summary["by_category"][category] = {"count": 0, "files": []}
        summary["by_category"][category]["count"] += 1
        summary["by_category"][category]["files"].append(
            {
                "filename": classification.filename,
                "reason": classification.reason,
                "method": classification.method,
            }
        )

    return summary
