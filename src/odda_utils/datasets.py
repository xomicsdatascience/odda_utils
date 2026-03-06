# Supplemental materials processing and classification module.
#
# This module provides tools for classifying supplemental files from scientific
# publications. Classification logic is delegated to the ingestion.analyze_directory
# module, which provides heuristic and LLM-based classification for omics data files.
#
# Categories: raw_data, quantitative_data, summary, processing_parameters,
# instrument_parameters, unknown.

import logging
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from odda_utils.database import (
    get_article,
    get_article_by_pmcid,
    get_article_by_pmid,
)
from odda_utils.ingestion.analyze_directory import (
    ArchiveFileInfo,
    DirectoryAnalysisResult,
    FileCategory,
    FileClassification,
    classify_file_by_heuristics,
    classify_file_deep_llm,
    classify_files_shallow_llm,
    extract_file_from_archive,
    get_file_header,
    get_file_header_from_archive,
    is_supported_archive,
    list_archive_contents,
)
from odda_utils.utils import AzureCredentialsError, get_azure_credentials

logger = logging.getLogger(__name__)

# Mapping from internal categories to database categories
# Internal: raw_data, instrument_parameters, quantitative_data, summary, processing_parameters, unknown
# Database: raw_data, quantitative_data, summary_data, supporting, unknown
CATEGORY_TO_DB_MAPPING = {
    "raw_data": "raw_data",
    "quantitative_data": "quantitative_data",
    "summary": "summary_data",
    "instrument_parameters": "supporting",
    "processing_parameters": "supporting",
    "unknown": "unknown",
}


# Re-export commonly used items for backward compatibility
__all__ = [
    "ArchiveFileInfo",
    "ArchiveExtractionResult",
    "FileCategory",
    "FileClassification",
    "SupplementalClassificationResult",
    "classify_article_supplementals",
    "classify_file_by_heuristics",
    "classify_files_shallow_llm",
    "extract_archive_to_temp",
    "extract_file_from_archive",
    "get_supplemental_classifications",
    "has_supplemental_classification",
    "is_supported_archive",
    "list_archive_contents",
]


@dataclass
class ArchiveExtractionResult:
    """Result of extracting an archive to a temporary directory."""

    temp_dir: Path
    files: list[Path]
    error: str | None = None

    def cleanup(self) -> None:
        """Remove the temporary directory and all extracted files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


@dataclass
class SupplementalClassificationResult:
    """Result of classifying all supplemental files for an article."""

    archive_path: str
    classifications: list[FileClassification] = field(default_factory=list)
    total_files: int = 0
    heuristic_classified: int = 0
    llm_classified: int = 0
    unknown: int = 0
    error: str | None = None


def extract_archive_to_temp(archive_path: str | Path) -> ArchiveExtractionResult:
    """Extract a .tar.gz archive to a temporary directory.

    Parameters
    ----------
    archive_path : str | Path
        Path to the .tar.gz archive file.

    Returns
    -------
    ArchiveExtractionResult
        Result containing the temp directory path and list of extracted files.
        Caller is responsible for calling cleanup() when done.
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        return ArchiveExtractionResult(
            temp_dir=Path(tempfile.gettempdir()),
            files=[],
            error=f"Archive not found: {archive_path}",
        )

    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="supplementals_"))
        files = []

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(temp_dir, filter="data")

            for member in tar.getmembers():
                if member.isfile():
                    files.append(temp_dir / member.name)

        return ArchiveExtractionResult(temp_dir=temp_dir, files=files)

    except tarfile.TarError as e:
        return ArchiveExtractionResult(
            temp_dir=Path(tempfile.gettempdir()),
            files=[],
            error=f"Failed to extract archive: {e}",
        )
    except Exception as e:
        return ArchiveExtractionResult(
            temp_dir=Path(tempfile.gettempdir()),
            files=[],
            error=f"Unexpected error: {e}",
        )


def classify_article_supplementals(
    archive_path: str | Path,
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    use_llm: bool = True,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    store_results: bool = True,
) -> SupplementalClassificationResult:
    """Classify all supplemental files for an article.

    This is the main entry point for supplemental classification. It:
    1. Extracts the archive to a temporary directory
    2. Classifies each file using heuristics
    3. For files that can't be confidently classified, uses LLM classification
    4. Optionally stores results in the database

    Parameters
    ----------
    archive_path : str | Path
        Path to the supplementals .tar.gz archive.
    conn : sqlite3.Connection
        Database connection.
    doi : str | None
        Article DOI.
    pmid : str | None
        Article PMID.
    pmcid : str | None
        Article PMCID.
    use_llm : bool
        Whether to use LLM for ambiguous files.
    llm_model : str
        Name of the LLM model deployment for LLM classification.
    endpoint_file : str | Path | None
        Path to file containing Azure OpenAI endpoint.
    api_key_file : str | Path | None
        Path to file containing Azure OpenAI API key.
    store_results : bool
        Whether to store classification results in the database.

    Returns
    -------
    SupplementalClassificationResult
        Result containing all file classifications and statistics.
    """
    archive_path = Path(archive_path)
    result = SupplementalClassificationResult(archive_path=str(archive_path))

    # Get Azure credentials if LLM is enabled
    endpoint = None
    api_key = None
    if use_llm:
        try:
            endpoint, api_key = get_azure_credentials(endpoint_file, api_key_file)
        except AzureCredentialsError as e:
            logger.warning("LLM classification disabled: %s", e)
            use_llm = False

    # Get article abstract for context
    article_abstract = None
    article = None
    if doi:
        article = get_article(conn, doi)
    elif pmid:
        article = get_article_by_pmid(conn, pmid)
    elif pmcid:
        article = get_article_by_pmcid(conn, pmcid)

    # sqlite3.Row does not support .get() method, use direct indexing with check
    if article and article["abstract"]:
        article_abstract = article["abstract"]

    # Extract archive
    extraction = extract_archive_to_temp(archive_path)
    if extraction.error:
        result.error = extraction.error
        return result

    try:
        result.total_files = len(extraction.files)

        # First pass: classify all files by heuristics
        heuristic_classifications = {}
        ambiguous_files = []

        for file_path in extraction.files:
            rel_path = file_path.relative_to(extraction.temp_dir)
            filename = str(rel_path)

            heuristic_result = classify_file_by_heuristics(filename)

            if heuristic_result is not None:
                category, reason = heuristic_result
                classification = FileClassification(
                    filename=filename,
                    category=category,
                    reason=reason,
                    method="heuristic",
                )
                heuristic_classifications[filename] = classification
                result.heuristic_classified += 1
            else:
                ambiguous_files.append((file_path, filename))

        # Second pass: batch classify ambiguous files with shallow LLM
        llm_classifications = {}
        if ambiguous_files and use_llm and endpoint and api_key:
            ambiguous_filenames = [fn for _, fn in ambiguous_files]
            llm_results = classify_files_shallow_llm(
                filenames=ambiguous_filenames,
                article_abstract=article_abstract,
                endpoint=endpoint,
                api_key=api_key,
                model=llm_model,
            )
            for classification in llm_results:
                llm_classifications[classification.filename] = classification
                if classification.category != "unknown":
                    result.llm_classified += 1
                else:
                    result.unknown += 1
        elif ambiguous_files:
            # LLM not available - mark all ambiguous files as unknown
            for _, filename in ambiguous_files:
                llm_classifications[filename] = FileClassification(
                    filename=filename,
                    category="unknown",
                    reason="Could not classify by heuristics, LLM not available",
                    method="heuristic",
                )
                result.unknown += 1

        # Combine results and store
        for file_path in extraction.files:
            rel_path = file_path.relative_to(extraction.temp_dir)
            filename = str(rel_path)

            if filename in heuristic_classifications:
                classification = heuristic_classifications[filename]
            else:
                classification = llm_classifications[filename]

            result.classifications.append(classification)

            if store_results:
                _store_classification(
                    conn=conn,
                    archive_path=str(archive_path),
                    file_path=filename,
                    classification=classification,
                    doi=doi,
                    pmid=pmid,
                    pmcid=pmcid,
                    model=llm_model if classification.method in ("llm", "shallow_llm") else None,
                )

    finally:
        extraction.cleanup()

    return result


def _store_classification(
    conn: sqlite3.Connection,
    archive_path: str,
    file_path: str,
    classification: FileClassification,
    doi: str | None,
    pmid: str | None,
    pmcid: str | None,
    model: str | None,
) -> None:
    """Store a file classification in the database."""
    # Map internal category to database category
    db_category = CATEGORY_TO_DB_MAPPING.get(classification.category, "unknown")
    
    conn.execute(
        """
        INSERT OR REPLACE INTO supplemental_file_classifications
        (doi, pmid, pmcid, archive_path, file_path, classification, justification, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doi,
            pmid,
            pmcid,
            archive_path,
            file_path,
            db_category,
            classification.reason,
            model,
        ),
    )
    conn.commit()


def get_supplemental_classifications(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    archive_path: str | None = None,
    category: FileCategory | None = None,
) -> list[dict]:
    """Retrieve supplemental file classifications from the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str | None
        Filter by article DOI.
    pmid : str | None
        Filter by article PMID.
    pmcid : str | None
        Filter by article PMCID.
    archive_path : str | None
        Filter by archive path.
    category : FileCategory | None
        Filter by classification category.

    Returns
    -------
    list[dict]
        List of classification records.
    """
    conditions = []
    params = []

    if doi:
        conditions.append("doi = ?")
        params.append(doi)
    if pmid:
        conditions.append("pmid = ?")
        params.append(pmid)
    if pmcid:
        conditions.append("pmcid = ?")
        params.append(pmcid)
    if archive_path:
        conditions.append("archive_path = ?")
        params.append(archive_path)
    if category:
        conditions.append("classification = ?")
        params.append(category)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    cursor = conn.execute(
        f"""
        SELECT * FROM supplemental_file_classifications
        WHERE {where_clause}
        ORDER BY file_path
        """,
        params,
    )

    return [dict(row) for row in cursor.fetchall()]


def has_supplemental_classification(
    conn: sqlite3.Connection,
    archive_path: str,
) -> bool:
    """Check if an archive has already been classified.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    archive_path : str
        Path to the supplementals archive.

    Returns
    -------
    bool
        True if classifications exist for this archive.
    """
    cursor = conn.execute(
        "SELECT 1 FROM supplemental_file_classifications WHERE archive_path = ? LIMIT 1",
        (archive_path,),
    )
    return cursor.fetchone() is not None
