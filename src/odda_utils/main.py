# Unified MCP server for article management, metadata extraction, dataset operations,
# validation, feature requests, and utility tools. Merges the former mcp_common and
# knowledge_search servers into a single "odda" server.

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from odda_utils.database import (
    init_db,
    get_article,
    get_article_by_pmid,
    get_article_by_pmcid,
    get_articles_needing_llm_extraction,
    has_llm_extraction,
    insert_or_update_dataset,
    get_dataset,
    insert_dataset_file,
    get_dataset_files,
    delete_dataset_files,
    _decode_json,
    insert_quantification_run,
    get_quantification_run as _get_quantification_run,
    get_quantification_runs as _get_quantification_runs,
    insert_analysis_run,
    get_analysis_run as _get_analysis_run,
    get_analysis_runs as _get_analysis_runs,
    insert_dep_results,
    get_dep_results as _get_dep_results,
    insert_benchmark_annotation,
    get_benchmark_annotations as _get_benchmark_annotations,
    insert_benchmark_prediction,
    get_benchmark_predictions as _get_benchmark_predictions,
)
from odda_utils.fetching import (
    catalog_local_dataset_files,
    download_files_from_urls,
    download_gse_dataset,
    download_ipx_dataset,
    download_msv_dataset,
    download_pxd_dataset,
    fetch_gse_metadata,
    fetch_ipx_metadata,
    fetch_msv_metadata,
    fetch_pxd_files_metadata,
    fetch_pxd_metadata,
    get_gse_file_sizes,
    get_msv_file_sizes,
    get_pxd_file_sizes,
    list_archive_contents,
)
from odda_utils.metadata.llm_metadata import (
    build_extraction_prompt,
    call_llm,
    parse_llm_response,
    store_extracted_keywords,
    store_extracted_raw_data,
    store_extracted_processed_data,
    store_extracted_analysis_methods,
    store_extracted_code,
    store_llm_extraction_record,
    LLMExtractionError,
)
from odda_utils.datasets import (
    classify_article_supplementals,
    classify_file_by_heuristics,
    classify_files_shallow_llm,
    get_supplemental_classifications,
    has_supplemental_classification,
    SupplementalClassificationResult,
)
from odda_utils.utils import (
    get_azure_credentials,
    AzureCredentialsError,
    search_articles_by_embedding,
    search_articles_by_embedding_filtered,
)
from odda_utils.article_validation import (
    validate_article as _validate_article,
    validate_article_batch as _validate_article_batch,
    fetch_crossref_metadata as _fetch_crossref_metadata,
    fetch_pubmed_metadata as _fetch_pubmed_metadata,
)
from odda_utils.feature_requests import (
    submit_feature_request as _submit_feature_request,
    verify_feature_request as _verify_feature_request,
    get_oldest_approved_request as _get_oldest_approved_request,
    mark_request_implemented as _mark_request_implemented,
    mark_request_in_progress as _mark_request_in_progress,
    mark_request_incomplete as _mark_request_incomplete,
    FeatureRequestResult,
    SimilarRequestResult,
    ApprovedRequestResult,
    MarkImplementedResult,
    MarkInProgressResult,
    MarkIncompleteResult,
)
from odda_utils.uniprot_fasta import (
    download_uniprot_fasta as _download_uniprot_fasta,
    UniProtFastaDownloadResult,
)
from odda_utils.schema_info import (
    get_schema_info as _get_schema_info,
    SchemaInfoResult,
    TableInfo,
    ColumnInfo,
)
from odda_utils.dataset_utils import (
    check_dataset_exists as _check_dataset_exists,
    DatasetExistsResult,
)
from odda_utils.fidelity import (
    FidelityReport,
    assemble_report,
    compare_deps,
    compare_identifications,
    compare_quantitative,
    compare_versions,
    load_dep_results,
    load_diann_pg_matrix,
    load_matrix,
    load_maxquant_protein_groups,
)
from odda_utils.meta_analysis import (
    run_meta_analysis as _run_meta_analysis,
    run_meta_analysis_batch as _run_meta_analysis_batch,
    MetaAnalysisBatchResult,
    MetaAnalysisResult,
    PooledEstimate,
    Heterogeneity,
)
from odda_utils.injection_scan import (
    scan_injection as _scan_injection,
    scan_injection_batch as _scan_injection_batch,
    InjectionScanResult,
    InjectionScanBatchResult,
    CategorySignal,
    InjectionMatch,
)

logger = logging.getLogger(__name__)
app = FastMCP("odda")


# ---------------------------------------------------------------------------
# Dataclasses for tool results
# ---------------------------------------------------------------------------

@dataclass
class ArticleFullTextEntry:
    """Entry for a single article's full text retrieval result."""

    text: str | None
    error: str | None = None


@dataclass
class ArticleFullTextResult:
    """Result of a get_article_full_text operation."""

    articles: dict[str, ArticleFullTextEntry]
    found: int
    not_found: int
    no_filepath: int
    file_read_errors: int


@dataclass
class LLMExtractionResult:
    """Result of an LLM metadata extraction operation."""

    total_articles: int
    already_extracted: int
    newly_extracted: int
    failed: int
    skipped_no_fulltext: int


@dataclass
class DatasetIngestionResult:
    """Result of a dataset metadata ingestion operation."""

    dataset_id: str
    title: str | None = None
    description: str | None = None
    species: str | None = None
    instrument: str | None = None
    repository: str | None = None
    submission_date: str | None = None
    publication_date: str | None = None
    linked_doi: str | None = None
    linked_pmid: str | None = None
    files_cataloged: int = 0
    files_heuristic_classified: int = 0
    files_llm_classified: int = 0
    files_unknown: int = 0
    local_filepath: str | None = None
    already_exists: bool = False
    error: str | None = None


@dataclass
class ArticleValidationResult:
    """Result of validating a single article's metadata consistency."""

    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    stored_title: Optional[str] = None
    stored_publication_date: Optional[str] = None

    crossref_title: Optional[str] = None
    crossref_date: Optional[str] = None
    crossref_error: Optional[str] = None

    pubmed_title: Optional[str] = None
    pubmed_date: Optional[str] = None
    pubmed_doi: Optional[str] = None
    pubmed_error: Optional[str] = None

    title_matches_crossref: Optional[bool] = None
    title_matches_pubmed: Optional[bool] = None
    date_matches_crossref: Optional[bool] = None
    date_matches_pubmed: Optional[bool] = None
    ids_consistent: Optional[bool] = None

    is_valid: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass
class BatchValidationResult:
    """Result of validating multiple articles."""

    total_articles: int
    valid_articles: int
    invalid_articles: int
    results: list[ArticleValidationResult]


@dataclass
class PublicationDateUpdateResult:
    """Result of updating publication dates from PubMed."""

    total_articles: int = 0
    updated: int = 0
    already_had_date: int = 0
    no_pmid: int = 0
    fetch_failed: int = 0
    no_date_found: int = 0


# ---------------------------------------------------------------------------
# Provenance / research-object layer dataclasses (Phase 2)
# ---------------------------------------------------------------------------


@dataclass
class QuantificationRun:
    """A quantification run provenance record."""

    id: int
    dataset_id: Optional[str] = None
    tool: Optional[str] = None
    tool_version: Optional[str] = None
    container_image: Optional[str] = None
    container_sha256: Optional[str] = None
    param_file_path: Optional[str] = None
    param_file_sha256: Optional[str] = None
    command: Optional[str] = None
    input_files: Optional[list] = None
    output_dir: Optional[str] = None
    exit_status: Optional[int] = None
    wall_time_sec: Optional[float] = None
    host: Optional[str] = None
    extraction_model: Optional[str] = None
    provider: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class AnalysisRun:
    """An analysis run provenance record."""

    id: int
    quantification_run_id: Optional[int] = None
    analysis_type: Optional[str] = None
    method: Optional[str] = None
    library: Optional[str] = None
    library_version: Optional[str] = None
    parameters: Optional[object] = None
    code_sha256: Optional[str] = None
    random_seed: Optional[int] = None
    input_paths: Optional[list] = None
    output_paths: Optional[list] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class DepResult:
    """A single differential expression result row."""

    id: int
    analysis_run_id: Optional[int] = None
    feature_id: Optional[str] = None
    log2fc: Optional[float] = None
    pvalue: Optional[float] = None
    padj: Optional[float] = None
    direction: Optional[str] = None
    significant: Optional[bool] = None
    created_at: Optional[str] = None


@dataclass
class DepResultsWriteResult:
    """Result of a bulk differential-expression results write."""

    analysis_run_id: int
    inserted: int
    ids: list[int] = field(default_factory=list)


@dataclass
class BenchmarkAnnotation:
    """A benchmark (ground-truth) annotation record."""

    id: int
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    dataset_id: Optional[str] = None
    annotator: Optional[str] = None
    label: Optional[str] = None
    category: Optional[str] = None
    evidence_text: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class BenchmarkPrediction:
    """A benchmark prediction record."""

    id: int
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    dataset_id: Optional[str] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    run_at: Optional[str] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _convert_validation_result(result) -> ArticleValidationResult:
    """Convert internal ValidationResult to API-friendly ArticleValidationResult."""
    return ArticleValidationResult(
        doi=result.doi,
        pmid=result.pmid,
        pmcid=result.pmcid,
        stored_title=result.stored_title,
        stored_publication_date=(
            result.stored_publication_date.isoformat()
            if result.stored_publication_date
            else None
        ),
        crossref_title=(
            result.crossref_metadata.title
            if result.crossref_metadata
            else None
        ),
        crossref_date=(
            result.crossref_metadata.publication_date.isoformat()
            if result.crossref_metadata and result.crossref_metadata.publication_date
            else None
        ),
        crossref_error=(
            result.crossref_metadata.error
            if result.crossref_metadata
            else None
        ),
        pubmed_title=(
            result.pubmed_metadata.title
            if result.pubmed_metadata
            else None
        ),
        pubmed_date=(
            result.pubmed_metadata.publication_date.isoformat()
            if result.pubmed_metadata and result.pubmed_metadata.publication_date
            else None
        ),
        pubmed_doi=(
            result.pubmed_metadata.doi
            if result.pubmed_metadata
            else None
        ),
        pubmed_error=(
            result.pubmed_metadata.error
            if result.pubmed_metadata
            else None
        ),
        title_matches_crossref=result.title_matches_crossref,
        title_matches_pubmed=result.title_matches_pubmed,
        date_matches_crossref=result.date_matches_crossref,
        date_matches_pubmed=result.date_matches_pubmed,
        ids_consistent=result.ids_consistent,
        is_valid=result.is_valid,
        issues=result.issues,
    )


def _detect_id_type(identifier: str) -> str:
    """Detect the type of article identifier.

    Parameters
    ----------
    identifier : str
        Article identifier string.

    Returns
    -------
    str
        One of "doi", "pmid", or "pmcid".
    """
    identifier = identifier.strip()
    if identifier.upper().startswith("PMC") and identifier[3:].isdigit():
        return "pmcid"
    if "/" in identifier or identifier.startswith("10."):
        return "doi"
    if identifier.isdigit():
        return "pmid"
    return "doi"


def _row_to_quantification_run(row) -> QuantificationRun:
    """Convert a quantification_runs row into a QuantificationRun dataclass."""
    return QuantificationRun(
        id=row["id"],
        dataset_id=row["dataset_id"],
        tool=row["tool"],
        tool_version=row["tool_version"],
        container_image=row["container_image"],
        container_sha256=row["container_sha256"],
        param_file_path=row["param_file_path"],
        param_file_sha256=row["param_file_sha256"],
        command=row["command"],
        input_files=_decode_json(row["input_files_json"]),
        output_dir=row["output_dir"],
        exit_status=row["exit_status"],
        wall_time_sec=row["wall_time_sec"],
        host=row["host"],
        extraction_model=row["extraction_model"],
        provider=row["provider"],
        created_at=row["created_at"],
    )


def _row_to_analysis_run(row) -> AnalysisRun:
    """Convert an analysis_runs row into an AnalysisRun dataclass."""
    return AnalysisRun(
        id=row["id"],
        quantification_run_id=row["quantification_run_id"],
        analysis_type=row["analysis_type"],
        method=row["method"],
        library=row["library"],
        library_version=row["library_version"],
        parameters=_decode_json(row["parameters_json"]),
        code_sha256=row["code_sha256"],
        random_seed=row["random_seed"],
        input_paths=_decode_json(row["input_paths_json"]),
        output_paths=_decode_json(row["output_paths_json"]),
        provider=row["provider"],
        model=row["model"],
        created_at=row["created_at"],
    )


def _row_to_dep_result(row) -> DepResult:
    """Convert a dep_results row into a DepResult dataclass."""
    significant = row["significant"]
    return DepResult(
        id=row["id"],
        analysis_run_id=row["analysis_run_id"],
        feature_id=row["feature_id"],
        log2fc=row["log2fc"],
        pvalue=row["pvalue"],
        padj=row["padj"],
        direction=row["direction"],
        significant=None if significant is None else bool(significant),
        created_at=row["created_at"],
    )


def _row_to_benchmark_annotation(row) -> BenchmarkAnnotation:
    """Convert a benchmark_annotations row into a BenchmarkAnnotation dataclass."""
    return BenchmarkAnnotation(
        id=row["id"],
        doi=row["doi"],
        pmid=row["pmid"],
        pmcid=row["pmcid"],
        dataset_id=row["dataset_id"],
        annotator=row["annotator"],
        label=row["label"],
        category=row["category"],
        evidence_text=row["evidence_text"],
        created_at=row["created_at"],
    )


def _row_to_benchmark_prediction(row) -> BenchmarkPrediction:
    """Convert a benchmark_predictions row into a BenchmarkPrediction dataclass."""
    return BenchmarkPrediction(
        id=row["id"],
        doi=row["doi"],
        pmid=row["pmid"],
        pmcid=row["pmcid"],
        dataset_id=row["dataset_id"],
        predicted_label=row["predicted_label"],
        confidence=row["confidence"],
        model=row["model"],
        provider=row["provider"],
        run_at=row["run_at"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Tools from knowledge_graph (article fetching, extraction, datasets)
# ---------------------------------------------------------------------------

@app.tool()
def get_article_full_text(
    db_path: str | Path,
    ids: list[str],
) -> ArticleFullTextResult:
    """Retrieve full text for articles by their identifiers.

    Takes a list of article identifiers (DOI, PMID, or PMCID) and returns
    the full text content for each article that has been downloaded.

    The function automatically detects the identifier type:
    - DOI: Contains "/" or starts with "10."
    - PMCID: Starts with "PMC" followed by digits
    - PMID: Purely numeric

    Args:
        db_path: Path to the SQLite database file.
        ids: List of article identifiers (DOI, PMID, or PMCID).

    Returns:
        ArticleFullTextResult containing:
        - articles: Dict mapping input IDs to ArticleFullTextEntry with text or error
        - found: Count of articles found in database
        - not_found: Count of articles not found in database
        - no_filepath: Count of articles without downloaded full text
        - file_read_errors: Count of file read failures
    """
    conn = init_db(db_path)

    result = ArticleFullTextResult(
        articles={},
        found=0,
        not_found=0,
        no_filepath=0,
        file_read_errors=0,
    )

    try:
        for identifier in ids:
            identifier = identifier.strip()
            id_type = _detect_id_type(identifier)

            article = None
            if id_type == "doi":
                article = get_article(conn, identifier)
            elif id_type == "pmid":
                article = get_article_by_pmid(conn, identifier)
            elif id_type == "pmcid":
                article = get_article_by_pmcid(conn, identifier.upper())

            if article is None:
                result.articles[identifier] = ArticleFullTextEntry(
                    text=None,
                    error=f"Article not found in database"
                )
                result.not_found += 1
                continue

            result.found += 1
            article_filepath = article["article_filepath"]

            if not article_filepath:
                result.articles[identifier] = ArticleFullTextEntry(
                    text=None,
                    error="Article has no downloaded full text"
                )
                result.no_filepath += 1
                continue

            filepath = Path(article_filepath)
            if not filepath.exists():
                result.articles[identifier] = ArticleFullTextEntry(
                    text=None,
                    error=f"File not found: {article_filepath}"
                )
                result.file_read_errors += 1
                continue

            try:
                text = filepath.read_text(encoding="utf-8")
                result.articles[identifier] = ArticleFullTextEntry(text=text)
            except Exception as e:
                result.articles[identifier] = ArticleFullTextEntry(
                    text=None,
                    error=f"Failed to read file: {e}"
                )
                result.file_read_errors += 1
    finally:
        conn.close()

    return result


@app.tool()
def extract_article_llm_metadata(
    db_path: str | Path,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    process_all_pending: bool = False,
    max_articles: int = 100,
    overwrite: bool = False,
) -> LLMExtractionResult:
    """Extract LLM metadata from articles already in the database.

    This tool extracts keywords, raw data references, processed data references,
    and analysis methods from downloaded article full text using an LLM. Results
    are stored in the database tables: llm_keywords, llm_raw_data, llm_processed_data,
    and llm_analysis_methods.

    Can process a single article by ID, or batch process all articles with
    downloaded full text that haven't been processed yet.

    Args:
        db_path: Path to the SQLite database file.
        llm_model: Name of the Azure OpenAI chat model deployment.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.
        doi: Extract metadata for specific article by DOI.
        pmid: Extract metadata for specific article by PMID.
        pmcid: Extract metadata for specific article by PMCID.
        process_all_pending: If True, process all articles with full text that
            haven't been extracted yet (ignores doi/pmid/pmcid).
        max_articles: Maximum number of articles to process when process_all_pending=True.
        overwrite: If True, re-extract metadata even if already exists for this model.

    Returns:
        LLMExtractionResult with statistics about the operation.
    """
    conn = init_db(db_path)

    try:
        return _extract_article_llm_metadata_impl(
            conn=conn,
            llm_model=llm_model,
            endpoint_file=endpoint_file,
            api_key_file=api_key_file,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            process_all_pending=process_all_pending,
            max_articles=max_articles,
            overwrite=overwrite,
        )
    finally:
        conn.close()


def _extract_article_llm_metadata_impl(
    conn: sqlite3.Connection,
    llm_model: str,
    endpoint_file: str | Path | None,
    api_key_file: str | Path | None,
    doi: str | None,
    pmid: str | None,
    pmcid: str | None,
    process_all_pending: bool,
    max_articles: int,
    overwrite: bool,
) -> LLMExtractionResult:
    """Implementation of extract_article_llm_metadata with an existing connection."""
    try:
        endpoint, api_key = get_azure_credentials(endpoint_file, api_key_file)
    except AzureCredentialsError as e:
        raise AzureCredentialsError(
            f"Azure OpenAI credentials required for LLM extraction. {e}"
        ) from e

    result = LLMExtractionResult(
        total_articles=0,
        already_extracted=0,
        newly_extracted=0,
        failed=0,
        skipped_no_fulltext=0,
    )

    articles_to_process = []

    if process_all_pending:
        articles_to_process = get_articles_needing_llm_extraction(
            conn, llm_model, limit=max_articles
        )
        result.total_articles = len(articles_to_process)
        logger.info("Found %d articles needing LLM extraction", len(articles_to_process))
    else:
        article = None
        if doi:
            article = get_article(conn, doi)
        elif pmid:
            article = get_article_by_pmid(conn, pmid)
        elif pmcid:
            article = get_article_by_pmcid(conn, pmcid)
        else:
            raise ValueError(
                "Must provide doi, pmid, pmcid, or set process_all_pending=True"
            )

        if article is None:
            id_str = doi or pmid or pmcid
            raise ValueError(f"Article not found: {id_str}")

        articles_to_process = [article]
        result.total_articles = 1

    for article in articles_to_process:
        article_doi = article["doi"]
        article_pmid = article["pmid"]
        article_pmcid = article["pmcid"]
        article_filepath = article["article_filepath"]
        article_id = article_doi or article_pmid or article_pmcid

        if not article_filepath:
            logger.debug("Article %s has no full text, skipping", article_id)
            result.skipped_no_fulltext += 1
            continue

        if not overwrite and has_llm_extraction(
            conn, llm_model, doi=article_doi, pmid=article_pmid, pmcid=article_pmcid
        ):
            logger.debug("Article %s already has LLM extraction, skipping", article_id)
            result.already_extracted += 1
            continue

        filepath = Path(article_filepath)
        if not filepath.exists():
            logger.warning("Article file not found: %s", article_filepath)
            result.failed += 1
            continue

        try:
            article_text = filepath.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read article %s: %s", article_id, e)
            result.failed += 1
            continue

        try:
            prompt = build_extraction_prompt(article_text)
            logger.info("Extracting LLM metadata for %s", article_id)
            response = call_llm(
                prompt=prompt,
                endpoint=endpoint,
                api_key=api_key,
                model=llm_model,
            )

            extraction_id = store_llm_extraction_record(
                conn=conn,
                prompt=prompt,
                response=response,
                model=llm_model,
                doi=article_doi,
                pmid=article_pmid,
                pmcid=article_pmcid,
            )
            logger.debug("Stored extraction record with ID %d for %s", extraction_id, article_id)

            extracted = parse_llm_response(response, llm_model)

            if extracted.keywords:
                store_extracted_keywords(
                    conn,
                    extracted.keywords,
                    llm_model,
                    doi=article_doi,
                    pmid=article_pmid,
                    pmcid=article_pmcid,
                )
                logger.debug(
                    "Stored %d LLM keywords for %s", len(extracted.keywords), article_id
                )

            if extracted.raw_data:
                store_extracted_raw_data(
                    conn,
                    extracted.raw_data,
                    llm_model,
                    doi=article_doi,
                    pmid=article_pmid,
                    pmcid=article_pmcid,
                )
                logger.debug(
                    "Stored %d raw data entries for %s",
                    len(extracted.raw_data),
                    article_id,
                )

            if extracted.processed_data:
                store_extracted_processed_data(
                    conn,
                    extracted.processed_data,
                    llm_model,
                    doi=article_doi,
                    pmid=article_pmid,
                    pmcid=article_pmcid,
                )
                logger.debug(
                    "Stored %d processed data entries for %s",
                    len(extracted.processed_data),
                    article_id,
                )

            if extracted.analysis_methods:
                store_extracted_analysis_methods(
                    conn,
                    extracted.analysis_methods,
                    llm_model,
                    doi=article_doi,
                    pmid=article_pmid,
                    pmcid=article_pmcid,
                )
                logger.debug(
                    "Stored %d analysis methods for %s",
                    len(extracted.analysis_methods),
                    article_id,
                )

            if extracted.code:
                store_extracted_code(
                    conn,
                    extracted.code,
                    llm_model,
                    doi=article_doi,
                    pmid=article_pmid,
                    pmcid=article_pmcid,
                )
                logger.debug(
                    "Stored %d code entries for %s",
                    len(extracted.code),
                    article_id,
                )

            result.newly_extracted += 1
            logger.info("Extracted LLM metadata for %s", article_id)

        except LLMExtractionError as e:
            logger.warning("LLM extraction failed for %s: %s", article_id, e)
            result.failed += 1
        except Exception as e:
            logger.warning(
                "Unexpected error during LLM extraction for %s: %s", article_id, e
            )
            result.failed += 1

    return result


@app.tool()
def download_pxd(
    pxd_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = True,
    timeout: float = 10.0,
):
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
    """
    return download_pxd_dataset(
        pxd_id=pxd_id,
        output_dir=output_dir,
        force=force,
        silent=silent,
        timeout=timeout,
    )


@app.tool()
def get_pxd_size(
    pxd_id: str,
    timeout: float = 30.0,
    max_workers: int = 5,
):
    """Get file names and sizes for a ProteomeXchange dataset without downloading.

    Queries the ProteomeXchange API for the dataset file list, then uses HTTP
    HEAD requests to determine file sizes. This allows checking dataset size
    before committing to a download.

    Args:
        pxd_id: ProteomeXchange dataset identifier (e.g., "PXD021040").
        timeout: Maximum time to wait for HTTP requests in seconds.
        max_workers: Number of parallel workers for fetching file sizes.

    Returns:
        PXDFileSizeResult with file information including:
        - pxd_id: The dataset identifier
        - title: Dataset title
        - files: List of PXDFileInfo with filename, url, and size_bytes
        - total_size_bytes: Total size of all files in bytes
        - file_count: Number of files in the dataset
        - error: Error message if the operation failed
    """
    return get_pxd_file_sizes(
        pxd_id=pxd_id,
        timeout=timeout,
        max_workers=max_workers,
    )


@app.tool()
def download_ipx(
    ipx_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = True,
    timeout: float = 30.0,
):
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
    """
    return download_ipx_dataset(
        ipx_id=ipx_id,
        output_dir=output_dir,
        force=force,
        silent=silent,
        timeout=timeout,
    )


@app.tool()
def download_msv(
    msv_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = True,
    timeout: float = 30.0,
):
    """Download all files from a MassIVE dataset.

    Args:
        msv_id: MassIVE dataset identifier (e.g., "MSV000092832").
        output_dir: Directory to save downloaded files. Files will be stored
            in a subdirectory named after the MSV ID, preserving the original
            directory structure.
        force: If True, re-download files even if they already exist.
        silent: If True, hide download progress bars.
        timeout: Maximum time to wait for server responses when fetching
            project metadata.

    Returns:
        MSVDownloadResult with information about the download.
    """
    return download_msv_dataset(
        msv_id=msv_id,
        output_dir=output_dir,
        force=force,
        silent=silent,
        timeout=timeout,
    )


@app.tool()
def get_msv_size(
    msv_id: str,
    timeout: float = 30.0,
):
    """Get file names and sizes for a MassIVE dataset without downloading.

    Connects to the MassIVE FTP server and lists all files in the dataset
    directory. This allows checking dataset size before committing to a
    download.

    Args:
        msv_id: MassIVE dataset identifier (e.g., "MSV000092832").
        timeout: Maximum time to wait for FTP operations in seconds.

    Returns:
        MSVFileSizeResult with file information including:
        - msv_id: The dataset identifier
        - title: Dataset title (if available from PROXI API)
        - files: List of MSVFileInfo with filename, path, and size_bytes
        - total_size_bytes: Total size of all files in bytes
        - file_count: Number of files in the dataset
        - error: Error message if the operation failed
    """
    return get_msv_file_sizes(
        msv_id=msv_id,
        timeout=timeout,
    )


@app.tool()
def download_gse(
    gse_id: str,
    output_dir: str | Path,
    force: bool = False,
    silent: bool = True,
    timeout: float = 30.0,
):
    """Download all files from a GEO (Gene Expression Omnibus) dataset.

    Args:
        gse_id: GEO Series identifier (e.g., "GSE12345").
        output_dir: Directory to save downloaded files. Files will be stored
            in a subdirectory named after the GSE ID, preserving the original
            directory structure.
        force: If True, re-download files even if they already exist.
        silent: If True, hide download progress bars.
        timeout: Maximum time to wait for server responses when fetching
            project metadata.

    Returns:
        GEODownloadResult with information about the download.
    """
    return download_gse_dataset(
        gse_id=gse_id,
        output_dir=output_dir,
        force=force,
        silent=silent,
        timeout=timeout,
    )


@app.tool()
def get_gse_size(
    gse_id: str,
    timeout: float = 30.0,
):
    """Get file names and sizes for a GEO dataset without downloading.

    Connects to the NCBI FTP server and lists all files in the dataset
    directory. This allows checking dataset size before committing to a
    download.

    Args:
        gse_id: GEO Series identifier (e.g., "GSE12345").
        timeout: Maximum time to wait for FTP operations in seconds.

    Returns:
        GEOFileSizeResult with file information including:
        - gse_id: The dataset identifier
        - title: Dataset title (if available from E-utilities)
        - files: List of GEOFileInfo with filename, path, and size_bytes
        - total_size_bytes: Total size of all files in bytes
        - file_count: Number of files in the dataset
        - error: Error message if the operation failed
    """
    return get_gse_file_sizes(
        gse_id=gse_id,
        timeout=timeout,
    )


@app.tool()
def download_from_urls(
    files: list[dict],
    output_dir: str | Path,
    force: bool = False,
    silent: bool = True,
    timeout: float = 30.0,
):
    """Download files directly from URLs.

    Use this tool when download_pxd or download_ipx fail due to unsupported
    repositories. First use get_pxd_size to get file information including URLs,
    then pass that file list to this tool.

    Args:
        files: List of file info dictionaries from get_pxd_size result. Each dict
            should contain:
            - filename: Name to save the file as
            - url: URL to download from
            - size_bytes: Expected file size (for progress bar)
        output_dir: Directory to save downloaded files.
        force: If True, re-download files even if they already exist.
        silent: If True, hide download progress bars.
        timeout: Timeout for HTTP requests in seconds.

    Returns:
        URLDownloadResult with information about the download including:
        - output_dir: Directory where files were saved
        - downloaded_files: List of successfully downloaded file paths
        - skipped_files: List of files that already existed (when force=False)
        - failed_files: List of files that failed to download
        - total_bytes_downloaded: Total bytes downloaded
        - error: Error message if the operation failed
    """
    return download_files_from_urls(
        files=files,
        output_dir=output_dir,
        force=force,
        silent=silent,
        timeout=timeout,
    )


@app.tool()
def list_pmc_archive(archive_path: str | Path):
    """List the contents of a PMC supplementals archive (.tar.gz).

    Examines a PMC supplementals archive and returns information about all
    files and directories within it, without extracting the archive.

    Args:
        archive_path: Path to the .tar.gz archive file.

    Returns:
        ArchiveContentsResult containing:
        - archive_path: The path that was examined
        - files: List of ArchiveFileInfo with name, size, and is_dir
        - total_files: Count of files (excluding directories)
        - total_size: Total uncompressed size in bytes
        - error: Error message if the operation failed
    """
    return list_archive_contents(archive_path)


@app.tool()
def search_articles_embedding(
    db_path: str | Path,
    query: str,
    top_k: int = 5,
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
) -> list[dict]:
    """Search for articles similar to a query using embedding similarity.

    Args:
        db_path: Path to the SQLite database file.
        query: Search query text (will be embedded and compared to article abstracts).
        top_k: Number of top results to return.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.

    Returns:
        List of dictionaries containing article info and similarity scores.
    """
    conn = init_db(db_path)
    try:
        return search_articles_by_embedding(
            query=query,
            conn=conn,
            top_k=top_k,
            endpoint_file=endpoint_file,
            api_key_file=api_key_file,
        )
    finally:
        conn.close()


@app.tool()
def search_articles_filtered(
    db_path: str | Path,
    query: str,
    top_k: int = 5,
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    keywords: list[str] | None = None,
    has_dataset: bool | None = None,
    has_quantified_data: bool | None = None,
    authors: list[str] | None = None,
    article_type: str | None = None,
    journal_name: str | None = None,
    publication_date_from: str | None = None,
    publication_date_to: str | None = None,
    mesh_terms: list[str] | None = None,
    language: str | None = None,
) -> list[dict]:
    """Search for articles using embedding similarity with metadata filtering.

    Filters articles based on metadata criteria first, then ranks the
    remaining articles by embedding similarity to the query. This is more
    efficient than searching all articles and then filtering, especially
    when the metadata filters are selective.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    query : str
        Search query text (will be embedded and compared to article
        embeddings).
    top_k : int, optional
        Number of top results to return. Default is 5.
    endpoint_file : str or Path, optional
        Path to file containing Azure OpenAI endpoint.
    api_key_file : str or Path, optional
        Path to file containing Azure OpenAI API key.
    keywords : list of str, optional
        Keywords to filter by.
    has_dataset : bool, optional
        If True, only include articles with associated datasets.
    has_quantified_data : bool, optional
        If True, only include articles with quantified/processed data.
    authors : list of str, optional
        Author names to filter by (partial, case-insensitive).
    article_type : str, optional
        Filter by article type (exact match).
    journal_name : str, optional
        Filter by journal name (partial, case-insensitive match).
    publication_date_from : str, optional
        Include articles published on or after this date (YYYY-MM-DD).
    publication_date_to : str, optional
        Include articles published on or before this date (YYYY-MM-DD).
    mesh_terms : list of str, optional
        MeSH descriptor names to filter by (partial, case-insensitive).
    language : str, optional
        Filter by article language (exact match, e.g. "eng").

    Returns
    -------
    list of dict
        List of dictionaries sorted by similarity score (descending).
    """
    conn = init_db(db_path)
    try:
        return search_articles_by_embedding_filtered(
            query=query,
            conn=conn,
            top_k=top_k,
            endpoint_file=endpoint_file,
            api_key_file=api_key_file,
            keywords=keywords,
            has_dataset=has_dataset,
            has_quantified_data=has_quantified_data,
            authors=authors,
            article_type=article_type,
            journal_name=journal_name,
            publication_date_from=publication_date_from,
            publication_date_to=publication_date_to,
            mesh_terms=mesh_terms,
            language=language,
        )
    finally:
        conn.close()


@app.tool()
def classify_supplementals(
    db_path: str | Path,
    archive_path: str | Path,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    use_llm: bool = True,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    overwrite: bool = False,
) -> SupplementalClassificationResult:
    """Classify supplemental files from a scientific publication.

    Classifies files in a supplemental materials archive into four categories:
    - raw_data: Raw instrument data from data collection (mass spec, sequencing)
    - quantitative_data: Processed omic quantities (protein abundances, expression levels)
    - summary_data: Summary tables, figures, documents presenting findings
    - supporting: Files needed for processing (reference databases, configs)

    Uses a hybrid approach:
    1. Heuristic classification based on file extensions and naming patterns
    2. LLM-based classification for ambiguous files (if enabled)

    Args:
        db_path: Path to the SQLite database file.
        archive_path: Path to the supplementals .tar.gz archive.
        doi: Article DOI (used for article context and database linking).
        pmid: Article PMID.
        pmcid: Article PMCID.
        use_llm: Whether to use LLM for ambiguous files (default True).
        llm_model: Name of the LLM model deployment for classification.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.
        overwrite: If True, re-classify files even if already classified.

    Returns:
        SupplementalClassificationResult with classification details.
    """
    conn = init_db(db_path)

    try:
        if not overwrite and has_supplemental_classification(conn, str(archive_path)):
            existing = get_supplemental_classifications(conn, archive_path=str(archive_path))
            result = SupplementalClassificationResult(
                archive_path=str(archive_path),
                total_files=len(existing),
            )
            for row in existing:
                from odda_utils.datasets import FileClassification
                result.classifications.append(
                    FileClassification(
                        filename=row["file_path"],
                        category=row["classification"],
                        justification=row["justification"] or "",
                        method="llm" if row["model"] else "heuristic",
                    )
                )
                if row["model"]:
                    result.llm_classified += 1
                elif row["classification"] != "unknown":
                    result.heuristic_classified += 1
                else:
                    result.unknown += 1
            return result

        return classify_article_supplementals(
            archive_path=archive_path,
            conn=conn,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            use_llm=use_llm,
            llm_model=llm_model,
            endpoint_file=endpoint_file,
            api_key_file=api_key_file,
            store_results=True,
        )
    finally:
        conn.close()


@app.tool()
def ingest_pxd_metadata(
    db_path: str | Path,
    pxd_id: str,
    datasets_dir: str | Path | None = None,
    catalog_local_files: bool = True,
    use_llm_classification: bool = True,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> DatasetIngestionResult:
    """Ingest ProteomeXchange dataset metadata into the database.

    Fetches metadata from the PRIDE Archive API for a ProteomeXchange dataset
    and stores it in the database. Optionally catalogs local files if the
    dataset has been downloaded.

    Args:
        db_path: Path to the SQLite database file.
        pxd_id: ProteomeXchange dataset identifier (e.g., "PXD012345").
        datasets_dir: Base directory where datasets are downloaded.
        catalog_local_files: If True and local files exist, catalog them.
        use_llm_classification: If True, use LLM for ambiguous file classification.
        llm_model: Name of the LLM model deployment for classification.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.
        overwrite: If True, update existing dataset record.
        timeout: Request timeout for PRIDE API calls in seconds.

    Returns:
        DatasetIngestionResult with dataset metadata and ingestion status.
    """
    pxd_id = pxd_id.strip().upper()
    result = DatasetIngestionResult(dataset_id=pxd_id)

    conn = init_db(db_path)

    try:
        existing = get_dataset(conn, pxd_id)
        if existing and not overwrite:
            result.already_exists = True
            result.title = existing["title"]
            result.description = existing["description"]
            result.species = existing["species"]
            result.instrument = existing["instrument"]
            result.repository = existing["repository"]
            result.submission_date = existing["submission_date"]
            result.publication_date = existing["publication_date"]
            result.linked_doi = existing["doi"]
            result.linked_pmid = existing["pmid"]
            result.local_filepath = existing["local_filepath"]

            existing_files = get_dataset_files(conn, pxd_id)
            result.files_cataloged = len(existing_files)

            logger.info("Dataset %s already exists in database", pxd_id)
            return result

        metadata = fetch_pxd_metadata(pxd_id, timeout=timeout)

        if metadata.error:
            result.error = metadata.error
            return result

        species_str = ", ".join(metadata.species) if metadata.species else None
        instrument_str = ", ".join(metadata.instruments) if metadata.instruments else None

        linked_doi = None
        linked_pmid = None
        for pub in metadata.publications:
            if pub.doi and not linked_doi:
                linked_doi = pub.doi
            if pub.pubmed_id and not linked_pmid:
                linked_pmid = pub.pubmed_id

        local_filepath = None
        if datasets_dir:
            local_path = Path(datasets_dir) / pxd_id
            if local_path.exists() and local_path.is_dir():
                local_filepath = str(local_path)

        insert_or_update_dataset(
            conn=conn,
            dataset_id=pxd_id,
            title=metadata.title,
            description=metadata.description,
            species=species_str,
            instrument=instrument_str,
            repository=metadata.repository,
            submission_date=metadata.submission_date,
            publication_date=metadata.publication_date,
            local_filepath=local_filepath,
            doi=linked_doi,
            pmid=linked_pmid,
        )

        result.title = metadata.title
        result.description = metadata.description
        result.species = species_str
        result.instrument = instrument_str
        result.repository = metadata.repository
        result.submission_date = metadata.submission_date
        result.publication_date = metadata.publication_date
        result.linked_doi = linked_doi
        result.linked_pmid = linked_pmid
        result.local_filepath = local_filepath

        if catalog_local_files and local_filepath:
            _catalog_local_files_impl(
                conn=conn,
                dataset_id=pxd_id,
                local_filepath=local_filepath,
                linked_doi=linked_doi,
                linked_pmid=linked_pmid,
                use_llm_classification=use_llm_classification,
                llm_model=llm_model,
                endpoint_file=endpoint_file,
                api_key_file=api_key_file,
                overwrite=overwrite,
                result=result,
            )

        logger.info(
            "Ingested metadata for %s: %s",
            pxd_id,
            result.title[:50] + "..." if result.title and len(result.title) > 50 else result.title,
        )

    except Exception as e:
        logger.error("Failed to ingest dataset %s: %s", pxd_id, e)
        result.error = str(e)
    finally:
        conn.close()

    return result


def _catalog_local_files_impl(
    conn: sqlite3.Connection,
    dataset_id: str,
    local_filepath: str,
    linked_doi: str | None,
    linked_pmid: str | None,
    use_llm_classification: bool,
    llm_model: str,
    endpoint_file: str | Path | None,
    api_key_file: str | Path | None,
    overwrite: bool,
    result: DatasetIngestionResult,
) -> None:
    """Catalog local files for a dataset, classifying by heuristics and optionally LLM.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str
        Dataset identifier.
    local_filepath : str
        Path to the local dataset directory.
    linked_doi : str or None
        Article DOI linked to the dataset.
    linked_pmid : str or None
        Article PMID linked to the dataset.
    use_llm_classification : bool
        Whether to use LLM for ambiguous files.
    llm_model : str
        LLM model deployment name.
    endpoint_file : str or Path or None
        Path to Azure endpoint file.
    api_key_file : str or Path or None
        Path to Azure API key file.
    overwrite : bool
        If True, clear existing file records first.
    result : DatasetIngestionResult
        Result object to update with classification counts.
    """
    if overwrite:
        delete_dataset_files(conn, dataset_id)

    endpoint = None
    api_key = None
    if use_llm_classification:
        try:
            endpoint, api_key = get_azure_credentials(endpoint_file, api_key_file)
        except AzureCredentialsError as e:
            logger.warning("LLM classification disabled: %s", e)
            use_llm_classification = False

    article_text = None
    if use_llm_classification and (linked_doi or linked_pmid):
        article = None
        if linked_doi:
            article = get_article(conn, linked_doi)
        elif linked_pmid:
            article = get_article_by_pmid(conn, linked_pmid)

        if article and article["article_filepath"]:
            try:
                article_text = Path(article["article_filepath"]).read_text(
                    encoding="utf-8"
                )
            except Exception as e:
                logger.warning("Could not read article text for context: %s", e)

    local_files = catalog_local_dataset_files(dataset_id, local_filepath)

    unknown_files_indices = []
    for i, f in enumerate(local_files):
        if f.file_type == "unknown":
            unknown_files_indices.append(i)
        else:
            result.files_heuristic_classified += 1

    if use_llm_classification and endpoint and api_key and unknown_files_indices:
        unknown_filenames = [local_files[i].filename for i in unknown_files_indices]
        logger.info(
            "Classifying %d unknown files with shallow LLM for %s",
            len(unknown_filenames),
            dataset_id,
        )

        llm_classifications = classify_files_shallow_llm(
            filenames=unknown_filenames,
            article_text=article_text,
            endpoint=endpoint,
            api_key=api_key,
            model=llm_model,
        )

        for idx, llm_result in zip(unknown_files_indices, llm_classifications):
            file_meta = local_files[idx]
            file_meta.file_type = llm_result.category
            file_meta.file_type_reason = llm_result.justification
            file_meta.method = "shallow_llm"
            file_meta.model = llm_model

            if llm_result.category == "unknown":
                result.files_unknown += 1
            else:
                result.files_llm_classified += 1
    else:
        result.files_unknown = len(unknown_files_indices)

    for file_meta in local_files:
        local_file_path = str(Path(local_filepath) / file_meta.filename)
        insert_dataset_file(
            conn=conn,
            dataset_id=dataset_id,
            filename=file_meta.filename,
            file_type=file_meta.file_type,
            file_type_reason=file_meta.file_type_reason,
            method=file_meta.method,
            model=file_meta.model,
            size_bytes=file_meta.size_bytes,
            local_path=local_file_path,
        )

    result.files_cataloged = len(local_files)
    logger.info(
        "Cataloged %d local files for %s (%d heuristic, %d shallow LLM, %d unknown)",
        len(local_files),
        dataset_id,
        result.files_heuristic_classified,
        result.files_llm_classified,
        result.files_unknown,
    )


@app.tool()
def ingest_gse_metadata(
    db_path: str | Path,
    gse_id: str,
    datasets_dir: str | Path | None = None,
    catalog_local_files: bool = True,
    use_llm_classification: bool = True,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> DatasetIngestionResult:
    """Ingest GEO (Gene Expression Omnibus) dataset metadata into the database.

    Fetches metadata from the NCBI GEO E-utilities API for a GEO Series dataset
    and stores it in the database. Optionally catalogs local files if the
    dataset has been downloaded.

    Args:
        db_path: Path to the SQLite database file.
        gse_id: GEO Series identifier (e.g., "GSE12345").
        datasets_dir: Base directory where datasets are downloaded.
        catalog_local_files: If True and local files exist, catalog them.
        use_llm_classification: If True, use LLM for ambiguous file classification.
        llm_model: Name of the LLM model deployment for classification.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.
        overwrite: If True, update existing dataset record.
        timeout: Request timeout for GEO API calls in seconds.

    Returns:
        DatasetIngestionResult with dataset metadata and ingestion status.
    """
    gse_id = gse_id.strip().upper()
    result = DatasetIngestionResult(dataset_id=gse_id)

    conn = init_db(db_path)

    try:
        existing = get_dataset(conn, gse_id)
        if existing and not overwrite:
            result.already_exists = True
            result.title = existing["title"]
            result.description = existing["description"]
            result.species = existing["species"]
            result.instrument = existing["instrument"]
            result.repository = existing["repository"]
            result.submission_date = existing["submission_date"]
            result.publication_date = existing["publication_date"]
            result.linked_doi = existing["doi"]
            result.linked_pmid = existing["pmid"]
            result.local_filepath = existing["local_filepath"]

            existing_files = get_dataset_files(conn, gse_id)
            result.files_cataloged = len(existing_files)

            logger.info("Dataset %s already exists in database", gse_id)
            return result

        metadata = fetch_gse_metadata(gse_id, timeout=timeout)

        if metadata.error:
            result.error = metadata.error
            return result

        species_str = ", ".join(metadata.species) if metadata.species else None

        local_filepath = None
        if datasets_dir:
            local_path = Path(datasets_dir) / gse_id
            if local_path.exists() and local_path.is_dir():
                local_filepath = str(local_path)

        insert_or_update_dataset(
            conn=conn,
            dataset_id=gse_id,
            title=metadata.title,
            description=metadata.description,
            species=species_str,
            instrument=metadata.platform,
            repository=metadata.repository,
            submission_date=metadata.submission_date,
            publication_date=metadata.publication_date,
            local_filepath=local_filepath,
            doi=metadata.linked_doi,
            pmid=metadata.linked_pmid,
            pmcid=metadata.linked_pmcid,
        )

        result.title = metadata.title
        result.description = metadata.description
        result.species = species_str
        result.instrument = metadata.platform
        result.repository = metadata.repository
        result.submission_date = metadata.submission_date
        result.publication_date = metadata.publication_date
        result.linked_doi = metadata.linked_doi
        result.linked_pmid = metadata.linked_pmid
        result.local_filepath = local_filepath

        if catalog_local_files and local_filepath:
            _catalog_local_files_impl(
                conn=conn,
                dataset_id=gse_id,
                local_filepath=local_filepath,
                linked_doi=metadata.linked_doi,
                linked_pmid=metadata.linked_pmid,
                use_llm_classification=use_llm_classification,
                llm_model=llm_model,
                endpoint_file=endpoint_file,
                api_key_file=api_key_file,
                overwrite=overwrite,
                result=result,
            )

        logger.info(
            "Ingested metadata for %s: %s",
            gse_id,
            result.title[:50] + "..." if result.title and len(result.title) > 50 else result.title,
        )

    except Exception as e:
        logger.error("Failed to ingest dataset %s: %s", gse_id, e)
        result.error = str(e)
    finally:
        conn.close()

    return result


@app.tool()
def ingest_msv_metadata(
    db_path: str | Path,
    msv_id: str,
    datasets_dir: str | Path | None = None,
    catalog_local_files: bool = True,
    use_llm_classification: bool = True,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> DatasetIngestionResult:
    """Ingest MassIVE dataset metadata into the database.

    Fetches metadata from the MassIVE PROXI API for a MassIVE dataset and
    stores it in the database. Optionally catalogs local files if the
    dataset has been downloaded.

    Args:
        db_path: Path to the SQLite database file.
        msv_id: MassIVE dataset identifier (e.g., "MSV000092832").
        datasets_dir: Base directory where datasets are downloaded.
        catalog_local_files: If True and local files exist, catalog them.
        use_llm_classification: If True, use LLM for ambiguous file classification.
        llm_model: Name of the LLM model deployment for classification.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.
        overwrite: If True, update existing dataset record.
        timeout: Request timeout for MassIVE API calls in seconds.

    Returns:
        DatasetIngestionResult with dataset metadata and ingestion status.
    """
    msv_id = msv_id.strip().upper()
    result = DatasetIngestionResult(dataset_id=msv_id)

    conn = init_db(db_path)

    try:
        existing = get_dataset(conn, msv_id)
        if existing and not overwrite:
            result.already_exists = True
            result.title = existing["title"]
            result.description = existing["description"]
            result.species = existing["species"]
            result.instrument = existing["instrument"]
            result.repository = existing["repository"]
            result.submission_date = existing["submission_date"]
            result.publication_date = existing["publication_date"]
            result.linked_doi = existing["doi"]
            result.linked_pmid = existing["pmid"]
            result.local_filepath = existing["local_filepath"]

            existing_files = get_dataset_files(conn, msv_id)
            result.files_cataloged = len(existing_files)

            logger.info("Dataset %s already exists in database", msv_id)
            return result

        metadata = fetch_msv_metadata(msv_id, timeout=timeout)

        if metadata.error:
            result.error = metadata.error
            return result

        species_str = ", ".join(metadata.species) if metadata.species else None
        instrument_str = ", ".join(metadata.instruments) if metadata.instruments else None

        local_filepath = None
        if datasets_dir:
            local_path = Path(datasets_dir) / msv_id
            if local_path.exists() and local_path.is_dir():
                local_filepath = str(local_path)

        insert_or_update_dataset(
            conn=conn,
            dataset_id=msv_id,
            title=metadata.title,
            description=metadata.description,
            species=species_str,
            instrument=instrument_str,
            repository=metadata.repository,
            submission_date=metadata.submission_date,
            publication_date=metadata.publication_date,
            local_filepath=local_filepath,
            doi=metadata.linked_doi,
            pmid=metadata.linked_pmid,
        )

        result.title = metadata.title
        result.description = metadata.description
        result.species = species_str
        result.instrument = instrument_str
        result.repository = metadata.repository
        result.submission_date = metadata.submission_date
        result.publication_date = metadata.publication_date
        result.linked_doi = metadata.linked_doi
        result.linked_pmid = metadata.linked_pmid
        result.local_filepath = local_filepath

        if catalog_local_files and local_filepath:
            _catalog_local_files_impl(
                conn=conn,
                dataset_id=msv_id,
                local_filepath=local_filepath,
                linked_doi=metadata.linked_doi,
                linked_pmid=metadata.linked_pmid,
                use_llm_classification=use_llm_classification,
                llm_model=llm_model,
                endpoint_file=endpoint_file,
                api_key_file=api_key_file,
                overwrite=overwrite,
                result=result,
            )

        logger.info(
            "Ingested metadata for %s: %s",
            msv_id,
            result.title[:50] + "..." if result.title and len(result.title) > 50 else result.title,
        )

    except Exception as e:
        logger.error("Failed to ingest dataset %s: %s", msv_id, e)
        result.error = str(e)
    finally:
        conn.close()

    return result


@app.tool()
def ingest_ipx_metadata(
    db_path: str | Path,
    ipx_id: str,
    datasets_dir: str | Path | None = None,
    catalog_local_files: bool = True,
    use_llm_classification: bool = True,
    llm_model: str = "gpt-5",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> DatasetIngestionResult:
    """Ingest iProX dataset metadata into the database.

    Fetches metadata from the iProX PROXI API for an iProX dataset and
    stores it in the database. Optionally catalogs local files if the
    dataset has been downloaded.

    Args:
        db_path: Path to the SQLite database file.
        ipx_id: iProX dataset identifier (e.g., "IPX0001234000").
        datasets_dir: Base directory where datasets are downloaded.
        catalog_local_files: If True and local files exist, catalog them.
        use_llm_classification: If True, use LLM for ambiguous file classification.
        llm_model: Name of the LLM model deployment for classification.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.
        overwrite: If True, update existing dataset record.
        timeout: Request timeout for iProX API calls in seconds.

    Returns:
        DatasetIngestionResult with dataset metadata and ingestion status.
    """
    ipx_id = ipx_id.strip().upper()
    result = DatasetIngestionResult(dataset_id=ipx_id)

    conn = init_db(db_path)

    try:
        existing = get_dataset(conn, ipx_id)
        if existing and not overwrite:
            result.already_exists = True
            result.title = existing["title"]
            result.description = existing["description"]
            result.species = existing["species"]
            result.instrument = existing["instrument"]
            result.repository = existing["repository"]
            result.submission_date = existing["submission_date"]
            result.publication_date = existing["publication_date"]
            result.linked_doi = existing["doi"]
            result.linked_pmid = existing["pmid"]
            result.local_filepath = existing["local_filepath"]

            existing_files = get_dataset_files(conn, ipx_id)
            result.files_cataloged = len(existing_files)

            logger.info("Dataset %s already exists in database", ipx_id)
            return result

        metadata = fetch_ipx_metadata(ipx_id, timeout=timeout)

        if metadata.error:
            result.error = metadata.error
            return result

        species_str = ", ".join(metadata.species) if metadata.species else None
        instrument_str = ", ".join(metadata.instruments) if metadata.instruments else None

        local_filepath = None
        if datasets_dir:
            local_path = Path(datasets_dir) / ipx_id
            if local_path.exists() and local_path.is_dir():
                local_filepath = str(local_path)

        insert_or_update_dataset(
            conn=conn,
            dataset_id=ipx_id,
            title=metadata.title,
            description=metadata.description,
            species=species_str,
            instrument=instrument_str,
            repository=metadata.repository,
            submission_date=metadata.submission_date,
            publication_date=metadata.publication_date,
            local_filepath=local_filepath,
            doi=metadata.linked_doi,
            pmid=metadata.linked_pmid,
        )

        result.title = metadata.title
        result.description = metadata.description
        result.species = species_str
        result.instrument = instrument_str
        result.repository = metadata.repository
        result.submission_date = metadata.submission_date
        result.publication_date = metadata.publication_date
        result.linked_doi = metadata.linked_doi
        result.linked_pmid = metadata.linked_pmid
        result.local_filepath = local_filepath

        if catalog_local_files and local_filepath:
            _catalog_local_files_impl(
                conn=conn,
                dataset_id=ipx_id,
                local_filepath=local_filepath,
                linked_doi=metadata.linked_doi,
                linked_pmid=metadata.linked_pmid,
                use_llm_classification=use_llm_classification,
                llm_model=llm_model,
                endpoint_file=endpoint_file,
                api_key_file=api_key_file,
                overwrite=overwrite,
                result=result,
            )

        logger.info(
            "Ingested metadata for %s: %s",
            ipx_id,
            result.title[:50] + "..." if result.title and len(result.title) > 50 else result.title,
        )

    except Exception as e:
        logger.error("Failed to ingest dataset %s: %s", ipx_id, e)
        result.error = str(e)
    finally:
        conn.close()

    return result


# ---------------------------------------------------------------------------
# Tools from mcp_common (validation, feature requests, utilities)
# ---------------------------------------------------------------------------

@app.tool()
async def validate_article(
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    stored_title: Optional[str] = None,
    stored_publication_date: Optional[str] = None,
    title_similarity_threshold: float = 0.85,
) -> ArticleValidationResult:
    """Validate article metadata consistency across identifiers.

    Checks that article identifiers (DOI, PMID, PMCID) all point to the same
    article, and that the stored title and publication date match external
    sources (CrossRef for DOI, PubMed for PMID/PMCID).

    Args:
        doi: DOI of the article (without https://doi.org/ prefix).
        pmid: PubMed ID of the article.
        pmcid: PubMed Central ID of the article.
        stored_title: Title stored in the database to validate.
        stored_publication_date: Publication date stored in database (ISO format: YYYY-MM-DD).
        title_similarity_threshold: Minimum Jaccard similarity (0-1) for titles
            to be considered matching. Default 0.85.

    Returns:
        ArticleValidationResult with metadata comparison and issues found.
    """
    pub_date = None
    if stored_publication_date:
        pub_date = date.fromisoformat(stored_publication_date)

    result = await _validate_article(
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        stored_title=stored_title,
        stored_publication_date=pub_date,
        title_similarity_threshold=title_similarity_threshold,
    )

    return _convert_validation_result(result)


@app.tool()
async def validate_articles_from_db(
    db_path: str | Path,
    limit: int = 100,
    title_similarity_threshold: float = 0.85,
    requests_per_second: float = 1.0,
) -> BatchValidationResult:
    """Validate all articles in a database for metadata consistency.

    Fetches articles from the database and validates each one against
    CrossRef (for DOI) and PubMed (for PMID/PMCID).

    Args:
        db_path: Path to the SQLite database containing articles.
        limit: Maximum number of articles to validate.
        title_similarity_threshold: Minimum Jaccard similarity for title matching.
        requests_per_second: Maximum API requests per second (default 3.0).

    Returns:
        BatchValidationResult with validation results for each article.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT doi, pmid, pmcid, title, publication_date, electronic_publication_date
        FROM articles
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    articles = []
    for doi, pmid, pmcid, title, pub_date, epub_date in rows:
        articles.append({
            "doi": doi,
            "pmid": pmid,
            "pmcid": pmcid,
            "title": title,
            "publication_date": date.fromisoformat(pub_date) if pub_date else None,
            "electronic_publication_date": date.fromisoformat(epub_date) if epub_date else None,
        })

    results = await _validate_article_batch(
        articles=articles,
        title_similarity_threshold=title_similarity_threshold,
        requests_per_second=requests_per_second,
    )

    converted_results = [_convert_validation_result(r) for r in results]
    valid_count = sum(1 for r in converted_results if r.is_valid)

    return BatchValidationResult(
        total_articles=len(converted_results),
        valid_articles=valid_count,
        invalid_articles=len(converted_results) - valid_count,
        results=converted_results,
    )


@app.tool()
async def fetch_crossref_metadata(
    doi: str,
    timeout: float = 10.0,
) -> dict:
    """Fetch article metadata from CrossRef API using DOI.

    Args:
        doi: The DOI to look up (without https://doi.org/ prefix).
        timeout: Request timeout in seconds.

    Returns:
        Dictionary containing title, publication_date, doi, source, and error.
    """
    result = await _fetch_crossref_metadata(doi=doi, timeout=timeout)

    return {
        "title": result.title,
        "publication_date": (
            result.publication_date.isoformat() if result.publication_date else None
        ),
        "doi": result.doi,
        "source": result.source,
        "error": result.error,
    }


@app.tool()
async def fetch_pubmed_metadata(
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    timeout: float = 10.0,
) -> dict:
    """Fetch article metadata from PubMed/NCBI API using PMID or PMCID.

    Args:
        pmid: PubMed ID to look up.
        pmcid: PubMed Central ID to look up (used if pmid not provided).
        timeout: Request timeout in seconds.

    Returns:
        Dictionary containing title, dates, identifiers, source, and error.
    """
    result = await _fetch_pubmed_metadata(pmid=pmid, pmcid=pmcid, timeout=timeout)

    return {
        "title": result.title,
        "publication_date": (
            result.publication_date.isoformat() if result.publication_date else None
        ),
        "electronic_publication_date": (
            result.electronic_publication_date.isoformat() if result.electronic_publication_date else None
        ),
        "doi": result.doi,
        "pmid": result.pmid,
        "pmcid": result.pmcid,
        "source": result.source,
        "error": result.error,
    }


@app.tool()
async def fetch_publication_dates_from_pubmed(
    db_path: str | Path,
    limit: int = 100,
    requests_per_second: float = 1.0,
    overwrite: bool = False,
) -> PublicationDateUpdateResult:
    """Fetch print publication dates from PubMed and update the database.

    For each article in the database that has a PMID, fetches the print
    publication date from PubMed's JournalIssue/PubDate and stores it in
    the publication_date column.

    Args:
        db_path: Path to the SQLite database file.
        limit: Maximum number of articles to process.
        requests_per_second: Maximum API requests per second (default 1.0).
        overwrite: If True, update even if publication_date already exists.

    Returns:
        PublicationDateUpdateResult with statistics about the operation.
    """
    import asyncio
    import sqlite3

    from odda_utils.article_validation import RateLimiter, fetch_pubmed_metadata

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if overwrite:
        cursor.execute(
            """
            SELECT doi, pmid FROM articles
            WHERE pmid IS NOT NULL
            LIMIT ?
            """,
            (limit,),
        )
    else:
        cursor.execute(
            """
            SELECT doi, pmid FROM articles
            WHERE pmid IS NOT NULL AND publication_date IS NULL
            LIMIT ?
            """,
            (limit,),
        )

    rows = cursor.fetchall()
    result = PublicationDateUpdateResult(total_articles=len(rows))

    rate_limiter = RateLimiter(requests_per_second=requests_per_second)

    for row in rows:
        doi = row["doi"]
        pmid = row["pmid"]

        if not pmid:
            result.no_pmid += 1
            continue

        metadata = await fetch_pubmed_metadata(pmid=pmid, rate_limiter=rate_limiter)

        if metadata.error:
            result.fetch_failed += 1
            continue

        if not metadata.publication_date:
            result.no_date_found += 1
            continue

        cursor.execute(
            """
            UPDATE articles
            SET publication_date = ?, updated_at = CURRENT_TIMESTAMP
            WHERE doi = ?
            """,
            (metadata.publication_date.isoformat(), doi),
        )
        result.updated += 1

    conn.commit()
    conn.close()

    return result


@app.tool()
async def submit_feature_request(
    agent_name: str,
    request: str,
    reason_for_request: Optional[str] = None,
    db_path: str | Path = "./articles.sqlite",
    endpoint_file: Optional[str | Path] = None,
    api_key_file: Optional[str | Path] = None,
    embedding_model: str = "text-embedding-3-small",
) -> FeatureRequestResult:
    """Submit a feature request to the database with a semantic embedding.

    Args:
        agent_name: Name of the agent submitting the request.
        request: The feature request text describing what is needed.
        reason_for_request: Optional reason explaining why the request is being made.
        db_path: Path to the SQLite database file.
        endpoint_file: Path to file containing the Azure OpenAI endpoint URL.
        api_key_file: Path to file containing the Azure OpenAI API key.
        embedding_model: Name of the embedding model to use.

    Returns:
        FeatureRequestResult with request_id and embedding status.
    """
    return await _submit_feature_request(
        agent_name=agent_name,
        request=request,
        reason_for_request=reason_for_request,
        db_path=db_path,
        endpoint_file=endpoint_file,
        api_key_file=api_key_file,
        embedding_model=embedding_model,
    )


@app.tool()
async def verify_feature_request(
    request: str,
    db_path: str | Path = "./articles.sqlite",
    endpoint_file: Optional[str | Path] = None,
    api_key_file: Optional[str | Path] = None,
    embedding_model: str = "text-embedding-3-small",
    similarity_threshold: float = 0.0,
) -> SimilarRequestResult:
    """Find the most similar existing feature request to check for redundancy.

    Args:
        request: The feature request text to check for duplicates.
        db_path: Path to the SQLite database file.
        endpoint_file: Path to file containing the Azure OpenAI endpoint URL.
        api_key_file: Path to file containing the Azure OpenAI API key.
        embedding_model: Name of the embedding model to use.
        similarity_threshold: Minimum similarity score (0.0 to 1.0).

    Returns:
        SimilarRequestResult with the most similar request and similarity score.
    """
    return await _verify_feature_request(
        request=request,
        db_path=db_path,
        endpoint_file=endpoint_file,
        api_key_file=api_key_file,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
    )


@app.tool()
def get_oldest_approved_request(
    db_path: str | Path = "./articles.sqlite",
) -> ApprovedRequestResult:
    """Fetch the oldest approved feature request from the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        ApprovedRequestResult with the oldest approved request details.
    """
    return _get_oldest_approved_request(db_path=db_path)


@app.tool()
def mark_request_in_progress(
    request_id: int,
    status_reason: Optional[str] = None,
    db_path: str | Path = "./articles.sqlite",
) -> MarkInProgressResult:
    """Mark a feature request as in progress.

    Updates the request_status from 'approved' to 'in_progress' and sets
    the assigned_time to the current timestamp.

    Args:
        request_id: The ID of the feature request to mark as in progress.
        status_reason: Optional reason or notes about the status change.
        db_path: Path to the SQLite database file.

    Returns:
        MarkInProgressResult with operation status and request details.
    """
    return _mark_request_in_progress(
        request_id=request_id,
        status_reason=status_reason,
        db_path=db_path,
    )


@app.tool()
def mark_request_implemented(
    request_id: int,
    db_path: str | Path = "./articles.sqlite",
) -> MarkImplementedResult:
    """Mark a feature request as implemented.

    Updates the request_status from 'approved' or 'in_progress' to 'implemented'.

    Args:
        request_id: The ID of the feature request to mark as implemented.
        db_path: Path to the SQLite database file.

    Returns:
        MarkImplementedResult with operation status and request details.
    """
    return _mark_request_implemented(request_id=request_id, db_path=db_path)


@app.tool()
def mark_request_incomplete(
    request_id: int,
    status_reason: str,
    db_path: str | Path = "./articles.sqlite",
) -> MarkIncompleteResult:
    """Mark a feature request as incomplete.

    Updates the request_status from 'approved' or 'in_progress' to 'incomplete'.
    Used when a feature cannot be fully implemented due to external dependencies.

    Args:
        request_id: The ID of the feature request to mark as incomplete.
        status_reason: Required explanation of why the feature could not be completed.
        db_path: Path to the SQLite database file.

    Returns:
        MarkIncompleteResult with operation status and request details.
    """
    return _mark_request_incomplete(
        request_id=request_id,
        status_reason=status_reason,
        db_path=db_path,
    )


@app.tool()
def download_uniprot_fasta(
    proteome_id: Optional[str] = None,
    tax_id: Optional[int] = None,
    oscode: Optional[str] = None,
    superregnum: Optional[str] = None,
    species_name: Optional[str] = None,
    db_path: str | Path = "./articles.sqlite",
    download_dir: str | Path = "/data/supporting/fasta/",
    decompress: bool = True,
    overwrite: bool = False,
) -> UniProtFastaDownloadResult:
    """Download a UniProt FASTA file based on proteome metadata from the database.

    Queries the uniprot_fasta table to find a matching proteome entry, constructs
    the UniProt FTP URL, downloads the FASTA file, and updates the database with
    the local file path.

    Args:
        proteome_id: UniProt proteome ID (e.g., "UP000005640" for human).
        tax_id: NCBI taxonomy ID (e.g., 9606 for human).
        oscode: UniProt organism code (e.g., "HUMAN").
        superregnum: Domain/kingdom (e.g., "eukaryota", "bacteria").
        species_name: Species name (partial match supported).
        db_path: Path to the SQLite database file.
        download_dir: Directory to save downloaded FASTA files.
        decompress: If True, decompress the gzipped FASTA file after download.
        overwrite: If True, download even if the file already exists locally.

    Returns:
        UniProtFastaDownloadResult with download status and file information.
    """
    return _download_uniprot_fasta(
        db_path=db_path,
        download_dir=download_dir,
        decompress=decompress,
        overwrite=overwrite,
        proteome_id=proteome_id,
        tax_id=tax_id,
        oscode=oscode,
        superregnum=superregnum,
        species_name=species_name,
    )


@app.tool()
def get_database_schema(
    db_path: str | Path = "./articles.sqlite",
    table_name: Optional[str] = None,
) -> SchemaInfoResult:
    """Get schema information for tables in a SQLite database.

    Args:
        db_path: Path to the SQLite database file.
        table_name: If provided, only return information for this specific table.

    Returns:
        SchemaInfoResult with table and column information.
    """
    return _get_schema_info(db_path=db_path, table_name=table_name)


@app.tool()
def check_dataset_exists(
    dataset_id: str,
    datasets_dir: str | Path = "/data/datasets/",
) -> DatasetExistsResult:
    """Check if a dataset exists as a directory or archive in the datasets directory.

    Args:
        dataset_id: The dataset identifier to check (e.g., "PXD012345", "GSE12345").
        datasets_dir: Directory where datasets are stored. Defaults to "/data/datasets/".

    Returns:
        DatasetExistsResult with existence status and path information.
    """
    return _check_dataset_exists(dataset_id=dataset_id, datasets_dir=datasets_dir)


# ---------------------------------------------------------------------------
# Provenance / research-object layer tools (Phase 2)
# ---------------------------------------------------------------------------


@app.tool()
def record_quantification_run(
    db_path: str | Path,
    dataset_id: Optional[str] = None,
    tool: Optional[str] = None,
    tool_version: Optional[str] = None,
    container_image: Optional[str] = None,
    container_sha256: Optional[str] = None,
    param_file_path: Optional[str] = None,
    param_file_sha256: Optional[str] = None,
    command: Optional[str] = None,
    input_files: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    exit_status: Optional[int] = None,
    wall_time_sec: Optional[float] = None,
    host: Optional[str] = None,
    extraction_model: Optional[str] = None,
    provider: Optional[str] = None,
) -> QuantificationRun:
    """Record a quantification run as a reproducible research object.

    Stores full provenance for a single execution of an omic quantification
    tool (e.g., DIA-NN, MaxQuant) against a dataset: tool version, container
    image + digest, parameter file + hash, command line, input files, output
    directory, exit status, wall time, host, and optional model/provider.

    Args:
        db_path: Path to the SQLite database file.
        dataset_id: Source dataset identifier (e.g., "PXD012345").
        tool: Quantification tool name.
        tool_version: Version string of the tool.
        container_image: Container image reference (name:tag).
        container_sha256: SHA-256 digest of the container image.
        param_file_path: Path to the parameter/config file used.
        param_file_sha256: SHA-256 hash of the parameter file contents.
        command: Full command line executed.
        input_files: List of input file paths (stored as JSON).
        output_dir: Directory where outputs were written.
        exit_status: Process exit status code.
        wall_time_sec: Wall-clock run time in seconds.
        host: Host/machine identifier.
        extraction_model: LLM model used to derive parameters, if any.
        provider: LLM/compute provider (e.g., "azure").

    Returns:
        The created QuantificationRun record, including its new id.
    """
    conn = init_db(db_path)
    try:
        run_id = insert_quantification_run(
            conn,
            dataset_id=dataset_id,
            tool=tool,
            tool_version=tool_version,
            container_image=container_image,
            container_sha256=container_sha256,
            param_file_path=param_file_path,
            param_file_sha256=param_file_sha256,
            command=command,
            input_files=input_files,
            output_dir=output_dir,
            exit_status=exit_status,
            wall_time_sec=wall_time_sec,
            host=host,
            extraction_model=extraction_model,
            provider=provider,
        )
        return _row_to_quantification_run(_get_quantification_run(conn, run_id))
    finally:
        conn.close()


@app.tool()
def get_quantification_run(
    db_path: str | Path,
    run_id: int,
) -> Optional[QuantificationRun]:
    """Retrieve a single quantification run by ID.

    Args:
        db_path: Path to the SQLite database file.
        run_id: The quantification run ID.

    Returns:
        The QuantificationRun record, or None if not found.
    """
    conn = init_db(db_path)
    try:
        row = _get_quantification_run(conn, run_id)
        return _row_to_quantification_run(row) if row else None
    finally:
        conn.close()


@app.tool()
def get_quantification_runs(
    db_path: str | Path,
    dataset_id: Optional[str] = None,
    tool: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[QuantificationRun]:
    """Retrieve quantification runs, optionally filtered.

    Args:
        db_path: Path to the SQLite database file.
        dataset_id: Filter by source dataset identifier.
        tool: Filter by tool name.
        limit: Maximum number of rows to return.

    Returns:
        List of QuantificationRun records, newest first.
    """
    conn = init_db(db_path)
    try:
        rows = _get_quantification_runs(conn, dataset_id=dataset_id, tool=tool, limit=limit)
        return [_row_to_quantification_run(r) for r in rows]
    finally:
        conn.close()


@app.tool()
def record_analysis_run(
    db_path: str | Path,
    analysis_type: Optional[str] = None,
    method: Optional[str] = None,
    quantification_run_id: Optional[int] = None,
    library: Optional[str] = None,
    library_version: Optional[str] = None,
    parameters: Optional[dict] = None,
    code_sha256: Optional[str] = None,
    random_seed: Optional[int] = None,
    input_paths: Optional[list[str]] = None,
    output_paths: Optional[list[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> AnalysisRun:
    """Record a downstream analysis run as a reproducible research object.

    Stores provenance for a QC / differential expression (DE) / enrichment (or
    other) analysis performed on quantified data. Optionally links back to the
    quantification run that produced its inputs.

    Args:
        db_path: Path to the SQLite database file.
        analysis_type: Type of analysis (e.g., "QC", "DE", "enrichment").
        method: Method/algorithm name.
        quantification_run_id: ID of the parent quantification run, if any.
        library: Analysis library/package name.
        library_version: Version of the analysis library.
        parameters: Analysis parameters dict (stored as JSON).
        code_sha256: SHA-256 hash of the analysis code.
        random_seed: Random seed used for reproducibility.
        input_paths: List of input paths (stored as JSON).
        output_paths: List of output paths (stored as JSON).
        provider: LLM/compute provider, if any.
        model: LLM model used, if any.

    Returns:
        The created AnalysisRun record, including its new id.
    """
    conn = init_db(db_path)
    try:
        run_id = insert_analysis_run(
            conn,
            analysis_type=analysis_type,
            method=method,
            quantification_run_id=quantification_run_id,
            library=library,
            library_version=library_version,
            parameters=parameters,
            code_sha256=code_sha256,
            random_seed=random_seed,
            input_paths=input_paths,
            output_paths=output_paths,
            provider=provider,
            model=model,
        )
        return _row_to_analysis_run(_get_analysis_run(conn, run_id))
    finally:
        conn.close()


@app.tool()
def get_analysis_run(
    db_path: str | Path,
    run_id: int,
) -> Optional[AnalysisRun]:
    """Retrieve a single analysis run by ID.

    Args:
        db_path: Path to the SQLite database file.
        run_id: The analysis run ID.

    Returns:
        The AnalysisRun record, or None if not found.
    """
    conn = init_db(db_path)
    try:
        row = _get_analysis_run(conn, run_id)
        return _row_to_analysis_run(row) if row else None
    finally:
        conn.close()


@app.tool()
def get_analysis_runs(
    db_path: str | Path,
    quantification_run_id: Optional[int] = None,
    analysis_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[AnalysisRun]:
    """Retrieve analysis runs, optionally filtered.

    Args:
        db_path: Path to the SQLite database file.
        quantification_run_id: Filter by parent quantification run ID.
        analysis_type: Filter by analysis type.
        limit: Maximum number of rows to return.

    Returns:
        List of AnalysisRun records, newest first.
    """
    conn = init_db(db_path)
    try:
        rows = _get_analysis_runs(
            conn,
            quantification_run_id=quantification_run_id,
            analysis_type=analysis_type,
            limit=limit,
        )
        return [_row_to_analysis_run(r) for r in rows]
    finally:
        conn.close()


@app.tool()
def record_dep_results(
    db_path: str | Path,
    analysis_run_id: int,
    results: list[dict],
) -> DepResultsWriteResult:
    """Record differential expression results for an analysis run.

    Bulk-inserts per-feature effect sizes and significance produced by a
    differential expression analysis run.

    Args:
        db_path: Path to the SQLite database file.
        analysis_run_id: ID of the analysis run that produced these results.
        results: List of dicts, each optionally containing: feature_id,
            log2fc, pvalue, padj, direction, significant.

    Returns:
        DepResultsWriteResult with the analysis_run_id, inserted count, and
        the new row ids.
    """
    conn = init_db(db_path)
    try:
        ids = insert_dep_results(conn, analysis_run_id=analysis_run_id, results=results)
        return DepResultsWriteResult(
            analysis_run_id=analysis_run_id,
            inserted=len(ids),
            ids=ids,
        )
    finally:
        conn.close()


@app.tool()
def get_dep_results(
    db_path: str | Path,
    analysis_run_id: int,
    significant_only: bool = False,
    limit: Optional[int] = None,
) -> list[DepResult]:
    """Retrieve differential expression results for an analysis run.

    Args:
        db_path: Path to the SQLite database file.
        analysis_run_id: The analysis run ID to fetch results for.
        significant_only: If True, only return significant features.
        limit: Maximum number of rows to return.

    Returns:
        List of DepResult records, ordered by adjusted p-value.
    """
    conn = init_db(db_path)
    try:
        rows = _get_dep_results(
            conn,
            analysis_run_id=analysis_run_id,
            significant_only=significant_only,
            limit=limit,
        )
        return [_row_to_dep_result(r) for r in rows]
    finally:
        conn.close()


@app.tool()
def record_benchmark_annotation(
    db_path: str | Path,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    dataset_id: Optional[str] = None,
    annotator: Optional[str] = None,
    label: Optional[str] = None,
    category: Optional[str] = None,
    evidence_text: Optional[str] = None,
) -> BenchmarkAnnotation:
    """Record a benchmark (ground-truth) annotation.

    Args:
        db_path: Path to the SQLite database file.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.
        dataset_id: Associated dataset identifier.
        annotator: Name/identifier of the annotator.
        label: The ground-truth label.
        category: Category/task the label belongs to.
        evidence_text: Supporting evidence for the annotation.

    Returns:
        The created BenchmarkAnnotation record, including its new id.
    """
    conn = init_db(db_path)
    try:
        ann_id = insert_benchmark_annotation(
            conn,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            dataset_id=dataset_id,
            annotator=annotator,
            label=label,
            category=category,
            evidence_text=evidence_text,
        )
        rows = _get_benchmark_annotations(conn, limit=None)
        for r in rows:
            if r["id"] == ann_id:
                return _row_to_benchmark_annotation(r)
        # Fallback: should not happen, but keep a typed return.
        return BenchmarkAnnotation(id=ann_id)
    finally:
        conn.close()


@app.tool()
def get_benchmark_annotations(
    db_path: str | Path,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    dataset_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[BenchmarkAnnotation]:
    """Retrieve benchmark annotations, optionally filtered.

    Args:
        db_path: Path to the SQLite database file.
        doi: Filter by article DOI.
        pmid: Filter by article PMID.
        pmcid: Filter by article PMCID.
        dataset_id: Filter by dataset identifier.
        category: Filter by category.
        limit: Maximum number of rows to return.

    Returns:
        List of BenchmarkAnnotation records, newest first.
    """
    conn = init_db(db_path)
    try:
        rows = _get_benchmark_annotations(
            conn,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            dataset_id=dataset_id,
            category=category,
            limit=limit,
        )
        return [_row_to_benchmark_annotation(r) for r in rows]
    finally:
        conn.close()


@app.tool()
def record_benchmark_prediction(
    db_path: str | Path,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    dataset_id: Optional[str] = None,
    predicted_label: Optional[str] = None,
    confidence: Optional[float] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    run_at: Optional[str] = None,
) -> BenchmarkPrediction:
    """Record a benchmark prediction (to be scored against annotations).

    Args:
        db_path: Path to the SQLite database file.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.
        dataset_id: Associated dataset identifier.
        predicted_label: The predicted label.
        confidence: Confidence score for the prediction.
        model: Model that produced the prediction.
        provider: Provider of the model (e.g., "azure").
        run_at: Timestamp when the prediction was produced (ISO format).

    Returns:
        The created BenchmarkPrediction record, including its new id.
    """
    conn = init_db(db_path)
    try:
        pred_id = insert_benchmark_prediction(
            conn,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            dataset_id=dataset_id,
            predicted_label=predicted_label,
            confidence=confidence,
            model=model,
            provider=provider,
            run_at=run_at,
        )
        rows = _get_benchmark_predictions(conn, limit=None)
        for r in rows:
            if r["id"] == pred_id:
                return _row_to_benchmark_prediction(r)
        # Fallback: should not happen, but keep a typed return.
        return BenchmarkPrediction(id=pred_id)
    finally:
        conn.close()


@app.tool()
def get_benchmark_predictions(
    db_path: str | Path,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    dataset_id: Optional[str] = None,
    model: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[BenchmarkPrediction]:
    """Retrieve benchmark predictions, optionally filtered.

    Args:
        db_path: Path to the SQLite database file.
        doi: Filter by article DOI.
        pmid: Filter by article PMID.
        pmcid: Filter by article PMCID.
        dataset_id: Filter by dataset identifier.
        model: Filter by model.
        limit: Maximum number of rows to return.

    Returns:
        List of BenchmarkPrediction records, newest first.
    """
    conn = init_db(db_path)
    try:
        rows = _get_benchmark_predictions(
            conn,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            dataset_id=dataset_id,
            model=model,
            limit=limit,
        )
        return [_row_to_benchmark_prediction(r) for r in rows]
    finally:
        conn.close()


@app.tool()
def compute_fidelity_report(
    reproduced_matrix_path: Optional[str] = None,
    published_matrix_path: Optional[str] = None,
    matrix_format: str = "generic",
    id_column: Optional[str] = None,
    intensity_columns: Optional[list[str]] = None,
    sep: Optional[str] = None,
    log_transform: bool = True,
    log_base: float = 2.0,
    pseudocount: float = 0.0,
    sample_map: Optional[dict] = None,
    reproduced_dep_path: Optional[str] = None,
    published_dep_path: Optional[str] = None,
    dep_id_column: str = "feature_id",
    dep_log2fc_column: str = "log2fc",
    dep_pvalue_column: Optional[str] = "pvalue",
    dep_padj_column: Optional[str] = "padj",
    dep_significant_column: Optional[str] = "significant",
    dep_sep: Optional[str] = None,
    significance_threshold: float = 0.05,
    lfc_threshold: float = 0.0,
    use_padj: bool = True,
    version_a_path: Optional[str] = None,
    version_b_path: Optional[str] = None,
    version_a_label: str = "version_a",
    version_b_label: str = "version_b",
    version_id_column: Optional[str] = None,
    include_feature_lists: bool = True,
    db_path: Optional[str] = None,
    record: bool = False,
) -> FidelityReport:
    """Quantify and decompose how closely a reproduced omics result matches a published one.

    Computes any subset of four comparison sections, depending on which inputs
    are supplied, using deterministic, network-free, LLM-free math:

    1. Identification overlap (shared / reproduced-only / published-only counts
       and Jaccard) from the two abundance matrices.
    2. Quantitative agreement: per-sample and pooled Pearson and Spearman
       correlations of (optionally log-transformed) intensities on shared
       features.
    3. DEP decomposition: overlap of the significant sets plus a four-bucket
       attribution (concordant, not_quantified, quantified_not_significant,
       significant_different_direction) of every published-significant feature,
       explaining the non-reproduced hits.
    4. Version comparison: gained / lost / shared identifications between two
       tool versions (e.g. DIA-NN v1.8.1 vs v2.3.1).

    Args:
        reproduced_matrix_path: Path to the reproduced abundance matrix file.
        published_matrix_path: Path to the published abundance matrix file.
        matrix_format: One of "generic", "diann", or "maxquant" (selects the
            loader and its default column detection).
        id_column: Feature-id column name (matrix format defaults apply when None).
        intensity_columns: Explicit list of sample/intensity columns; auto-detected
            when None.
        sep: Field delimiter for matrix files; inferred from extension when None.
        log_transform: Log-transform intensities before correlating. Default True.
        log_base: Logarithm base used when log_transform is True. Default 2.0.
        pseudocount: Value added before taking the logarithm. Default 0.0.
        sample_map: Mapping of reproduced sample name -> published sample name;
            when None, identically named samples are paired.
        reproduced_dep_path: Path to the reproduced DEP results file.
        published_dep_path: Path to the published DEP results file.
        dep_id_column: DEP feature-id column name.
        dep_log2fc_column: DEP log2 fold-change column name.
        dep_pvalue_column: DEP raw p-value column name.
        dep_padj_column: DEP adjusted p-value column name.
        dep_significant_column: DEP explicit significance-flag column name.
        dep_sep: Field delimiter for DEP files; inferred from extension when None.
        significance_threshold: Threshold for derived DEP significance. Default 0.05.
        lfc_threshold: Minimum absolute log2 fold change for derived significance.
        use_padj: Prefer padj over pvalue for derived significance. Default True.
        version_a_path: Path to identification set for version A (baseline).
        version_b_path: Path to identification set for version B (comparison).
        version_a_label: Label for version A.
        version_b_label: Label for version B.
        version_id_column: Feature-id column for the version files (falls back to
            id_column, then the matrix-format default).
        include_feature_lists: Include per-section feature-id lists. Default True.
        db_path: Optional SQLite database path used only when record=True.
        record: If True and db_path is set, persist a compact summary via an
            analysis_runs record (analysis_type="fidelity"). Failures to record
            are non-fatal and noted in the report.

    Returns:
        FidelityReport with the requested sections populated; sections without
        inputs are left as None. When persisted, recorded_analysis_run_id is set.
    """

    def _load_matrix_by_format(path: str):
        if matrix_format == "diann":
            return load_diann_pg_matrix(
                path,
                id_column=id_column or "Protein.Group",
                intensity_columns=intensity_columns,
                sep=sep,
            )
        if matrix_format == "maxquant":
            kwargs = {"intensity_columns": intensity_columns, "sep": sep}
            if id_column:
                kwargs["id_column"] = id_column
            return load_maxquant_protein_groups(path, **kwargs)
        return load_matrix(
            path,
            id_column=id_column,
            intensity_columns=intensity_columns,
            sep=sep,
        )

    notes: list[str] = []
    identification = None
    quantitative = None
    dep = None
    version = None

    reproduced_matrix = None
    published_matrix = None

    if reproduced_matrix_path and published_matrix_path:
        reproduced_matrix = _load_matrix_by_format(reproduced_matrix_path)
        published_matrix = _load_matrix_by_format(published_matrix_path)
        identification = compare_identifications(
            reproduced_matrix,
            published_matrix,
            include_feature_lists=include_feature_lists,
        )
        quantitative = compare_quantitative(
            reproduced_matrix,
            published_matrix,
            sample_map=sample_map,
            log_transform=log_transform,
            log_base=log_base,
            pseudocount=pseudocount,
        )
    elif reproduced_matrix_path or published_matrix_path:
        notes.append(
            "Both reproduced_matrix_path and published_matrix_path are required "
            "for identification/quantitative comparison; skipping those sections."
        )

    if reproduced_dep_path and published_dep_path:
        reproduced_deps = load_dep_results(
            reproduced_dep_path,
            id_column=dep_id_column,
            log2fc_column=dep_log2fc_column,
            pvalue_column=dep_pvalue_column,
            padj_column=dep_padj_column,
            significant_column=dep_significant_column,
            sep=dep_sep,
        )
        published_deps = load_dep_results(
            published_dep_path,
            id_column=dep_id_column,
            log2fc_column=dep_log2fc_column,
            pvalue_column=dep_pvalue_column,
            padj_column=dep_padj_column,
            significant_column=dep_significant_column,
            sep=dep_sep,
        )
        reproduced_quantified_ids = (
            list(reproduced_matrix.feature_ids) if reproduced_matrix else None
        )
        dep = compare_deps(
            reproduced_deps,
            published_deps,
            reproduced_quantified_ids=reproduced_quantified_ids,
            significance_threshold=significance_threshold,
            lfc_threshold=lfc_threshold,
            use_padj=use_padj,
            include_feature_lists=include_feature_lists,
        )
    elif reproduced_dep_path or published_dep_path:
        notes.append(
            "Both reproduced_dep_path and published_dep_path are required for the "
            "DEP decomposition; skipping that section."
        )

    if version_a_path and version_b_path:
        v_id_col = version_id_column or id_column
        version_a = load_matrix(version_a_path, id_column=v_id_col, sep=sep)
        version_b = load_matrix(version_b_path, id_column=v_id_col, sep=sep)
        version = compare_versions(
            version_a,
            version_b,
            label_a=version_a_label,
            label_b=version_b_label,
            include_feature_lists=include_feature_lists,
        )
    elif version_a_path or version_b_path:
        notes.append(
            "Both version_a_path and version_b_path are required for the version "
            "comparison; skipping that section."
        )

    report = assemble_report(
        identification=identification,
        quantitative=quantitative,
        dep=dep,
        version=version,
        notes=notes,
    )

    if record and db_path:
        summary = {
            "identification": {
                "n_shared": identification.n_shared,
                "jaccard": identification.jaccard,
            }
            if identification
            else None,
            "quantitative": {
                "pooled_pearson": quantitative.pooled_pearson,
                "pooled_spearman": quantitative.pooled_spearman,
            }
            if quantitative
            else None,
            "dep": {
                "overlap_pct_of_published": dep.overlap_pct_of_published,
                "not_quantified": dep.not_quantified,
                "quantified_not_significant": dep.quantified_not_significant,
                "significant_different_direction": dep.significant_different_direction,
            }
            if dep
            else None,
            "version": {
                "n_gained": version.n_gained,
                "n_lost": version.n_lost,
            }
            if version
            else None,
        }
        try:
            conn = init_db(db_path)
            try:
                input_paths = [
                    p
                    for p in (
                        reproduced_matrix_path,
                        published_matrix_path,
                        reproduced_dep_path,
                        published_dep_path,
                        version_a_path,
                        version_b_path,
                    )
                    if p
                ]
                run_id = insert_analysis_run(
                    conn,
                    analysis_type="fidelity",
                    method="fidelity_report",
                    library="odda_utils.fidelity",
                    parameters=summary,
                    input_paths=input_paths,
                )
                report.recorded_analysis_run_id = run_id
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("Failed to record fidelity analysis run: %s", exc)
            report.notes.append(f"Failed to record analysis run: {exc}")

    return report


@app.tool()
def meta_analysis(
    effects: Optional[list[float]] = None,
    variances: Optional[list[float]] = None,
    standard_errors: Optional[list[float]] = None,
    pvalues: Optional[list[float]] = None,
    entities: Optional[dict[str, list[dict[str, float]]]] = None,
    name: str = "effect",
) -> MetaAnalysisBatchResult:
    """Statistically combine per-study effect sizes across studies.

    Runs both a fixed-effect (inverse-variance) and a DerSimonian-Laird
    random-effects meta-analysis, generalizing the system's cross-study
    comparison into a formal pooled estimate. Two calling styles are supported:

    1. Single entity: pass parallel ``effects`` plus exactly one of
       ``variances``, ``standard_errors``, or ``pvalues``. Standard errors are
       squared to variances; p-values are converted to standard errors via
       SE = |effect| / z (two-sided) and then squared.
    2. Many entities at once (the typical use for proteins/genes): pass
       ``entities`` mapping each entity name to a list of per-study records.
       Each record is a dict with an effect key ("yi"/"effect"/"effect_size"/
       "es"/"log2fc"/"logfc") and one uncertainty key (a variance "vi"/
       "variance"/"var", a standard error "se"/"standard_error"/"std_error", or
       a p-value "p"/"pvalue"/"p_value"/"pval"). Per-entity errors are captured
       on that entity's result and do not abort the batch.

    When ``entities`` is provided it takes precedence over the single-entity
    arguments. Studies with a non-finite effect or a non-positive variance are
    dropped before pooling.

    Args:
        effects: Per-study effect sizes for a single entity (e.g. log2 fold
            changes).
        variances: Per-study variances (SE**2). Mutually exclusive with
            standard_errors and pvalues.
        standard_errors: Per-study standard errors.
        pvalues: Per-study two-sided p-values.
        entities: Mapping of entity name to a list of per-study record dicts,
            for meta-analyzing many entities in one call.
        name: Label for the single-entity result (default "effect").

    Returns:
        MetaAnalysisBatchResult keyed by entity name. Each MetaAnalysisResult
        holds the number of pooled studies (k), the fixed- and random-effects
        pooled estimates (estimate, se, 95% CI, and z/p for random effects), and
        heterogeneity statistics (Q, Q_p, df, I2, tau2). Entities that could not
        be analyzed carry an ``error`` message and k = 0.
    """
    if entities is not None:
        return _run_meta_analysis_batch(entities)
    single = _run_meta_analysis(
        effects=effects,
        variances=variances,
        standard_errors=standard_errors,
        pvalues=pvalues,
        name=name,
    )
    return MetaAnalysisBatchResult(
        results={single.name or name: single},
        n_entities=1,
        n_succeeded=0 if single.error else 1,
        n_failed=1 if single.error else 0,
    )


@app.tool()
def scan_injection(
    text: Optional[str] = None,
    source_label: Optional[str] = None,
    items: Optional[dict[str, str]] = None,
    flag_threshold: float = 40.0,
    snippet_len: int = 160,
    include_snippets: bool = True,
    max_matches_per_category: int = 50,
    min_base64_len: int = 48,
    max_chars: Optional[int] = 2_000_000,
) -> InjectionScanBatchResult:
    """Scan untrusted article/supplemental text for prompt-injection patterns.

    Defensive telemetry for the ODDA trust boundary. Extracted text from an
    article and its supplements is untrusted input; this tool measures it for
    instruction-like / command-injection patterns directed at an AI so that
    suspicious inputs can be flagged for human review and the signal stored as a
    provenance field. It is pure and side-effect-free: it NEVER executes,
    follows, downloads, or otherwise acts on the scanned content -- it only
    counts matches and computes a bounded risk score.

    Detected pattern categories:
    - instruction_override ("ignore previous instructions", "disregard", "forget")
    - role_manipulation ("as an AI", "system prompt", "developer mode", "jailbreak")
    - imperative_to_ai ("you must", "you should", "make sure to", "do not reveal")
    - database_manipulation ("add the keyword", "insert into", "classify as")
    - tool_command_injection (os.system(, subprocess, eval(, rm -rf, curl ... | sh)
    - url_exfiltration (URLs, "send/upload the data to ...", IP addresses)
    - encoded_payload (base64 blobs, long hex strings, \\x escapes, data: URIs)

    Two calling styles are supported (both return the same batch container so
    callers can treat the output uniformly):

    1. Single text: pass ``text`` (and optionally ``source_label``). The result
       is keyed by ``source_label`` (or ``"text"``).
    2. Many texts at once (typical: main text plus each supplemental file): pass
       ``items`` mapping a label (filename or ``"main_text"``) to its text.
       Per-item errors are captured on that item and do not abort the batch.

    When both are given, ``items`` takes precedence.

    This is deterministic pattern telemetry, not a classifier; false positives
    (e.g. a methods section literally discussing a "system prompt") are expected
    and acceptable because the signal only gates human review, never an
    automated action on the untrusted text.

    Args:
        text: A single text to scan (single-text style).
        source_label: Label for the single text (e.g. a DOI or filename).
        items: Mapping of label -> text for scanning many texts at once.
        flag_threshold: risk_score at/above which an item is counted as flagged
            (default 40.0, the medium-risk cutoff).
        snippet_len: Maximum length of each returned match snippet.
        include_snippets: If False, omit match snippets (offsets/counts remain),
            so the signal can be stored without echoing the payload.
        max_matches_per_category: Cap on retained spans per category (the
            reported count is still the true total).
        min_base64_len: Minimum base64-like run length to flag as encoded_payload.
        max_chars: Only the leading max_chars characters are scanned (None to
            scan everything).

    Returns:
        InjectionScanBatchResult keyed by item label. Each InjectionScanResult
        holds per-category counts and matched spans, the matched-category list,
        an unbounded weighted_score, a bounded risk_score in [0, 100], and a
        coarse risk_level ("none"/"low"/"medium"/"high"). The batch adds flag
        and error counts and the list of flagged labels.
    """
    if items is not None:
        return _scan_injection_batch(
            items,
            flag_threshold=flag_threshold,
            snippet_len=snippet_len,
            include_snippets=include_snippets,
            max_matches_per_category=max_matches_per_category,
            min_base64_len=min_base64_len,
            max_chars=max_chars,
        )
    label = source_label or "text"
    return _scan_injection_batch(
        {label: text or ""},
        flag_threshold=flag_threshold,
        snippet_len=snippet_len,
        include_snippets=include_snippets,
        max_matches_per_category=max_matches_per_category,
        min_base64_len=min_base64_len,
        max_chars=max_chars,
    )


def main():
    """Run the odda MCP server."""
    from odda_utils.articles.pubmed import search_and_fetch

    app.add_tool(search_and_fetch)
    app.run()


if __name__ == "__main__":
    main()
