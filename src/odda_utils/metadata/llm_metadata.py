# LLM-based metadata extraction from article full text.

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from odda_utils import llm

from odda_utils.database import (
    get_article,
    get_article_by_pmcid,
    get_article_by_pmid,
    init_db,
    insert_llm_extraction,
    insert_or_get_llm_keyword,
    link_article_llm_keyword,
)
from odda_utils.prompts import (
    analysis_methods,
    code_prompt,
    keyword_data_prompt,
    postamble,
    preamble,
    processed_data_prompt,
    raw_data_prompt,
)
from odda_utils.utils import AzureCredentialsError, get_azure_credentials

logger = logging.getLogger(__name__)


class LLMExtractionError(Exception):
    """Raised when LLM extraction fails."""

    pass


class JSONValidationError(LLMExtractionError):
    """Raised when LLM response is not valid JSON."""

    pass


class ArticleNotFoundError(LLMExtractionError):
    """Raised when article is not found in the database."""

    pass


@dataclass
class RawDataEntry:
    """Extracted raw data entry."""

    dataset_id: str
    data_repository: str
    url: str | None = None
    evidence_text: str | None = None


@dataclass
class ProcessedDataEntry:
    """Extracted processed data entry."""

    dataset_id: str
    data_repository: str
    url: str | None = None
    file: str | None = None
    evidence_text: str | None = None


@dataclass
class AnalysisMethod:
    """Extracted analysis method."""

    method_name: str
    method_description: str | None = None


@dataclass
class CodeEntry:
    """Extracted code entry."""

    repository: str
    url: str | None = None
    description: str | None = None


@dataclass
class ExtractedMetadata:
    """Container for all extracted metadata."""

    keywords: list[str] = field(default_factory=list)
    raw_data: list[RawDataEntry] = field(default_factory=list)
    processed_data: list[ProcessedDataEntry] = field(default_factory=list)
    analysis_methods: list[AnalysisMethod] = field(default_factory=list)
    code: list[CodeEntry] = field(default_factory=list)
    raw_response: str | None = None
    model: str | None = None


def build_extraction_prompt(
    text: str,
    include_keywords: bool = True,
    include_raw_data: bool = True,
    include_processed_data: bool = True,
    include_analysis_methods: bool = True,
    include_code: bool = True,
) -> str:
    """Build the full extraction prompt from subsections.

    Args:
        text: The article text to process.
        include_keywords: Whether to include keyword extraction.
        include_raw_data: Whether to include raw data extraction.
        include_processed_data: Whether to include processed data extraction.
        include_analysis_methods: Whether to include analysis methods extraction.
        include_code: Whether to include code extraction.

    Returns:
        The complete prompt string.
    """
    parts = [preamble.strip()]

    if include_keywords:
        parts.append(keyword_data_prompt.strip())
    if include_raw_data:
        parts.append(raw_data_prompt.strip())
    if include_processed_data:
        parts.append(processed_data_prompt.strip())
    if include_analysis_methods:
        parts.append(analysis_methods.strip())
    if include_code:
        parts.append(code_prompt.strip())

    parts.append(postamble.strip())
    parts.append(text)

    return "\n\n".join(parts)


_EXTRACTION_SYSTEM_PROMPT = (
    "You are a scientific data extraction assistant. Extract structured "
    "information from scientific articles and return the results as valid JSON."
)


def call_llm(
    prompt: str,
    endpoint: str,
    api_key: str,
    model: str,
    api_version: str = "2024-02-01",
    max_tokens: int = 16384,
    temperature: float = 1.0,
) -> str:
    """Call the configured chat LLM with the given prompt.

    Delegates to the provider-agnostic :mod:`odda_utils.llm` abstraction. The
    ``endpoint``, ``api_key``, ``model`` and ``api_version`` arguments are
    Azure-OpenAI hints preserved for backward compatibility; they are honoured
    only when the resolved chat provider is ``azure_openai``. For other
    providers (e.g. Claude via Azure) the configured credentials/model are used.

    Args:
        prompt: The prompt to send to the LLM.
        endpoint: Azure OpenAI endpoint URL (azure_openai hint).
        api_key: Azure OpenAI API key (azure_openai hint).
        model: Name of the model deployment (azure_openai hint).
        api_version: Azure OpenAI API version.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (OpenAI-family providers only).

    Returns:
        The LLM response text.

    Raises:
        LLMExtractionError: If the LLM call fails.
    """
    try:
        result = llm.complete_json(
            prompt,
            system=_EXTRACTION_SYSTEM_PROMPT,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            api_version=api_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise LLMExtractionError(f"LLM call failed: {e}") from e
    return result.text


def parse_llm_response(response: str, model: str) -> ExtractedMetadata:
    """Parse and validate the LLM JSON response.

    Args:
        response: The raw LLM response text.
        model: The model name used for extraction.

    Returns:
        ExtractedMetadata with parsed data.

    Raises:
        JSONValidationError: If the response is not valid JSON.
    """
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise JSONValidationError(f"Invalid JSON response: {e}") from e

    if not isinstance(data, dict):
        raise JSONValidationError(f"Expected JSON object, got {type(data).__name__}")

    result = ExtractedMetadata(raw_response=response, model=model)

    # Parse keywords
    if "keywords" in data:
        keywords = data["keywords"]
        if isinstance(keywords, list):
            result.keywords = [str(k) for k in keywords if k]

    # Parse raw data
    if "raw_data" in data:
        raw_data = data["raw_data"]
        if isinstance(raw_data, list):
            for entry in raw_data:
                if isinstance(entry, dict) and "dataset_id" in entry:
                    result.raw_data.append(
                        RawDataEntry(
                            dataset_id=str(entry.get("dataset_id", "")),
                            data_repository=str(entry.get("data_repository", "")),
                            url=entry.get("url"),
                            evidence_text=entry.get("evidence_text"),
                        )
                    )

    # Parse processed data
    if "processed_data" in data:
        processed_data = data["processed_data"]
        if isinstance(processed_data, list):
            for entry in processed_data:
                if isinstance(entry, dict) and "dataset_id" in entry:
                    result.processed_data.append(
                        ProcessedDataEntry(
                            dataset_id=str(entry.get("dataset_id", "")),
                            data_repository=str(entry.get("data_repository", "")),
                            url=entry.get("url"),
                            file=entry.get("file"),
                            evidence_text=entry.get("evidence_text"),
                        )
                    )

    # Parse analysis methods
    if "analysis_methods" in data:
        methods = data["analysis_methods"]
        if isinstance(methods, list):
            for entry in methods:
                if isinstance(entry, dict) and "method_name" in entry:
                    result.analysis_methods.append(
                        AnalysisMethod(
                            method_name=str(entry.get("method_name", "")),
                            method_description=entry.get("method_description"),
                        )
                    )

    # Parse code
    if "code" in data:
        code_entries = data["code"]
        if isinstance(code_entries, list):
            for entry in code_entries:
                if isinstance(entry, dict) and "repository" in entry:
                    result.code.append(
                        CodeEntry(
                            repository=str(entry.get("repository", "")),
                            url=entry.get("url"),
                            description=entry.get("description"),
                        )
                    )

    return result


def get_article_identifiers(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Get article identifiers from the database.

    Args:
        conn: Database connection.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        Tuple of (doi, pmid, pmcid) from the database record.

    Raises:
        ArticleNotFoundError: If the article is not found.
    """
    article = None

    if doi:
        article = get_article(conn, doi)
    elif pmid:
        article = get_article_by_pmid(conn, pmid)
    elif pmcid:
        article = get_article_by_pmcid(conn, pmcid)

    if article is None:
        id_str = doi or pmid or pmcid or "no ID provided"
        raise ArticleNotFoundError(f"Article not found: {id_str}")

    return article["doi"], article["pmid"], article["pmcid"]


def store_extracted_keywords(
    conn: sqlite3.Connection,
    keywords: list[str],
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Store extracted keywords in the database.

    Args:
        conn: Database connection.
        keywords: List of keyword strings.
        model: Name of the LLM model used for extraction.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        Number of keywords stored.
    """
    count = 0
    for keyword in keywords:
        keyword = keyword.strip()
        if not keyword:
            continue

        llm_keyword_id = insert_or_get_llm_keyword(conn, keyword, model)
        link_article_llm_keyword(
            conn,
            llm_keyword_id=llm_keyword_id,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
        )
        count += 1

    return count


def store_extracted_raw_data(
    conn: sqlite3.Connection,
    raw_data: list[RawDataEntry],
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Store extracted raw data entries in the database.

    Args:
        conn: Database connection.
        raw_data: List of RawDataEntry objects.
        model: Name of the LLM model used for extraction.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        Number of raw data entries stored.
    """
    count = 0
    for entry in raw_data:
        if not entry.dataset_id:
            continue

        conn.execute(
            """
            INSERT OR IGNORE INTO llm_raw_data
            (doi, pmid, pmcid, dataset_id, data_repository, url, evidence_text, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doi,
                pmid,
                pmcid,
                entry.dataset_id,
                entry.data_repository,
                entry.url,
                entry.evidence_text,
                model,
            ),
        )
        conn.commit()
        count += 1

    return count


def store_extracted_processed_data(
    conn: sqlite3.Connection,
    processed_data: list[ProcessedDataEntry],
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Store extracted processed data entries in the database.

    Args:
        conn: Database connection.
        processed_data: List of ProcessedDataEntry objects.
        model: Name of the LLM model used for extraction.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        Number of processed data entries stored.
    """
    count = 0
    for entry in processed_data:
        if not entry.dataset_id:
            continue

        conn.execute(
            """
            INSERT OR IGNORE INTO llm_processed_data
            (doi, pmid, pmcid, dataset_id, data_repository, url, file, evidence_text, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doi,
                pmid,
                pmcid,
                entry.dataset_id,
                entry.data_repository,
                entry.url,
                entry.file,
                entry.evidence_text,
                model,
            ),
        )
        conn.commit()
        count += 1

    return count


def store_extracted_analysis_methods(
    conn: sqlite3.Connection,
    analysis_methods_list: list[AnalysisMethod],
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Store extracted analysis methods in the database.

    Args:
        conn: Database connection.
        analysis_methods_list: List of AnalysisMethod objects.
        model: Name of the LLM model used for extraction.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        Number of analysis methods stored.
    """
    count = 0
    for method in analysis_methods_list:
        if not method.method_name:
            continue

        conn.execute(
            """
            INSERT OR IGNORE INTO llm_analysis_methods
            (doi, pmid, pmcid, method_name, method_description, model)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doi,
                pmid,
                pmcid,
                method.method_name,
                method.method_description,
                model,
            ),
        )
        conn.commit()
        count += 1

    return count


def store_extracted_code(
    conn: sqlite3.Connection,
    code_entries: list[CodeEntry],
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Store extracted code entries in the database.

    Args:
        conn: Database connection.
        code_entries: List of CodeEntry objects.
        model: Name of the LLM model used for extraction.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        Number of code entries stored.
    """
    count = 0
    for entry in code_entries:
        if not entry.repository:
            continue

        conn.execute(
            """
            INSERT OR IGNORE INTO llm_code
            (doi, pmid, pmcid, repository, url, description, model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doi,
                pmid,
                pmcid,
                entry.repository,
                entry.url,
                entry.description,
                model,
            ),
        )
        conn.commit()
        count += 1

    return count


def store_llm_extraction_record(
    conn: sqlite3.Connection,
    prompt: str,
    response: str,
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Store the LLM extraction prompt and response in the database.

    Args:
        conn: Database connection.
        prompt: The full prompt sent to the LLM.
        response: The raw LLM response text.
        model: Name of the LLM model used for extraction.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        The ID of the inserted extraction record.
    """
    return insert_llm_extraction(
        conn=conn,
        prompt=prompt,
        response=response,
        model=model,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
    )


@dataclass
class ExtractionResult:
    """Result of metadata extraction operation."""

    keywords_stored: int = 0
    raw_data_stored: int = 0
    processed_data_stored: int = 0
    analysis_methods_stored: int = 0
    code_stored: int = 0
    extraction_id: int | None = None
    extracted_metadata: ExtractedMetadata | None = None
    error: str | None = None


def extract_and_store_metadata(
    db_path: str | Path,
    text: str,
    model: str,
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    include_keywords: bool = True,
    include_raw_data: bool = True,
    include_processed_data: bool = True,
    include_analysis_methods: bool = True,
    include_code: bool = True,
    api_version: str = "2024-02-01",
    max_tokens: int = 4096,
) -> ExtractionResult:
    """Extract metadata from article text using an LLM and store in database.

    This function:
    1. Validates Azure credentials
    2. Looks up the article in the database
    3. Builds the extraction prompt
    4. Calls the LLM
    5. Parses and validates the JSON response
    6. Stores extracted data in the database

    Args:
        db_path: Path to the SQLite database file.
        text: The article full text to process.
        model: Name of the Azure OpenAI model deployment.
        endpoint_file: Path to file containing Azure OpenAI endpoint URL.
        api_key_file: Path to file containing Azure OpenAI API key.
        doi: Article DOI (provide at least one identifier).
        pmid: Article PMID.
        pmcid: Article PMCID.
        include_keywords: Whether to extract keywords.
        include_raw_data: Whether to extract raw data information.
        include_processed_data: Whether to extract processed data information.
        include_analysis_methods: Whether to extract analysis methods.
        include_code: Whether to extract code information.
        api_version: Azure OpenAI API version.
        max_tokens: Maximum tokens in the LLM response.

    Returns:
        ExtractionResult with counts of stored items and any errors.

    Raises:
        AzureCredentialsError: If Azure credentials are not available.
        ArticleNotFoundError: If the article is not in the database.
        JSONValidationError: If the LLM response is not valid JSON.
        LLMExtractionError: If the LLM call fails.
    """
    result = ExtractionResult()

    # Validate credentials
    endpoint, api_key = get_azure_credentials(endpoint_file, api_key_file)

    # Connect to database and get article identifiers
    conn = init_db(db_path)
    try:
        article_doi, article_pmid, article_pmcid = get_article_identifiers(
            conn, doi=doi, pmid=pmid, pmcid=pmcid
        )

        # Build prompt and call LLM
        prompt = build_extraction_prompt(
            text=text,
            include_keywords=include_keywords,
            include_raw_data=include_raw_data,
            include_processed_data=include_processed_data,
            include_analysis_methods=include_analysis_methods,
            include_code=include_code,
        )

        logger.info("Calling LLM for metadata extraction (model: %s)", model)
        response = call_llm(
            prompt=prompt,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            api_version=api_version,
            max_tokens=max_tokens,
        )

        # Store the prompt and response
        result.extraction_id = store_llm_extraction_record(
            conn=conn,
            prompt=prompt,
            response=response,
            model=model,
            doi=article_doi,
            pmid=article_pmid,
            pmcid=article_pmcid,
        )
        logger.info("Stored extraction record with ID %d", result.extraction_id)

        # Parse response
        extracted = parse_llm_response(response, model)
        result.extracted_metadata = extracted

        # Store extracted data
        if include_keywords and extracted.keywords:
            result.keywords_stored = store_extracted_keywords(
                conn,
                extracted.keywords,
                model,
                doi=article_doi,
                pmid=article_pmid,
                pmcid=article_pmcid,
            )
            logger.info("Stored %d keywords", result.keywords_stored)

        if include_raw_data and extracted.raw_data:
            result.raw_data_stored = store_extracted_raw_data(
                conn,
                extracted.raw_data,
                model,
                doi=article_doi,
                pmid=article_pmid,
                pmcid=article_pmcid,
            )
            logger.info("Stored %d raw data entries", result.raw_data_stored)

        if include_processed_data and extracted.processed_data:
            result.processed_data_stored = store_extracted_processed_data(
                conn,
                extracted.processed_data,
                model,
                doi=article_doi,
                pmid=article_pmid,
                pmcid=article_pmcid,
            )
            logger.info("Stored %d processed data entries", result.processed_data_stored)

        if include_analysis_methods and extracted.analysis_methods:
            result.analysis_methods_stored = store_extracted_analysis_methods(
                conn,
                extracted.analysis_methods,
                model,
                doi=article_doi,
                pmid=article_pmid,
                pmcid=article_pmcid,
            )
            logger.info("Stored %d analysis methods", result.analysis_methods_stored)

        if include_code and extracted.code:
            result.code_stored = store_extracted_code(
                conn,
                extracted.code,
                model,
                doi=article_doi,
                pmid=article_pmid,
                pmcid=article_pmcid,
            )
            logger.info("Stored %d code entries", result.code_stored)

    finally:
        conn.close()

    return result
