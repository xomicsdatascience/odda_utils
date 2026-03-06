# PubMed search-and-fetch pipeline: search PubMed, fetch article metadata,
# download full text, generate embeddings, and optionally extract LLM metadata.

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from odda_utils.database import (
    init_db,
    insert_embedding,
    insert_full_article,
    insert_or_get_journal,
    insert_or_get_author,
    insert_or_get_affiliation,
    link_author_affiliation,
    link_article_author,
    insert_or_get_keyword,
    link_article_keyword,
    insert_or_get_grant,
    link_article_grant,
    insert_or_get_mesh_term,
    insert_or_get_mesh_qualifier,
    link_article_mesh_term,
    link_article_mesh_qualifier,
)
from odda_utils.fetching import search_pubmed, fetch_article_metadata, download_pmc_article
from odda_utils.fetching.pmc import DateType
from odda_utils.metadata import FullArticleMetadata
from odda_utils.metadata.llm_metadata import (
    build_extraction_prompt,
    call_llm,
    store_llm_extraction_record,
    parse_llm_response,
    store_extracted_keywords,
    store_extracted_raw_data,
    store_extracted_processed_data,
    store_extracted_analysis_methods,
    LLMExtractionError,
)
from odda_utils.utils import (
    get_azure_credentials,
    AzureCredentialsError,
    check_existing_article,
    get_text_embedding,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchAndFetchResult:
    """Result of a search_and_fetch operation."""

    total_found: int
    already_processed: int
    newly_processed: int
    overwritten: int
    failed: int
    skipped_no_abstract: int
    downloaded: int = 0
    download_failed: int = 0
    llm_extracted: int = 0
    llm_extraction_failed: int = 0


def insert_article_metadata(
    conn: sqlite3.Connection,
    metadata: FullArticleMetadata,
    article_doi: str,
    article_filepath: str | None = None,
    supplementals_filepath: str | None = None,
) -> None:
    """Insert full article metadata into all database tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    metadata : FullArticleMetadata
        Full article metadata from PubMed.
    article_doi : str
        DOI to use as primary key (may include pmid: prefix).
    article_filepath : str or None
        Path to downloaded article file.
    supplementals_filepath : str or None
        Path to supplementals archive.
    """
    journal_id = None
    if metadata.journal and metadata.journal.name:
        journal_id = insert_or_get_journal(
            conn=conn,
            name=metadata.journal.name,
            issn=metadata.journal.issn,
            eissn=metadata.journal.eissn,
            iso_abbreviation=metadata.journal.iso_abbreviation,
        )

    pub_date_str = None
    if metadata.publication_date:
        pub_date_str = metadata.publication_date.isoformat()

    epub_date_str = None
    if metadata.electronic_publication_date:
        epub_date_str = metadata.electronic_publication_date.isoformat()

    volume = metadata.journal.volume if metadata.journal else None
    issue = metadata.journal.issue if metadata.journal else None

    insert_full_article(
        conn=conn,
        doi=article_doi,
        pmid=metadata.pmid,
        pmcid=metadata.pmcid,
        title=metadata.title,
        abstract=metadata.abstract,
        publication_date=pub_date_str,
        electronic_publication_date=epub_date_str,
        journal_id=journal_id,
        volume=volume,
        issue=issue,
        article_type=metadata.article_type,
        language=metadata.language,
        copyright_text=metadata.copyright,
        article_filepath=article_filepath,
        supplementals_filepath=supplementals_filepath,
    )

    for position, author in enumerate(metadata.authors, start=1):
        author_id = insert_or_get_author(
            conn=conn,
            last_name=author.last_name,
            first_name=author.first_name,
            initials=author.initials,
            orcid=author.orcid,
            is_collective=author.is_collective,
        )

        link_article_author(
            conn=conn,
            author_id=author_id,
            author_position=position,
            doi=article_doi,
            pmid=metadata.pmid,
            pmcid=metadata.pmcid,
        )

        for affiliation_text in author.affiliations:
            affiliation_id = insert_or_get_affiliation(conn, affiliation_text)
            link_author_affiliation(conn, author_id, affiliation_id)

    for keyword in metadata.keywords:
        keyword_id = insert_or_get_keyword(conn, keyword)
        link_article_keyword(
            conn=conn,
            keyword_id=keyword_id,
            doi=article_doi,
            pmid=metadata.pmid,
            pmcid=metadata.pmcid,
        )

    for grant in metadata.grants:
        grant_id = insert_or_get_grant(
            conn=conn,
            grant_number=grant.grant_id,
            acronym=grant.acronym,
            agency=grant.agency,
            country=grant.country,
        )
        link_article_grant(
            conn=conn,
            grant_id=grant_id,
            doi=article_doi,
            pmid=metadata.pmid,
            pmcid=metadata.pmcid,
        )

    for mesh_term in metadata.mesh_terms:
        mesh_term_id = insert_or_get_mesh_term(
            conn=conn,
            descriptor_name=mesh_term.descriptor_name,
            descriptor_ui=mesh_term.descriptor_ui,
        )

        article_mesh_term_id = link_article_mesh_term(
            conn=conn,
            mesh_term_id=mesh_term_id,
            is_major_topic=mesh_term.is_major_topic,
            doi=article_doi,
            pmid=metadata.pmid,
            pmcid=metadata.pmcid,
        )

        for qualifier_name in mesh_term.qualifiers:
            qualifier_id = insert_or_get_mesh_qualifier(conn, qualifier_name)
            link_article_mesh_qualifier(conn, article_mesh_term_id, qualifier_id)


def search_and_fetch(
    db_path: str | Path,
    query: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    date_type: DateType = "edat",
    max_results: int = 100,
    embedding_model: str = "text-embedding-3-small",
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    overwrite: bool = False,
    download_dir: str | Path | None = None,
    extract_llm_metadata: bool = True,
    llm_model: str = "gpt-5",
) -> SearchAndFetchResult:
    """Search PubMed and fetch/process articles that haven't been processed yet.

    This function:
    1. Searches PubMed for articles matching the query
    2. For each article, checks if it's already in the database
    3. If not (or if overwrite=True), fetches metadata and stores it
    4. Extracts the abstract and generates a text embedding
    5. Stores the embedding in the database
    6. If download_dir is provided, downloads full text and supplementals from PMC
    7. If extract_llm_metadata is True, extracts keywords, raw data, processed data,
       and analysis methods from the downloaded full text using an LLM

    Args:
        db_path: Path to the SQLite database file.
        query: PubMed articles query string.
        start_date: Start date for filtering (inclusive).
        end_date: End date for filtering (inclusive).
        date_type: Type of date to filter on ("edat", "pdat", "mdat").
        max_results: Maximum number of results to process.
        embedding_model: Name of the Azure OpenAI embedding model deployment.
        endpoint_file: Path to file containing Azure OpenAI endpoint.
        api_key_file: Path to file containing Azure OpenAI API key.
        overwrite: If True, re-process articles that already exist in the database.
        download_dir: Directory to save downloaded article text and supplementals.
            If None, articles will not be downloaded.
        extract_llm_metadata: If True, use LLM to extract metadata from downloaded
            full text. Requires download_dir to be set.
        llm_model: Name of the Azure OpenAI chat model deployment for LLM extraction.

    Returns:
        SearchAndFetchResult with statistics about the operation.
    """
    conn = init_db(db_path)

    try:
        return _search_and_fetch_impl(
            conn=conn,
            query=query,
            start_date=start_date,
            end_date=end_date,
            date_type=date_type,
            max_results=max_results,
            embedding_model=embedding_model,
            endpoint_file=endpoint_file,
            api_key_file=api_key_file,
            overwrite=overwrite,
            download_dir=download_dir,
            extract_llm_metadata=extract_llm_metadata,
            llm_model=llm_model,
        )
    finally:
        conn.close()


def _search_and_fetch_impl(
    conn: sqlite3.Connection,
    query: str,
    start_date: date | str | None,
    end_date: date | str | None,
    date_type: DateType,
    max_results: int,
    embedding_model: str,
    endpoint_file: str | Path | None,
    api_key_file: str | Path | None,
    overwrite: bool,
    download_dir: str | Path | None,
    extract_llm_metadata: bool,
    llm_model: str,
) -> SearchAndFetchResult:
    """Implementation of search_and_fetch with an existing connection."""
    # Validate Azure credentials before starting the pipeline
    # This ensures we fail fast if credentials are missing
    try:
        get_azure_credentials(endpoint_file, api_key_file)
    except AzureCredentialsError as e:
        raise AzureCredentialsError(
            f"Azure OpenAI credentials required for embedding generation. {e}"
        ) from e

    # Search PubMed
    logger.info("Searching PubMed for: %s", query)
    pmids = search_pubmed(
        query=query,
        start_date=start_date,
        end_date=end_date,
        date_type=date_type,
        max_results=max_results,
    )

    result = SearchAndFetchResult(
        total_found=len(pmids),
        already_processed=0,
        newly_processed=0,
        overwritten=0,
        failed=0,
        skipped_no_abstract=0,
    )

    logger.info("Found %d articles", len(pmids))

    for pmid in pmids:
        # Fetch metadata first to get all identifiers
        try:
            metadata = fetch_article_metadata(pmid)
        except Exception as e:
            logger.warning("Failed to fetch metadata for %s: %s", pmid, e)
            result.failed += 1
            continue

        # Check if already processed by any identifier (DOI, PMID, or PMCID)
        is_existing = check_existing_article(conn, metadata)
        if is_existing and not overwrite:
            logger.debug("Article %s already processed, skipping", pmid)
            result.already_processed += 1
            continue

        # Skip if no abstract available
        if not metadata.abstract:
            logger.debug("Article %s has no abstract, skipping", pmid)
            result.skipped_no_abstract += 1
            continue

        # Use DOI as primary key if available, otherwise use PMID as fallback
        article_doi = metadata.doi or f"pmid:{pmid}"

        # Download full text and supplementals if download_dir is provided
        article_filepath = None
        supplementals_filepath = None
        if download_dir is not None:
            try:
                download_result = download_pmc_article(
                    output_dir=download_dir,
                    pmid=pmid,
                )
                if download_result.error:
                    logger.debug(
                        "Could not download article %s: %s", pmid, download_result.error
                    )
                    result.download_failed += 1
                else:
                    if download_result.text_filepath:
                        article_filepath = str(download_result.text_filepath)
                    if download_result.supplementals_filepath:
                        supplementals_filepath = str(download_result.supplementals_filepath)
                    if article_filepath or supplementals_filepath:
                        logger.info("Downloaded article %s", pmid)
                        result.downloaded += 1
            except Exception as e:
                logger.warning("Failed to download article %s: %s", pmid, e)
                result.download_failed += 1

        # Store article and all metadata in database
        logger.debug("Inserting article metadata: %s", article_doi)
        insert_article_metadata(
            conn=conn,
            metadata=metadata,
            article_doi=article_doi,
            article_filepath=article_filepath,
            supplementals_filepath=supplementals_filepath,
        )

        # Generate embedding for abstract
        try:
            embedding = get_text_embedding(
                text=metadata.abstract,
                endpoint_file=endpoint_file,
                api_key_file=api_key_file,
                deployment_name=embedding_model,
            )
        except Exception as e:
            logger.warning("Failed to generate embedding for %s: %s", pmid, e)
            result.failed += 1
            continue

        # Store embedding
        insert_embedding(
            conn=conn,
            embedding=embedding,
            model=embedding_model,
            doi=article_doi,
            pmid=metadata.pmid,
            pmcid=metadata.pmcid,
        )

        # Extract LLM metadata if enabled and full text is available
        if extract_llm_metadata and article_filepath:
            try:
                # Read the article text
                article_text = Path(article_filepath).read_text(encoding="utf-8")

                # Get Azure credentials
                endpoint, api_key = get_azure_credentials(endpoint_file, api_key_file)

                # Build prompt and call LLM
                prompt = build_extraction_prompt(article_text)
                logger.info("Extracting LLM metadata for %s", pmid)
                response = call_llm(
                    prompt=prompt,
                    endpoint=endpoint,
                    api_key=api_key,
                    model=llm_model,
                )

                # Store the prompt and response
                extraction_id = store_llm_extraction_record(
                    conn=conn,
                    prompt=prompt,
                    response=response,
                    model=llm_model,
                    doi=article_doi,
                    pmid=metadata.pmid,
                    pmcid=metadata.pmcid,
                )
                logger.debug("Stored extraction record with ID %d for %s", extraction_id, pmid)

                # Parse and store extracted data
                extracted = parse_llm_response(response, llm_model)

                if extracted.keywords:
                    store_extracted_keywords(
                        conn,
                        extracted.keywords,
                        llm_model,
                        doi=article_doi,
                        pmid=metadata.pmid,
                        pmcid=metadata.pmcid,
                    )
                    logger.debug("Stored %d LLM keywords for %s", len(extracted.keywords), pmid)

                if extracted.raw_data:
                    store_extracted_raw_data(
                        conn,
                        extracted.raw_data,
                        llm_model,
                        doi=article_doi,
                        pmid=metadata.pmid,
                        pmcid=metadata.pmcid,
                    )
                    logger.debug("Stored %d raw data entries for %s", len(extracted.raw_data), pmid)

                if extracted.processed_data:
                    store_extracted_processed_data(
                        conn,
                        extracted.processed_data,
                        llm_model,
                        doi=article_doi,
                        pmid=metadata.pmid,
                        pmcid=metadata.pmcid,
                    )
                    logger.debug("Stored %d processed data entries for %s", len(extracted.processed_data), pmid)

                if extracted.analysis_methods:
                    store_extracted_analysis_methods(
                        conn,
                        extracted.analysis_methods,
                        llm_model,
                        doi=article_doi,
                        pmid=metadata.pmid,
                        pmcid=metadata.pmcid,
                    )
                    logger.debug("Stored %d analysis methods for %s", len(extracted.analysis_methods), pmid)

                result.llm_extracted += 1
                logger.info("Extracted LLM metadata for %s", pmid)

            except LLMExtractionError as e:
                logger.warning("LLM extraction failed for %s: %s", pmid, e)
                result.llm_extraction_failed += 1
            except Exception as e:
                logger.warning("Unexpected error during LLM extraction for %s: %s", pmid, e)
                result.llm_extraction_failed += 1

        if is_existing:
            logger.info("Overwrote article %s: %s", pmid, metadata.title)
            result.overwritten += 1
        else:
            logger.info("Processed article %s: %s", pmid, metadata.title)
            result.newly_processed += 1

    return result
