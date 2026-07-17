"""Utility functions for ID conversion, Azure credentials, text embeddings,
and article search including both unfiltered and metadata-filtered
embedding-based similarity search.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import requests
import numpy as np

if TYPE_CHECKING:
    from odda_utils.metadata import FullArticleMetadata

from odda_utils.database import (
    get_article,
    get_article_by_pmid,
    get_article_by_pmcid,
    _blob_to_embedding,
)
from odda_utils.metadata import logger

NCBI_ID_CONVERTER_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

_search_logger = logging.getLogger(__name__)


@dataclass
class ArticleIds:
    """Container for article identifiers."""

    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None

    @classmethod
    def from_doi(cls, doi: str) -> Self:
        """Create ArticleIds from a DOI, fetching other IDs from NCBI."""
        return _convert_id(doi=doi)

    @classmethod
    def from_pmid(cls, pmid: str) -> Self:
        """Create ArticleIds from a PMID, fetching other IDs from NCBI."""
        return _convert_id(pmid=pmid)

    @classmethod
    def from_pmcid(cls, pmcid: str) -> Self:
        """Create ArticleIds from a PMCID, fetching other IDs from NCBI."""
        return _convert_id(pmcid=pmcid)

    def has_any_id(self) -> bool:
        """Check if at least one ID is present."""
        return any([self.doi, self.pmid, self.pmcid])


def _convert_id(
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> ArticleIds:
    """Convert between article identifiers using NCBI ID Converter API.

    Provide exactly one of doi, pmid, or pmcid.

    Args:
        doi: Digital Object Identifier.
        pmid: PubMed ID.
        pmcid: PubMed Central ID.

    Returns:
        ArticleIds with all available identifiers.

    Raises:
        ValueError: If no ID provided or multiple IDs provided.
        requests.HTTPError: If the API request fails.
    """
    ids = [x for x in [doi, pmid, pmcid] if x is not None]
    if len(ids) != 1:
        raise ValueError("Provide exactly one of doi, pmid, or pmcid")

    if doi:
        id_value = doi
        id_type = "doi"
    elif pmid:
        id_value = pmid
        id_type = "pmid"
    else:
        id_value = pmcid
        id_type = "pmcid"

    params = {
        "ids": id_value,
        "idtype": id_type,
        "format": "json",
        "tool": "odda",
        "email": "user@example.com",
    }

    response = requests.get(NCBI_ID_CONVERTER_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    records = data.get("records", [])
    if not records:
        return ArticleIds(**{id_type: id_value})

    record = records[0]

    if "errmsg" in record:
        return ArticleIds(**{id_type: id_value})

    return ArticleIds(
        doi=record.get("doi"),
        pmid=record.get("pmid"),
        pmcid=record.get("pmcid"),
    )


def convert_ids(
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> ArticleIds:
    """Convert between article identifiers.

    Provide exactly one of doi, pmid, or pmcid.

    Args:
        doi: Digital Object Identifier.
        pmid: PubMed ID.
        pmcid: PubMed Central ID.

    Returns:
        ArticleIds with all available identifiers.
    """
    return _convert_id(doi=doi, pmid=pmid, pmcid=pmcid)


class AzureCredentialsError(Exception):
    """Raised when Azure credentials cannot be found."""

    pass


def get_azure_credentials(
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
) -> tuple[str, str]:
    """Get Azure OpenAI credentials from files or environment variables.

    Checks for credentials in the following order:
    1. If file paths are provided, read from files
    2. Fall back to environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY

    Args:
        endpoint_file: Path to file containing the Azure OpenAI endpoint URL.
        api_key_file: Path to file containing the Azure OpenAI API key.

    Returns:
        Tuple of (endpoint, api_key).

    Raises:
        AzureCredentialsError: If credentials cannot be found.
    """
    endpoint = None
    api_key = None

    # Try to read from files if provided
    if endpoint_file is not None:
        endpoint_path = Path(endpoint_file).expanduser()
        if endpoint_path.exists():
            endpoint = endpoint_path.read_text().strip()
        else:
            raise AzureCredentialsError(f"Endpoint file not found: {endpoint_file}")

    if api_key_file is not None:
        api_key_path = Path(api_key_file).expanduser()
        if api_key_path.exists():
            api_key = api_key_path.read_text().strip()
        else:
            raise AzureCredentialsError(f"API key file not found: {api_key_file}")

    # Fall back to environment variables for any missing values
    if endpoint is None:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if endpoint is None:
            raise AzureCredentialsError(
                "Azure endpoint not found. Provide endpoint_file or set AZURE_OPENAI_ENDPOINT."
            )

    if api_key is None:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise AzureCredentialsError(
                "Azure API key not found. Provide api_key_file or set AZURE_OPENAI_API_KEY."
            )

    return endpoint, api_key


def get_text_embedding(
    text: str,
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    deployment_name: str = "text-embedding-3-small",
    api_version: str = "2024-02-01",
) -> list[float]:
    """Get a text embedding via the configured embedding provider.

    Delegates to the provider-agnostic :mod:`odda_utils.llm` abstraction. The
    ``endpoint_file``, ``api_key_file``, ``deployment_name`` and ``api_version``
    arguments are Azure-OpenAI hints, preserved for backward compatibility; they
    are honoured only when the resolved embedding provider is ``azure_openai``.

    Args:
        text: The text to embed.
        endpoint_file: Path to file containing the Azure OpenAI endpoint URL.
        api_key_file: Path to file containing the Azure OpenAI API key.
        deployment_name: Name of the embedding model deployment (azure_openai).
        api_version: Azure OpenAI API version.

    Returns:
        List of floats representing the embedding vector.

    Raises:
        odda_utils.llm.ModelConfigError: If no embedding provider is configured.
        odda_utils.llm.LLMProviderError: If the embedding request fails.
    """
    # Imported lazily to avoid a circular import (llm imports from utils).
    from odda_utils import llm

    result = llm.embed(
        text,
        endpoint_file=endpoint_file,
        api_key_file=api_key_file,
        model=deployment_name,
        api_version=api_version,
    )
    return result.vector


def check_existing_article(
    conn: sqlite3.Connection,
    metadata: "FullArticleMetadata",
) -> bool:
    """Check if an article already exists in the database by any identifier.

    Checks all three identifiers (DOI, PMID, PMCID) and warns if there are
    mismatches between the metadata and what's stored in the database.

    Args:
        conn: Database connection.
        metadata: Article metadata with DOI, PMID, and PMCID.

    Returns:
        True if the article already exists in the database, False otherwise.
    """
    existing_records = []

    # Check by DOI
    if metadata.doi:
        existing_by_doi = get_article(conn, metadata.doi)
        if existing_by_doi:
            existing_records.append(("doi", metadata.doi, existing_by_doi))

    # Check by PMID
    if metadata.pmid:
        existing_by_pmid = get_article_by_pmid(conn, metadata.pmid)
        if existing_by_pmid:
            existing_records.append(("pmid", metadata.pmid, existing_by_pmid))

    # Check by PMCID
    if metadata.pmcid:
        existing_by_pmcid = get_article_by_pmcid(conn, metadata.pmcid)
        if existing_by_pmcid:
            existing_records.append(("pmcid", metadata.pmcid, existing_by_pmcid))

    if not existing_records:
        return False

    # Article exists - check for consistency
    _check_id_consistency(metadata, existing_records)

    return True


def _check_id_consistency(
    metadata: "FullArticleMetadata",
    existing_records: list[tuple[str, str, sqlite3.Row]],
) -> None:
    """Check for ID consistency between metadata and database records.

    Args:
        metadata: Article metadata from PubMed.
        existing_records: List of (id_type, id_value, db_record) tuples.
    """
    # Get the first existing record as reference
    _, _, ref_record = existing_records[0]

    # Check if all found records point to the same article
    for id_type, id_value, record in existing_records:
        if ref_record["doi"] != record["doi"]:
            logger.warning(
                "ID mismatch for article %s=%s: found different DOIs in database "
                "(%s vs %s)",
                id_type,
                id_value,
                ref_record["doi"],
                record["doi"],
            )

    # Check if metadata IDs match the database record
    if metadata.doi and ref_record["doi"] and metadata.doi != ref_record["doi"]:
        logger.warning(
            "DOI mismatch for PMID %s: metadata has %s, database has %s",
            metadata.pmid,
            metadata.doi,
            ref_record["doi"],
        )

    if metadata.pmid and ref_record["pmid"] and metadata.pmid != ref_record["pmid"]:
        logger.warning(
            "PMID mismatch for DOI %s: metadata has %s, database has %s",
            metadata.doi,
            metadata.pmid,
            ref_record["pmid"],
        )

    if metadata.pmcid and ref_record["pmcid"] and metadata.pmcid != ref_record["pmcid"]:
        logger.warning(
            "PMCID mismatch for PMID %s: metadata has %s, database has %s",
            metadata.pmid,
            metadata.pmcid,
            ref_record["pmcid"],
        )


def search_articles_by_embedding(
    query: str,
    conn: sqlite3.Connection,
    top_k: int = 5,
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
) -> list[dict]:
    """Search for articles similar to a query using dot product of embeddings.

    Args:
        query: Search query text.
        conn: Database connection.
        top_k: Number of top results to return.
        endpoint_file: Path to file containing the Azure OpenAI endpoint URL.
        api_key_file: Path to file containing the Azure OpenAI API key.

    Returns:
        List of dictionaries containing article info and similarity score.
    """
    query_embedding = get_text_embedding(
        query, endpoint_file=endpoint_file, api_key_file=api_key_file
    )
    query_embedding = np.array(query_embedding)
    cursor = conn.execute(
        "SELECT doi, pmid, pmcid, embedding FROM embeddings"
    )
    results = []
    for row in cursor:
        article_embedding = np.array(_blob_to_embedding(row["embedding"]))

        # Compute dot product
        score = query_embedding.T @ article_embedding
        results.append({
            "doi": row["doi"],
            "pmid": row["pmid"],
            "pmcid": row["pmcid"],
            "score": score
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def _build_filtered_doi_query(
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
) -> tuple[str, list]:
    """Build a SQL query to filter articles based on metadata criteria.

    Constructs a query that returns DOIs of articles matching all specified
    filters. Each filter is applied as an additional condition (AND logic),
    but within a filter that accepts a list (e.g. keywords, authors), items
    are combined with OR logic.

    Parameters
    ----------
    keywords : list of str, optional
        Keywords to match against the keywords and llm_keywords tables.
        Articles matching any of the provided keywords are included.
    has_dataset : bool, optional
        If True, only include articles that have at least one associated
        dataset in llm_raw_data or llm_processed_data. If False, only
        include articles without datasets.
    has_quantified_data : bool, optional
        If True, only include articles that have quantified/processed data
        in llm_processed_data. If False, only include articles without.
    authors : list of str, optional
        Author names to match (partial, case-insensitive). Articles with
        any matching author are included.
    article_type : str, optional
        Filter by article type (exact match).
    journal_name : str, optional
        Filter by journal name (partial, case-insensitive).
    publication_date_from : str, optional
        Include articles published on or after this date (YYYY-MM-DD).
    publication_date_to : str, optional
        Include articles published on or before this date (YYYY-MM-DD).
    mesh_terms : list of str, optional
        MeSH descriptor names to match (partial, case-insensitive).
        Articles matching any of the provided terms are included.
    language : str, optional
        Filter by article language (exact match).

    Returns
    -------
    tuple of (str, list)
        A tuple of (SQL query string, parameter list) that selects DOIs
        of matching articles.
    """
    conditions: list[str] = []
    params: list = []

    # Keyword filter: match against keywords or llm_keywords tables using UNION.
    # The UNION merges DOIs from both keyword tables, and each SELECT within
    # the UNION needs its own set of parameter placeholders.
    if keywords:
        kw_placeholders = ", ".join(["?"] * len(keywords))
        lowered_keywords = [kw.lower() for kw in keywords]
        conditions.append(f"""
            a.doi IN (
                SELECT ak.doi FROM article_keywords ak
                JOIN keywords k ON ak.keyword_id = k.keyword_id
                WHERE LOWER(k.keyword) IN ({kw_placeholders})
                AND ak.doi IS NOT NULL
                UNION
                SELECT alk.doi FROM article_llm_keywords alk
                JOIN llm_keywords lk ON alk.llm_keyword_id = lk.llm_keyword_id
                WHERE LOWER(lk.keyword) IN ({kw_placeholders})
                AND alk.doi IS NOT NULL
            )
        """)
        params.extend(lowered_keywords)  # for the article_keywords subquery
        params.extend(lowered_keywords)  # for the article_llm_keywords subquery

    # Has dataset filter
    if has_dataset is True:
        conditions.append("""
            a.doi IN (
                SELECT lrd.doi FROM llm_raw_data lrd WHERE lrd.doi IS NOT NULL
                UNION
                SELECT lpd.doi FROM llm_processed_data lpd WHERE lpd.doi IS NOT NULL
            )
        """)
    elif has_dataset is False:
        conditions.append("""
            a.doi NOT IN (
                SELECT lrd.doi FROM llm_raw_data lrd WHERE lrd.doi IS NOT NULL
                UNION
                SELECT lpd.doi FROM llm_processed_data lpd WHERE lpd.doi IS NOT NULL
            )
        """)

    # Has quantified data filter
    if has_quantified_data is True:
        conditions.append("""
            a.doi IN (
                SELECT lpd.doi FROM llm_processed_data lpd WHERE lpd.doi IS NOT NULL
            )
        """)
    elif has_quantified_data is False:
        conditions.append("""
            a.doi NOT IN (
                SELECT lpd.doi FROM llm_processed_data lpd WHERE lpd.doi IS NOT NULL
            )
        """)

    # Author filter: partial name match using LIKE on a combined name string.
    # COALESCE handles NULL first/last names. A single LIKE per author against
    # the concatenated "first last" string catches matches in either part.
    if authors:
        author_like_clauses = []
        for author_name in authors:
            author_like_clauses.append(
                "LOWER(COALESCE(ai.first_name, '') || ' ' || "
                "COALESCE(ai.last_name, '')) LIKE ?"
            )
            params.append(f"%{author_name.lower()}%")
        conditions.append(f"""
            a.doi IN (
                SELECT aa.doi FROM article_authors aa
                JOIN author_info ai ON aa.author_id = ai.author_id
                WHERE ({" OR ".join(author_like_clauses)})
                AND aa.doi IS NOT NULL
            )
        """)

    # Article type filter
    if article_type:
        conditions.append("a.article_type = ?")
        params.append(article_type)

    # Journal name filter (partial, case-insensitive)
    if journal_name:
        conditions.append("""
            a.journal_id IN (
                SELECT j.journal_id FROM journals j
                WHERE LOWER(j.name) LIKE ?
            )
        """)
        params.append(f"%{journal_name.lower()}%")

    # Publication date range filters
    if publication_date_from:
        conditions.append("a.publication_date >= ?")
        params.append(publication_date_from)

    if publication_date_to:
        conditions.append("a.publication_date <= ?")
        params.append(publication_date_to)

    # MeSH terms filter
    if mesh_terms:
        mesh_like_clauses = []
        for term in mesh_terms:
            mesh_like_clauses.append("LOWER(mt.descriptor_name) LIKE ?")
            params.append(f"%{term.lower()}%")
        conditions.append(f"""
            a.doi IN (
                SELECT amt.doi FROM article_mesh_terms amt
                JOIN mesh_terms mt ON amt.mesh_term_id = mt.mesh_term_id
                WHERE ({" OR ".join(mesh_like_clauses)})
                AND amt.doi IS NOT NULL
            )
        """)

    # Language filter
    if language:
        conditions.append("a.language = ?")
        params.append(language)

    # Build the final query
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT DISTINCT a.doi
        FROM articles a
        WHERE a.doi IS NOT NULL
        AND {where_clause}
    """

    return query, params


def search_articles_by_embedding_filtered(
    query: str,
    conn: sqlite3.Connection,
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

    First filters articles based on the provided metadata criteria, then
    retrieves embeddings for the matching articles and computes the dot
    product similarity between the query embedding and each article
    embedding. Embeddings are assumed to be unit length, so the dot product
    is equivalent to cosine similarity.

    Parameters
    ----------
    query : str
        Search query text that will be embedded and compared against
        article embeddings.
    conn : sqlite3.Connection
        Database connection.
    top_k : int, optional
        Number of top results to return. Default is 5.
    endpoint_file : str or Path, optional
        Path to file containing the Azure OpenAI endpoint URL.
    api_key_file : str or Path, optional
        Path to file containing the Azure OpenAI API key.
    keywords : list of str, optional
        Keywords to filter by. Matches against both article keywords and
        LLM-extracted keywords. Articles matching any keyword are included.
    has_dataset : bool, optional
        If True, only include articles with associated datasets. If False,
        only include articles without datasets. If None, no filtering.
    has_quantified_data : bool, optional
        If True, only include articles with quantified/processed data.
        If False, only include articles without. If None, no filtering.
    authors : list of str, optional
        Author names to filter by (partial, case-insensitive match).
        Articles with any matching author are included.
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
        Articles matching any term are included.
    language : str, optional
        Filter by article language (exact match, e.g. "eng").

    Returns
    -------
    list of dict
        List of dictionaries sorted by similarity score (descending),
        each containing:
        - doi: Article DOI
        - pmid: Article PMID
        - pmcid: Article PMCID
        - title: Article title
        - score: Dot product similarity score (float)
    """
    # Step 1: Build the filter query to get matching article DOIs
    filter_query, filter_params = _build_filtered_doi_query(
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

    # Step 2: Execute the filter query to get matching DOIs
    _search_logger.debug("Executing filter query: %s", filter_query)
    cursor = conn.execute(filter_query, filter_params)
    filtered_dois = [row["doi"] for row in cursor]

    _search_logger.info(
        "Metadata filters matched %d articles", len(filtered_dois)
    )

    if not filtered_dois:
        return []

    # Step 3: Get embeddings only for the filtered articles
    doi_placeholders = ", ".join(["?"] * len(filtered_dois))
    embedding_query = f"""
        SELECT e.doi, e.pmid, e.pmcid, e.embedding
        FROM embeddings e
        WHERE e.doi IN ({doi_placeholders})
    """
    cursor = conn.execute(embedding_query, filtered_dois)
    embedding_rows = cursor.fetchall()

    _search_logger.info(
        "Found embeddings for %d of %d filtered articles",
        len(embedding_rows),
        len(filtered_dois),
    )

    if not embedding_rows:
        return []

    # Step 4: Get the query embedding
    query_embedding = get_text_embedding(
        query, endpoint_file=endpoint_file, api_key_file=api_key_file
    )
    query_vec = np.array(query_embedding)

    # Step 5: Compute dot product similarity for each filtered article.
    # Dot product is equivalent to cosine similarity for unit-length vectors.
    results = []
    for row in embedding_rows:
        article_embedding = np.array(_blob_to_embedding(row["embedding"]))
        score = float(query_vec @ article_embedding)
        results.append({
            "doi": row["doi"],
            "pmid": row["pmid"],
            "pmcid": row["pmcid"],
            "score": score,
        })

    # Step 6: Sort by score descending and take top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k]

    # Step 7: Enrich results with article title
    for result in top_results:
        article = get_article(conn, result["doi"])
        if article:
            result["title"] = article["title"]
        else:
            result["title"] = None

    return top_results
