"""Database module for tracking downloaded scientific articles and datasets.

This module provides functions for initializing the SQLite database and performing
CRUD operations on articles, embeddings, authors, affiliations, journals, keywords,
grants, MeSH terms, LLM extractions, supplemental file classifications, datasets,
and agent requests. Agent requests support status tracking including 'in_progress'
status with assigned_time timestamp.

It also provides the provenance / research-object layer (Phase 2): insert and
query helpers for quantification_runs, analysis_runs, dep_results,
benchmark_annotations, and benchmark_predictions. These record full provenance
(tool/library versions, container and parameter hashes, commands, hosts, and
model/provider) so every quantification/analysis result is reproducible. List
and dict values are stored in JSON TEXT columns via the ``_encode_json`` /
``_decode_json`` helpers.
"""

import json
import re
import sqlite3
import struct
from importlib.resources import files
from pathlib import Path

_SCHEMA_PATH = files("odda_utils.static").joinpath("schema.sql")


def _strip_html_tags(text: str | None) -> str | None:
    """Remove HTML tags from text.

    Parameters
    ----------
    text : str, optional
        Text potentially containing HTML tags.

    Returns
    -------
    str or None
        Text with HTML tags removed, or None if input is None.
    """
    if text is None:
        return None
    # Remove HTML tags like <sup>, </sup>, <sub>, </sub>, <i>, </i>, etc.
    return re.sub(r"<[^>]+>", "", text)


def _load_schema() -> str:
    """Load the SQL schema from the static file."""
    return _SCHEMA_PATH.read_text()


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialize the database and create tables if they don't exist.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Database connection.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_load_schema())
    conn.commit()
    return conn


def insert_article(
    conn: sqlite3.Connection,
    doi: str,
    pmid: str | None = None,
    pmcid: str | None = None,
    title: str | None = None,
    article_filepath: str | None = None,
    supplementals_filepath: str | None = None,
) -> None:
    """Insert or replace an article record.

    Args:
        conn: Database connection.
        doi: Digital Object Identifier (primary key).
        pmid: PubMed ID.
        pmcid: PubMed Central ID.
        title: Article title (HTML tags will be stripped).
        article_filepath: Path to downloaded article file.
        supplementals_filepath: Path to supplementals archive.
    """
    # Strip HTML tags from title
    clean_title = _strip_html_tags(title)

    conn.execute(
        """
        INSERT INTO articles (doi, pmid, pmcid, title, article_filepath, supplementals_filepath, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(doi) DO UPDATE SET
            pmid = COALESCE(excluded.pmid, pmid),
            pmcid = COALESCE(excluded.pmcid, pmcid),
            title = CASE
                WHEN excluded.title IS NULL THEN title
                WHEN title IS NULL THEN excluded.title
                WHEN LENGTH(excluded.title) > LENGTH(title) THEN excluded.title
                ELSE title
            END,
            article_filepath = COALESCE(excluded.article_filepath, article_filepath),
            supplementals_filepath = COALESCE(excluded.supplementals_filepath, supplementals_filepath),
            updated_at = CURRENT_TIMESTAMP
        """,
        (doi, pmid, pmcid, clean_title, article_filepath, supplementals_filepath),
    )
    conn.commit()


def get_article(conn: sqlite3.Connection, doi: str) -> sqlite3.Row | None:
    """Retrieve an article by DOI.

    Args:
        conn: Database connection.
        doi: Digital Object Identifier.

    Returns:
        Article row or None if not found.
    """
    cursor = conn.execute("SELECT * FROM articles WHERE doi = ?", (doi,))
    return cursor.fetchone()


def get_article_by_pmid(conn: sqlite3.Connection, pmid: str) -> sqlite3.Row | None:
    """Retrieve an article by PMID.

    Args:
        conn: Database connection.
        pmid: PubMed ID.

    Returns:
        Article row or None if not found.
    """
    cursor = conn.execute("SELECT * FROM articles WHERE pmid = ?", (pmid,))
    return cursor.fetchone()


def get_article_by_pmcid(conn: sqlite3.Connection, pmcid: str) -> sqlite3.Row | None:
    """Retrieve an article by PMCID.

    Args:
        conn: Database connection.
        pmcid: PubMed Central ID.

    Returns:
        Article row or None if not found.
    """
    cursor = conn.execute("SELECT * FROM articles WHERE pmcid = ?", (pmcid,))
    return cursor.fetchone()


def _embedding_to_blob(embedding: list[float]) -> bytes:
    """Convert an embedding list to a binary blob.

    Args:
        embedding: List of floats.

    Returns:
        Binary representation of the embedding.
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes) -> list[float]:
    """Convert a binary blob back to an embedding list.

    Args:
        blob: Binary representation of the embedding.

    Returns:
        List of floats.
    """
    count = len(blob) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", blob))


def insert_embedding(
    conn: sqlite3.Connection,
    embedding: list[float],
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> None:
    """Insert or replace an embedding record.

    Args:
        conn: Database connection.
        embedding: List of floats representing the embedding vector.
        model: Name of the model used to generate the embedding.
        doi: Digital Object Identifier.
        pmid: PubMed ID.
        pmcid: PubMed Central ID.
    """
    blob = _embedding_to_blob(embedding)
    conn.execute(
        """
        INSERT INTO embeddings (doi, pmid, pmcid, embedding, model)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(doi) DO UPDATE SET
            pmid = COALESCE(excluded.pmid, pmid),
            pmcid = COALESCE(excluded.pmcid, pmcid),
            embedding = excluded.embedding,
            model = excluded.model
        """,
        (doi, pmid, pmcid, blob, model),
    )
    conn.commit()


def get_embedding(conn: sqlite3.Connection, doi: str) -> list[float] | None:
    """Retrieve an embedding by DOI.

    Args:
        conn: Database connection.
        doi: Digital Object Identifier.

    Returns:
        Embedding as list of floats, or None if not found.
    """
    cursor = conn.execute("SELECT embedding FROM embeddings WHERE doi = ?", (doi,))
    row = cursor.fetchone()
    if row is None:
        return None
    return _blob_to_embedding(row["embedding"])


def get_embedding_by_pmid(conn: sqlite3.Connection, pmid: str) -> list[float] | None:
    """Retrieve an embedding by PMID.

    Args:
        conn: Database connection.
        pmid: PubMed ID.

    Returns:
        Embedding as list of floats, or None if not found.
    """
    cursor = conn.execute("SELECT embedding FROM embeddings WHERE pmid = ?", (pmid,))
    row = cursor.fetchone()
    if row is None:
        return None
    return _blob_to_embedding(row["embedding"])


def insert_or_get_journal(
    conn: sqlite3.Connection,
    name: str,
    issn: str | None = None,
    eissn: str | None = None,
    iso_abbreviation: str | None = None,
) -> int:
    """Insert a journal or get existing journal_id.

    Args:
        conn: Database connection.
        name: Journal name.
        issn: Print ISSN.
        eissn: Electronic ISSN.
        iso_abbreviation: ISO abbreviation.

    Returns:
        journal_id of the inserted or existing journal.
    """
    # Try to find existing journal by name
    cursor = conn.execute("SELECT journal_id FROM journals WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row["journal_id"]

    # Insert new journal
    cursor = conn.execute(
        """
        INSERT INTO journals (name, issn, eissn, iso_abbreviation)
        VALUES (?, ?, ?, ?)
        """,
        (name, issn, eissn, iso_abbreviation),
    )
    conn.commit()
    return cursor.lastrowid


def insert_or_get_author(
    conn: sqlite3.Connection,
    last_name: str | None,
    first_name: str | None = None,
    initials: str | None = None,
    orcid: str | None = None,
    is_collective: bool = False,
) -> int:
    """Insert an author or get existing author_id.

    Args:
        conn: Database connection.
        last_name: Author's last name (or collective name).
        first_name: Author's first name.
        initials: Author's initials.
        orcid: Author's ORCID.
        is_collective: Whether this is a collective/organization name.

    Returns:
        author_id of the inserted or existing author.
    """
    # Try to find existing author by ORCID if available
    if orcid:
        cursor = conn.execute(
            "SELECT author_id FROM author_info WHERE author_orcid = ?", (orcid,)
        )
        row = cursor.fetchone()
        if row:
            return row["author_id"]

    # Try to find by name combination
    cursor = conn.execute(
        """
        SELECT author_id FROM author_info
        WHERE last_name = ? AND (first_name = ? OR (first_name IS NULL AND ? IS NULL))
        AND (initials = ? OR (initials IS NULL AND ? IS NULL))
        AND (author_orcid IS NULL OR ? IS NULL)
        """,
        (last_name, first_name, first_name, initials, initials, orcid),
    )
    row = cursor.fetchone()
    if row:
        return row["author_id"]

    # Insert new author
    cursor = conn.execute(
        """
        INSERT INTO author_info (first_name, last_name, initials, author_orcid, is_collective)
        VALUES (?, ?, ?, ?, ?)
        """,
        (first_name, last_name, initials, orcid, is_collective),
    )
    conn.commit()
    return cursor.lastrowid


def insert_or_get_affiliation(conn: sqlite3.Connection, affiliation_text: str) -> int:
    """Insert an affiliation or get existing affiliation_id.

    Args:
        conn: Database connection.
        affiliation_text: Full affiliation text.

    Returns:
        affiliation_id of the inserted or existing affiliation.
    """
    cursor = conn.execute(
        "SELECT affiliation_id FROM affiliations WHERE affiliation_text = ?",
        (affiliation_text,),
    )
    row = cursor.fetchone()
    if row:
        return row["affiliation_id"]

    cursor = conn.execute(
        "INSERT INTO affiliations (affiliation_text) VALUES (?)",
        (affiliation_text,),
    )
    conn.commit()
    return cursor.lastrowid


def link_author_affiliation(
    conn: sqlite3.Connection, author_id: int, affiliation_id: int
) -> None:
    """Link an author to an affiliation.

    Args:
        conn: Database connection.
        author_id: Author ID.
        affiliation_id: Affiliation ID.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO author_affiliations (author_id, affiliation_id)
        VALUES (?, ?)
        """,
        (author_id, affiliation_id),
    )
    conn.commit()


def link_article_author(
    conn: sqlite3.Connection,
    author_id: int,
    author_position: int,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    is_corresponding: bool = False,
) -> None:
    """Link an author to an article.

    Args:
        conn: Database connection.
        author_id: Author ID.
        author_position: Position in author list (1-indexed).
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.
        is_corresponding: Whether this is the corresponding author.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO article_authors (doi, pmid, pmcid, author_id, author_position, is_corresponding)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, author_id, author_position, is_corresponding),
    )
    conn.commit()


def insert_or_get_keyword(conn: sqlite3.Connection, keyword: str) -> int:
    """Insert a keyword or get existing keyword_id.

    Args:
        conn: Database connection.
        keyword: Keyword text.

    Returns:
        keyword_id of the inserted or existing keyword.
    """
    cursor = conn.execute(
        "SELECT keyword_id FROM keywords WHERE keyword = ?", (keyword,)
    )
    row = cursor.fetchone()
    if row:
        return row["keyword_id"]

    cursor = conn.execute("INSERT INTO keywords (keyword) VALUES (?)", (keyword,))
    conn.commit()
    return cursor.lastrowid


def link_article_keyword(
    conn: sqlite3.Connection,
    keyword_id: int,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> None:
    """Link a keyword to an article.

    Args:
        conn: Database connection.
        keyword_id: Keyword ID.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO article_keywords (doi, pmid, pmcid, keyword_id)
        VALUES (?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, keyword_id),
    )
    conn.commit()


def insert_or_get_llm_keyword(
    conn: sqlite3.Connection,
    keyword: str,
    model: str,
) -> int:
    """Insert an LLM-generated keyword or get existing llm_keyword_id.

    Args:
        conn: Database connection.
        keyword: Keyword text.
        model: Name of the LLM model that generated the keyword.

    Returns:
        llm_keyword_id of the inserted or existing keyword.
    """
    cursor = conn.execute(
        "SELECT llm_keyword_id FROM llm_keywords WHERE keyword = ? AND model = ?",
        (keyword, model),
    )
    row = cursor.fetchone()
    if row:
        return row["llm_keyword_id"]

    cursor = conn.execute(
        "INSERT INTO llm_keywords (keyword, model) VALUES (?, ?)",
        (keyword, model),
    )
    conn.commit()
    return cursor.lastrowid


def link_article_llm_keyword(
    conn: sqlite3.Connection,
    llm_keyword_id: int,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> None:
    """Link an LLM-generated keyword to an article.

    Args:
        conn: Database connection.
        llm_keyword_id: LLM keyword ID.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO article_llm_keywords (doi, pmid, pmcid, llm_keyword_id)
        VALUES (?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, llm_keyword_id),
    )
    conn.commit()


def insert_or_get_grant(
    conn: sqlite3.Connection,
    grant_number: str | None = None,
    acronym: str | None = None,
    agency: str | None = None,
    country: str | None = None,
) -> int:
    """Insert a grant or get existing grant_id.

    Args:
        conn: Database connection.
        grant_number: Grant number/ID.
        acronym: Grant acronym.
        agency: Funding agency.
        country: Country.

    Returns:
        grant_id of the inserted or existing grant.
    """
    # Try to find existing grant by number and agency
    if grant_number and agency:
        cursor = conn.execute(
            "SELECT grant_id FROM grants WHERE grant_number = ? AND agency = ?",
            (grant_number, agency),
        )
        row = cursor.fetchone()
        if row:
            return row["grant_id"]

    cursor = conn.execute(
        """
        INSERT INTO grants (grant_number, acronym, agency, country)
        VALUES (?, ?, ?, ?)
        """,
        (grant_number, acronym, agency, country),
    )
    conn.commit()
    return cursor.lastrowid


def link_article_grant(
    conn: sqlite3.Connection,
    grant_id: int,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> None:
    """Link a grant to an article.

    Args:
        conn: Database connection.
        grant_id: Grant ID.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO article_grants (doi, pmid, pmcid, grant_id)
        VALUES (?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, grant_id),
    )
    conn.commit()


def insert_or_get_mesh_term(
    conn: sqlite3.Connection,
    descriptor_name: str,
    descriptor_ui: str | None = None,
) -> int:
    """Insert a MeSH term or get existing mesh_term_id.

    Args:
        conn: Database connection.
        descriptor_name: MeSH descriptor name.
        descriptor_ui: MeSH descriptor UI.

    Returns:
        mesh_term_id of the inserted or existing MeSH term.
    """
    # Try to find by UI first if available
    if descriptor_ui:
        cursor = conn.execute(
            "SELECT mesh_term_id FROM mesh_terms WHERE descriptor_ui = ?",
            (descriptor_ui,),
        )
        row = cursor.fetchone()
        if row:
            return row["mesh_term_id"]

    # Try to find by name
    cursor = conn.execute(
        "SELECT mesh_term_id FROM mesh_terms WHERE descriptor_name = ? AND descriptor_ui IS NULL",
        (descriptor_name,),
    )
    row = cursor.fetchone()
    if row:
        return row["mesh_term_id"]

    cursor = conn.execute(
        """
        INSERT INTO mesh_terms (descriptor_name, descriptor_ui)
        VALUES (?, ?)
        """,
        (descriptor_name, descriptor_ui),
    )
    conn.commit()
    return cursor.lastrowid


def insert_or_get_mesh_qualifier(conn: sqlite3.Connection, qualifier_name: str) -> int:
    """Insert a MeSH qualifier or get existing qualifier_id.

    Args:
        conn: Database connection.
        qualifier_name: Qualifier name.

    Returns:
        qualifier_id of the inserted or existing qualifier.
    """
    cursor = conn.execute(
        "SELECT qualifier_id FROM mesh_qualifiers WHERE qualifier_name = ?",
        (qualifier_name,),
    )
    row = cursor.fetchone()
    if row:
        return row["qualifier_id"]

    cursor = conn.execute(
        "INSERT INTO mesh_qualifiers (qualifier_name) VALUES (?)",
        (qualifier_name,),
    )
    conn.commit()
    return cursor.lastrowid


def link_article_mesh_term(
    conn: sqlite3.Connection,
    mesh_term_id: int,
    is_major_topic: bool = False,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Link a MeSH term to an article.

    Args:
        conn: Database connection.
        mesh_term_id: MeSH term ID.
        is_major_topic: Whether this is a major topic.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        article_mesh_term_id for linking qualifiers.
    """
    # Check if link already exists
    cursor = conn.execute(
        """
        SELECT id FROM article_mesh_terms
        WHERE (doi = ? OR (doi IS NULL AND ? IS NULL))
        AND (pmid = ? OR (pmid IS NULL AND ? IS NULL))
        AND mesh_term_id = ?
        """,
        (doi, doi, pmid, pmid, mesh_term_id),
    )
    row = cursor.fetchone()
    if row:
        return row["id"]

    cursor = conn.execute(
        """
        INSERT INTO article_mesh_terms (doi, pmid, pmcid, mesh_term_id, is_major_topic)
        VALUES (?, ?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, mesh_term_id, is_major_topic),
    )
    conn.commit()
    return cursor.lastrowid


def link_article_mesh_qualifier(
    conn: sqlite3.Connection,
    article_mesh_term_id: int,
    qualifier_id: int,
) -> None:
    """Link a MeSH qualifier to an article-mesh-term relationship.

    Args:
        conn: Database connection.
        article_mesh_term_id: Article-MeSH term link ID.
        qualifier_id: Qualifier ID.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO article_mesh_qualifiers (article_mesh_term_id, qualifier_id)
        VALUES (?, ?)
        """,
        (article_mesh_term_id, qualifier_id),
    )
    conn.commit()


def insert_full_article(
    conn: sqlite3.Connection,
    doi: str,
    pmid: str | None = None,
    pmcid: str | None = None,
    title: str | None = None,
    abstract: str | None = None,
    publication_date: str | None = None,
    electronic_publication_date: str | None = None,
    journal_id: int | None = None,
    volume: str | None = None,
    issue: str | None = None,
    article_type: str | None = None,
    language: str | None = None,
    copyright_text: str | None = None,
    article_filepath: str | None = None,
    supplementals_filepath: str | None = None,
) -> None:
    """Insert or replace an article record with full metadata.

    Args:
        conn: Database connection.
        doi: Digital Object Identifier (primary key).
        pmid: PubMed ID.
        pmcid: PubMed Central ID.
        title: Article title (HTML tags will be stripped).
        abstract: Article abstract.
        publication_date: Publication date (YYYY-MM-DD format).
        electronic_publication_date: Electronic publication date.
        journal_id: Journal ID from journals table.
        volume: Journal volume.
        issue: Journal issue.
        article_type: Type of article.
        language: Article language.
        copyright_text: Copyright information.
        article_filepath: Path to downloaded article file.
        supplementals_filepath: Path to supplementals archive.
    """
    # Strip HTML tags from title
    clean_title = _strip_html_tags(title)

    conn.execute(
        """
        INSERT INTO articles (
            doi, pmid, pmcid, title, abstract, publication_date,
            electronic_publication_date, journal_id, volume, issue,
            article_type, language, copyright, article_filepath,
            supplementals_filepath, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(doi) DO UPDATE SET
            pmid = COALESCE(excluded.pmid, pmid),
            pmcid = COALESCE(excluded.pmcid, pmcid),
            title = CASE
                WHEN excluded.title IS NULL THEN title
                WHEN title IS NULL THEN excluded.title
                WHEN LENGTH(excluded.title) > LENGTH(title) THEN excluded.title
                ELSE title
            END,
            abstract = COALESCE(excluded.abstract, abstract),
            publication_date = COALESCE(excluded.publication_date, publication_date),
            electronic_publication_date = COALESCE(excluded.electronic_publication_date, electronic_publication_date),
            journal_id = COALESCE(excluded.journal_id, journal_id),
            volume = COALESCE(excluded.volume, volume),
            issue = COALESCE(excluded.issue, issue),
            article_type = COALESCE(excluded.article_type, article_type),
            language = COALESCE(excluded.language, language),
            copyright = COALESCE(excluded.copyright, copyright),
            article_filepath = COALESCE(excluded.article_filepath, article_filepath),
            supplementals_filepath = COALESCE(excluded.supplementals_filepath, supplementals_filepath),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            doi, pmid, pmcid, clean_title, abstract, publication_date,
            electronic_publication_date, journal_id, volume, issue,
            article_type, language, copyright_text, article_filepath,
            supplementals_filepath,
        ),
    )
    conn.commit()


def get_articles_with_fulltext(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Get articles that have downloaded full text.

    Args:
        conn: Database connection.
        limit: Maximum number of articles to return.

    Returns:
        List of article rows with full text available.
    """
    query = """
        SELECT * FROM articles
        WHERE article_filepath IS NOT NULL
        ORDER BY created_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)
    return cursor.fetchall()


def get_articles_needing_llm_extraction(
    conn: sqlite3.Connection,
    model: str,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Get articles with full text that haven't had LLM metadata extracted.

    Finds articles that have downloaded full text but don't have any entries
    in the LLM metadata tables (llm_keywords, llm_raw_data, llm_processed_data,
    llm_analysis_methods, or llm_code) for the specified model.

    Args:
        conn: Database connection.
        model: LLM model name to check for existing extractions.
        limit: Maximum number of articles to return.

    Returns:
        List of article rows needing LLM extraction.
    """
    query = """
        SELECT a.* FROM articles a
        WHERE a.article_filepath IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM article_llm_keywords alk
            JOIN llm_keywords lk ON alk.llm_keyword_id = lk.llm_keyword_id
            WHERE (alk.doi = a.doi OR alk.pmid = a.pmid OR alk.pmcid = a.pmcid)
            AND lk.model = ?
        )
        AND NOT EXISTS (
            SELECT 1 FROM llm_raw_data lrd
            WHERE (lrd.doi = a.doi OR lrd.pmid = a.pmid OR lrd.pmcid = a.pmcid)
            AND lrd.model = ?
        )
        AND NOT EXISTS (
            SELECT 1 FROM llm_processed_data lpd
            WHERE (lpd.doi = a.doi OR lpd.pmid = a.pmid OR lpd.pmcid = a.pmcid)
            AND lpd.model = ?
        )
        AND NOT EXISTS (
            SELECT 1 FROM llm_analysis_methods lam
            WHERE (lam.doi = a.doi OR lam.pmid = a.pmid OR lam.pmcid = a.pmcid)
            AND lam.model = ?
        )
        AND NOT EXISTS (
            SELECT 1 FROM llm_code lc
            WHERE (lc.doi = a.doi OR lc.pmid = a.pmid OR lc.pmcid = a.pmcid)
            AND lc.model = ?
        )
        ORDER BY a.created_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query, (model, model, model, model, model))
    return cursor.fetchall()


def has_llm_extraction(
    conn: sqlite3.Connection,
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> bool:
    """Check if an article has LLM metadata extracted for a given model.

    Args:
        conn: Database connection.
        model: LLM model name to check.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        True if any LLM metadata exists for this article and model.
    """
    # Check each LLM table for existing data
    tables_and_conditions = [
        (
            "article_llm_keywords alk JOIN llm_keywords lk ON alk.llm_keyword_id = lk.llm_keyword_id",
            "lk.model = ?",
            "(alk.doi = ? OR alk.pmid = ? OR alk.pmcid = ?)",
        ),
        ("llm_raw_data", "model = ?", "(doi = ? OR pmid = ? OR pmcid = ?)"),
        ("llm_processed_data", "model = ?", "(doi = ? OR pmid = ? OR pmcid = ?)"),
        ("llm_analysis_methods", "model = ?", "(doi = ? OR pmid = ? OR pmcid = ?)"),
        ("llm_code", "model = ?", "(doi = ? OR pmid = ? OR pmcid = ?)"),
    ]

    for table, model_cond, id_cond in tables_and_conditions:
        query = f"SELECT 1 FROM {table} WHERE {model_cond} AND {id_cond} LIMIT 1"
        cursor = conn.execute(query, (model, doi, pmid, pmcid))
        if cursor.fetchone():
            return True

    return False


def insert_llm_extraction(
    conn: sqlite3.Connection,
    prompt: str,
    response: str,
    model: str,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Insert an LLM extraction record with the prompt and response.

    Args:
        conn: Database connection.
        prompt: The full prompt sent to the LLM.
        response: The raw LLM response text.
        model: Name of the LLM model used for extraction.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        The ID of the inserted record.
    """
    cursor = conn.execute(
        """
        INSERT INTO llm_extractions (doi, pmid, pmcid, prompt, response, model)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, prompt, response, model),
    )
    conn.commit()
    return cursor.lastrowid


def get_llm_extraction(
    conn: sqlite3.Connection,
    extraction_id: int,
) -> sqlite3.Row | None:
    """Retrieve an LLM extraction record by ID.

    Args:
        conn: Database connection.
        extraction_id: The ID of the extraction record.

    Returns:
        The extraction record or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM llm_extractions WHERE id = ?",
        (extraction_id,),
    )
    return cursor.fetchone()


def get_llm_extractions_for_article(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    model: str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve all LLM extraction records for an article.

    Args:
        conn: Database connection.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.
        model: Optional model name to filter by.

    Returns:
        List of extraction records.
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
    if model:
        conditions.append("model = ?")
        params.append(model)

    if not conditions:
        return []

    # Use OR for article identifiers, AND for model filter
    id_conditions = []
    if doi:
        id_conditions.append("doi = ?")
    if pmid:
        id_conditions.append("pmid = ?")
    if pmcid:
        id_conditions.append("pmcid = ?")

    where_clause = "(" + " OR ".join(id_conditions) + ")"
    params_list = [p for p, c in zip(params[:len(id_conditions)], id_conditions)]

    if model:
        where_clause += " AND model = ?"
        params_list.append(model)

    query = f"""
        SELECT * FROM llm_extractions
        WHERE {where_clause}
        ORDER BY created_at DESC
    """
    cursor = conn.execute(query, params_list)
    return cursor.fetchall()


def insert_supplemental_classification(
    conn: sqlite3.Connection,
    archive_path: str,
    file_path: str,
    classification: str,
    justification: str | None = None,
    model: str | None = None,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> int:
    """Insert or replace a supplemental file classification.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    archive_path : str
        Path to the supplementals archive.
    file_path : str
        Path of the file within the archive.
    classification : str
        Classification category. Must be one of: raw_data, quantitative_data,
        summary, processing_parameters, instrument_parameters, unknown.
    justification : str, optional
        Brief explanation for the classification.
    model : str, optional
        LLM model used for classification (None if heuristic).
    doi : str, optional
        Article DOI.
    pmid : str, optional
        Article PMID.
    pmcid : str, optional
        Article PMCID.

    Returns
    -------
    int
        The ID of the inserted or updated record.
    """
    cursor = conn.execute(
        """
        INSERT INTO supplemental_file_classifications
        (doi, pmid, pmcid, archive_path, file_path, classification, justification, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(archive_path, file_path) DO UPDATE SET
            doi = COALESCE(excluded.doi, doi),
            pmid = COALESCE(excluded.pmid, pmid),
            pmcid = COALESCE(excluded.pmcid, pmcid),
            classification = excluded.classification,
            justification = excluded.justification,
            model = excluded.model
        """,
        (doi, pmid, pmcid, archive_path, file_path, classification, justification, model),
    )
    conn.commit()
    return cursor.lastrowid


def get_supplemental_classifications(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    archive_path: str | None = None,
    classification: str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve supplemental file classifications from the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str, optional
        Filter by article DOI.
    pmid : str, optional
        Filter by article PMID.
    pmcid : str, optional
        Filter by article PMCID.
    archive_path : str, optional
        Filter by archive path.
    classification : str, optional
        Filter by classification category.

    Returns
    -------
    list[sqlite3.Row]
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
    if classification:
        conditions.append("classification = ?")
        params.append(classification)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    cursor = conn.execute(
        f"""
        SELECT * FROM supplemental_file_classifications
        WHERE {where_clause}
        ORDER BY file_path
        """,
        params,
    )

    return cursor.fetchall()


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


def get_datasets(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> list[dict]:
    """Retrieve datasets associated with an article from LLM extraction tables.

    Args:
        conn: Database connection.
        doi: Article DOI.
        pmid: Article PMID.
        pmcid: Article PMCID.

    Returns:
        List of dataset dictionaries containing ID, repository, and URL.
    """
    # 1. Resolve all identifiers for the article
    id_params = []
    id_conds = []
    if doi:
        id_conds.append("doi = ?")
        id_params.append(doi)
    if pmid:
        id_conds.append("pmid = ?")
        id_params.append(pmid)
    if pmcid:
        id_conds.append("pmcid = ?")
        id_params.append(pmcid)

    if not id_conds:
        return []

    query_article = f"SELECT doi, pmid, pmcid FROM articles WHERE {' OR '.join(id_conds)}"
    cursor = conn.execute(query_article, id_params)
    row = cursor.fetchone()

    # Use provided IDs if article not found in DB yet
    all_doi = row["doi"] if row and row["doi"] else doi
    all_pmid = row["pmid"] if row and row["pmid"] else pmid
    all_pmcid = row["pmcid"] if row and row["pmcid"] else pmcid

    datasets = []

    # 2. Build where clause for LLM tables
    llm_id_conds = []
    llm_params = []
    if all_doi:
        llm_id_conds.append("doi = ?")
        llm_params.append(all_doi)
    if all_pmid:
        llm_id_conds.append("pmid = ?")
        llm_params.append(all_pmid)
    if all_pmcid:
        llm_id_conds.append("pmcid = ?")
        llm_params.append(all_pmcid)

    if not llm_id_conds:
        return []

    where_clause = " OR ".join(llm_id_conds)

    # Query raw data table
    query_raw = f"""
        SELECT dataset_id, data_repository, url
        FROM llm_raw_data
        WHERE {where_clause}
    """
    cursor = conn.execute(query_raw, llm_params)
    for row in cursor:
        datasets.append(
            {
                "id": row["dataset_id"],
                "repository": row["data_repository"],
                "url": row["url"],
            }
        )

    # Query processed data table
    query_processed = f"""
        SELECT dataset_id, data_repository, url
        FROM llm_processed_data
        WHERE {where_clause}
    """
    cursor = conn.execute(query_processed, llm_params)
    for row in cursor:
        # Avoid duplicates if the same dataset is in both tables
        dataset_entry = {
            "id": row["dataset_id"],
            "repository": row["data_repository"],
            "url": row["url"],
        }
        if dataset_entry not in datasets:
            datasets.append(dataset_entry)

    return datasets


def insert_or_update_dataset(
    conn: sqlite3.Connection,
    dataset_id: str,
    title: str | None = None,
    description: str | None = None,
    species: str | None = None,
    instrument: str | None = None,
    repository: str | None = None,
    submission_date: str | None = None,
    publication_date: str | None = None,
    local_filepath: str | None = None,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> None:
    """Insert or update a dataset record.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str
        Dataset identifier (e.g., PXD012345).
    title : str, optional
        Dataset title.
    description : str, optional
        Dataset description.
    species : str, optional
        Species studied (comma-separated if multiple).
    instrument : str, optional
        Instrument(s) used (comma-separated if multiple).
    repository : str, optional
        Repository name (e.g., PRIDE, MassIVE).
    submission_date : str, optional
        Dataset submission date (YYYY-MM-DD format).
    publication_date : str, optional
        Dataset publication date (YYYY-MM-DD format).
    local_filepath : str, optional
        Local path where dataset files are stored.
    doi : str, optional
        DOI of associated publication.
    pmid : str, optional
        PubMed ID of associated publication.
    pmcid : str, optional
        PubMed Central ID of associated publication.
    """
    conn.execute(
        """
        INSERT INTO datasets (
            dataset_id, title, description, species, instrument, repository,
            submission_date, publication_date, local_filepath, doi, pmid, pmcid,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(dataset_id) DO UPDATE SET
            title = COALESCE(excluded.title, title),
            description = COALESCE(excluded.description, description),
            species = COALESCE(excluded.species, species),
            instrument = COALESCE(excluded.instrument, instrument),
            repository = COALESCE(excluded.repository, repository),
            submission_date = COALESCE(excluded.submission_date, submission_date),
            publication_date = COALESCE(excluded.publication_date, publication_date),
            local_filepath = COALESCE(excluded.local_filepath, local_filepath),
            doi = COALESCE(excluded.doi, doi),
            pmid = COALESCE(excluded.pmid, pmid),
            pmcid = COALESCE(excluded.pmcid, pmcid),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            dataset_id, title, description, species, instrument, repository,
            submission_date, publication_date, local_filepath, doi, pmid, pmcid,
        ),
    )
    conn.commit()


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> sqlite3.Row | None:
    """Retrieve a dataset by its identifier.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str
        Dataset identifier (e.g., PXD012345).

    Returns
    -------
    sqlite3.Row or None
        Dataset row or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM datasets WHERE dataset_id = ?",
        (dataset_id,),
    )
    return cursor.fetchone()


def get_datasets_by_article(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve datasets linked to an article.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str, optional
        Article DOI.
    pmid : str, optional
        Article PMID.
    pmcid : str, optional
        Article PMCID.

    Returns
    -------
    list[sqlite3.Row]
        List of dataset rows linked to the article.
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

    if not conditions:
        return []

    where_clause = " OR ".join(conditions)
    cursor = conn.execute(
        f"SELECT * FROM datasets WHERE {where_clause} ORDER BY submission_date DESC",
        params,
    )
    return cursor.fetchall()


def insert_dataset_file(
    conn: sqlite3.Connection,
    dataset_id: str,
    filename: str,
    file_type: str | None = None,
    file_type_reason: str | None = None,
    method: str | None = None,
    model: str | None = None,
    size_bytes: int | None = None,
    local_path: str | None = None,
) -> int:
    """Insert or update a dataset file record.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str
        Dataset identifier (e.g., PXD012345).
    filename : str
        Name of the file.
    file_type : str, optional
        Type/category of the file (e.g., raw_data, quantitative_data, etc.).
    file_type_reason : str, optional
        Reason/justification for the file type classification.
    method : str, optional
        Classification method used: "heuristic", "shallow_llm", or "llm".
    model : str, optional
        LLM model used for classification (if method is "shallow_llm" or "llm").
    size_bytes : int, optional
        File size in bytes.
    local_path : str, optional
        Local path to the downloaded file.

    Returns
    -------
    int
        ID of the inserted or updated record.
    """
    cursor = conn.execute(
        """
        INSERT INTO dataset_files (
            dataset_id, filename, file_type, file_type_reason, method, model,
            size_bytes, local_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, filename) DO UPDATE SET
            file_type = COALESCE(excluded.file_type, file_type),
            file_type_reason = COALESCE(excluded.file_type_reason, file_type_reason),
            method = COALESCE(excluded.method, method),
            model = COALESCE(excluded.model, model),
            size_bytes = COALESCE(excluded.size_bytes, size_bytes),
            local_path = COALESCE(excluded.local_path, local_path)
        """,
        (dataset_id, filename, file_type, file_type_reason, method, model, size_bytes, local_path),
    )
    conn.commit()
    return cursor.lastrowid


def get_dataset_files(
    conn: sqlite3.Connection,
    dataset_id: str,
    file_type: str | None = None,
) -> list[sqlite3.Row]:
    """Retrieve files for a dataset.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str
        Dataset identifier (e.g., PXD012345).
    file_type : str, optional
        Filter by file type.

    Returns
    -------
    list[sqlite3.Row]
        List of file records for the dataset.
    """
    if file_type:
        cursor = conn.execute(
            """
            SELECT * FROM dataset_files
            WHERE dataset_id = ? AND file_type = ?
            ORDER BY filename
            """,
            (dataset_id, file_type),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM dataset_files
            WHERE dataset_id = ?
            ORDER BY filename
            """,
            (dataset_id,),
        )
    return cursor.fetchall()


def delete_dataset_files(conn: sqlite3.Connection, dataset_id: str) -> int:
    """Delete all file records for a dataset.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str
        Dataset identifier (e.g., PXD012345).

    Returns
    -------
    int
        Number of records deleted.
    """
    cursor = conn.execute(
        "DELETE FROM dataset_files WHERE dataset_id = ?",
        (dataset_id,),
    )
    conn.commit()
    return cursor.rowcount


def update_agent_request_status(
    conn: sqlite3.Connection,
    request_id: int,
    status: str,
    status_reason: str | None = None,
) -> bool:
    """Update the status of an agent request.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    request_id : int
        The ID of the agent request to update.
    status : str
        New status value. Must be one of: 'pending', 'approved', 'rejected',
        'in_progress', 'implemented', 'completed', 'cancelled'.
    status_reason : str, optional
        Reason for the status change.

    Returns
    -------
    bool
        True if the request was found and updated, False otherwise.
    """
    cursor = conn.execute(
        """
        UPDATE agent_requests
        SET request_status = ?, status_reason = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (status, status_reason, request_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def mark_agent_request_in_progress(
    conn: sqlite3.Connection,
    request_id: int,
    status_reason: str | None = None,
) -> bool:
    """Mark an agent request as in progress and set the assigned_time.

    This function updates the request status to 'in_progress' and sets the
    assigned_time to the current timestamp. It is used to indicate that an
    agent has started working on implementing the feature.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    request_id : int
        The ID of the agent request to mark as in progress.
    status_reason : str, optional
        Reason or notes about the status change.

    Returns
    -------
    bool
        True if the request was found and updated, False otherwise.
    """
    cursor = conn.execute(
        """
        UPDATE agent_requests
        SET request_status = 'in_progress',
            status_reason = ?,
            assigned_time = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (status_reason, request_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_agent_request(conn: sqlite3.Connection, request_id: int) -> sqlite3.Row | None:
    """Retrieve an agent request by ID.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    request_id : int
        The ID of the agent request.

    Returns
    -------
    sqlite3.Row or None
        The agent request row or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM agent_requests WHERE id = ?",
        (request_id,),
    )
    return cursor.fetchone()


def get_oldest_approved_agent_request(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Retrieve the oldest approved agent request.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    sqlite3.Row or None
        The oldest approved agent request or None if none found.
    """
    cursor = conn.execute(
        """
        SELECT * FROM agent_requests
        WHERE request_status = 'approved'
        ORDER BY creation_time ASC
        LIMIT 1
        """,
    )
    return cursor.fetchone()


# ---------------------------------------------------------------------------
# Provenance / research-object layer (Phase 2)
# ---------------------------------------------------------------------------


def _encode_json(value: object | None) -> str | None:
    """Encode a Python object as a JSON string for a ``*_json`` TEXT column.

    Parameters
    ----------
    value : object or None
        A JSON-serializable value (typically a list or dict). If already a
        string it is stored verbatim.

    Returns
    -------
    str or None
        The JSON-encoded string, or None if ``value`` is None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _decode_json(text: str | None) -> object | None:
    """Decode a JSON string from a ``*_json`` TEXT column into a Python object.

    Parameters
    ----------
    text : str or None
        The stored JSON text.

    Returns
    -------
    object or None
        The decoded Python object, or None if ``text`` is None or invalid JSON.
    """
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def insert_quantification_run(
    conn: sqlite3.Connection,
    dataset_id: str | None = None,
    tool: str | None = None,
    tool_version: str | None = None,
    container_image: str | None = None,
    container_sha256: str | None = None,
    param_file_path: str | None = None,
    param_file_sha256: str | None = None,
    command: str | None = None,
    input_files: list | dict | str | None = None,
    output_dir: str | None = None,
    exit_status: int | None = None,
    wall_time_sec: float | None = None,
    host: str | None = None,
    extraction_model: str | None = None,
    provider: str | None = None,
) -> int:
    """Insert a quantification run provenance record.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str, optional
        Source dataset identifier (e.g., "PXD012345").
    tool : str, optional
        Quantification tool name (e.g., "DIA-NN", "MaxQuant").
    tool_version : str, optional
        Version string of the tool.
    container_image : str, optional
        Container image reference (name:tag) used for the run.
    container_sha256 : str, optional
        SHA-256 digest of the container image.
    param_file_path : str, optional
        Path to the parameter/config file used.
    param_file_sha256 : str, optional
        SHA-256 hash of the parameter file contents.
    command : str, optional
        Full command line executed.
    input_files : list or dict or str, optional
        Input file paths; stored as JSON in ``input_files_json``.
    output_dir : str, optional
        Directory where outputs were written.
    exit_status : int, optional
        Process exit status code.
    wall_time_sec : float, optional
        Wall-clock run time in seconds.
    host : str, optional
        Host/machine identifier where the run executed.
    extraction_model : str, optional
        LLM model used to derive parameters, if any.
    provider : str, optional
        LLM/compute provider (e.g., "azure").

    Returns
    -------
    int
        The ID of the inserted quantification run.
    """
    cursor = conn.execute(
        """
        INSERT INTO quantification_runs (
            dataset_id, tool, tool_version, container_image, container_sha256,
            param_file_path, param_file_sha256, command, input_files_json,
            output_dir, exit_status, wall_time_sec, host, extraction_model, provider
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id, tool, tool_version, container_image, container_sha256,
            param_file_path, param_file_sha256, command, _encode_json(input_files),
            output_dir, exit_status, wall_time_sec, host, extraction_model, provider,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_quantification_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> sqlite3.Row | None:
    """Retrieve a quantification run by ID.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    run_id : int
        The quantification run ID.

    Returns
    -------
    sqlite3.Row or None
        The quantification run row, or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM quantification_runs WHERE id = ?",
        (run_id,),
    )
    return cursor.fetchone()


def get_quantification_runs(
    conn: sqlite3.Connection,
    dataset_id: str | None = None,
    tool: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Retrieve quantification runs, optionally filtered.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    dataset_id : str, optional
        Filter by source dataset identifier.
    tool : str, optional
        Filter by tool name.
    limit : int, optional
        Maximum number of rows to return.

    Returns
    -------
    list[sqlite3.Row]
        Matching quantification run rows, newest first.
    """
    conditions = []
    params: list = []
    if dataset_id:
        conditions.append("dataset_id = ?")
        params.append(dataset_id)
    if tool:
        conditions.append("tool = ?")
        params.append(tool)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM quantification_runs WHERE {where_clause} ORDER BY created_at DESC, id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return cursor.fetchall()


def insert_analysis_run(
    conn: sqlite3.Connection,
    analysis_type: str | None = None,
    method: str | None = None,
    quantification_run_id: int | None = None,
    library: str | None = None,
    library_version: str | None = None,
    parameters: dict | list | str | None = None,
    code_sha256: str | None = None,
    random_seed: int | None = None,
    input_paths: list | dict | str | None = None,
    output_paths: list | dict | str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> int:
    """Insert an analysis run provenance record.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    analysis_type : str, optional
        Type of analysis (e.g., "QC", "DE", "enrichment").
    method : str, optional
        Method/algorithm name.
    quantification_run_id : int, optional
        ID of the quantification run that produced the analyzed inputs.
    library : str, optional
        Analysis library/package name.
    library_version : str, optional
        Version of the analysis library.
    parameters : dict or list or str, optional
        Analysis parameters; stored as JSON in ``parameters_json``.
    code_sha256 : str, optional
        SHA-256 hash of the analysis code.
    random_seed : int, optional
        Random seed used for reproducibility.
    input_paths : list or dict or str, optional
        Input paths; stored as JSON in ``input_paths_json``.
    output_paths : list or dict or str, optional
        Output paths; stored as JSON in ``output_paths_json``.
    provider : str, optional
        LLM/compute provider, if any.
    model : str, optional
        LLM model used, if any.

    Returns
    -------
    int
        The ID of the inserted analysis run.
    """
    cursor = conn.execute(
        """
        INSERT INTO analysis_runs (
            quantification_run_id, analysis_type, method, library, library_version,
            parameters_json, code_sha256, random_seed, input_paths_json,
            output_paths_json, provider, model
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            quantification_run_id, analysis_type, method, library, library_version,
            _encode_json(parameters), code_sha256, random_seed,
            _encode_json(input_paths), _encode_json(output_paths), provider, model,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_analysis_run(
    conn: sqlite3.Connection,
    run_id: int,
) -> sqlite3.Row | None:
    """Retrieve an analysis run by ID.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    run_id : int
        The analysis run ID.

    Returns
    -------
    sqlite3.Row or None
        The analysis run row, or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM analysis_runs WHERE id = ?",
        (run_id,),
    )
    return cursor.fetchone()


def get_analysis_runs(
    conn: sqlite3.Connection,
    quantification_run_id: int | None = None,
    analysis_type: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Retrieve analysis runs, optionally filtered.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    quantification_run_id : int, optional
        Filter by parent quantification run ID.
    analysis_type : str, optional
        Filter by analysis type.
    limit : int, optional
        Maximum number of rows to return.

    Returns
    -------
    list[sqlite3.Row]
        Matching analysis run rows, newest first.
    """
    conditions = []
    params: list = []
    if quantification_run_id is not None:
        conditions.append("quantification_run_id = ?")
        params.append(quantification_run_id)
    if analysis_type:
        conditions.append("analysis_type = ?")
        params.append(analysis_type)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM analysis_runs WHERE {where_clause} ORDER BY created_at DESC, id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return cursor.fetchall()


def insert_dep_result(
    conn: sqlite3.Connection,
    analysis_run_id: int,
    feature_id: str | None = None,
    log2fc: float | None = None,
    pvalue: float | None = None,
    padj: float | None = None,
    direction: str | None = None,
    significant: bool | None = None,
) -> int:
    """Insert a single differential expression result row.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    analysis_run_id : int
        ID of the analysis run that produced this result.
    feature_id : str, optional
        Feature identifier (protein/peptide/gene).
    log2fc : float, optional
        Log2 fold change.
    pvalue : float, optional
        Raw p-value.
    padj : float, optional
        Adjusted p-value (e.g., BH-corrected).
    direction : str, optional
        Direction of change (e.g., "up", "down").
    significant : bool, optional
        Whether the feature is significant.

    Returns
    -------
    int
        The ID of the inserted result row.
    """
    cursor = conn.execute(
        """
        INSERT INTO dep_results (
            analysis_run_id, feature_id, log2fc, pvalue, padj, direction, significant
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_run_id, feature_id, log2fc, pvalue, padj, direction,
            None if significant is None else int(bool(significant)),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_dep_results(
    conn: sqlite3.Connection,
    analysis_run_id: int,
    results: list[dict],
) -> list[int]:
    """Insert multiple differential expression result rows.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    analysis_run_id : int
        ID of the analysis run that produced these results.
    results : list of dict
        Each dict may contain keys: ``feature_id``, ``log2fc``, ``pvalue``,
        ``padj``, ``direction``, ``significant``.

    Returns
    -------
    list[int]
        IDs of the inserted result rows, in input order.
    """
    ids: list[int] = []
    for r in results:
        significant = r.get("significant")
        cursor = conn.execute(
            """
            INSERT INTO dep_results (
                analysis_run_id, feature_id, log2fc, pvalue, padj, direction, significant
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_run_id,
                r.get("feature_id"),
                r.get("log2fc"),
                r.get("pvalue"),
                r.get("padj"),
                r.get("direction"),
                None if significant is None else int(bool(significant)),
            ),
        )
        ids.append(cursor.lastrowid)
    conn.commit()
    return ids


def get_dep_results(
    conn: sqlite3.Connection,
    analysis_run_id: int,
    significant_only: bool = False,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Retrieve differential expression results for an analysis run.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    analysis_run_id : int
        The analysis run ID to fetch results for.
    significant_only : bool, optional
        If True, only return rows flagged as significant.
    limit : int, optional
        Maximum number of rows to return.

    Returns
    -------
    list[sqlite3.Row]
        Matching result rows, ordered by adjusted p-value.
    """
    params: list = [analysis_run_id]
    query = "SELECT * FROM dep_results WHERE analysis_run_id = ?"
    if significant_only:
        query += " AND significant = 1"
    query += " ORDER BY (padj IS NULL), padj ASC, id ASC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return cursor.fetchall()


def insert_benchmark_annotation(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    dataset_id: str | None = None,
    annotator: str | None = None,
    label: str | None = None,
    category: str | None = None,
    evidence_text: str | None = None,
) -> int:
    """Insert a benchmark (ground-truth) annotation record.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str, optional
        Article DOI.
    pmid : str, optional
        Article PMID.
    pmcid : str, optional
        Article PMCID.
    dataset_id : str, optional
        Associated dataset identifier.
    annotator : str, optional
        Name/identifier of the annotator.
    label : str, optional
        The ground-truth label.
    category : str, optional
        Category/task the label belongs to.
    evidence_text : str, optional
        Supporting evidence for the annotation.

    Returns
    -------
    int
        The ID of the inserted annotation.
    """
    cursor = conn.execute(
        """
        INSERT INTO benchmark_annotations (
            doi, pmid, pmcid, dataset_id, annotator, label, category, evidence_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, dataset_id, annotator, label, category, evidence_text),
    )
    conn.commit()
    return cursor.lastrowid


def get_benchmark_annotations(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    dataset_id: str | None = None,
    category: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Retrieve benchmark annotations, optionally filtered.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str, optional
        Filter by article DOI.
    pmid : str, optional
        Filter by article PMID.
    pmcid : str, optional
        Filter by article PMCID.
    dataset_id : str, optional
        Filter by dataset identifier.
    category : str, optional
        Filter by category.
    limit : int, optional
        Maximum number of rows to return.

    Returns
    -------
    list[sqlite3.Row]
        Matching annotation rows, newest first.
    """
    conditions = []
    params: list = []
    if doi:
        conditions.append("doi = ?")
        params.append(doi)
    if pmid:
        conditions.append("pmid = ?")
        params.append(pmid)
    if pmcid:
        conditions.append("pmcid = ?")
        params.append(pmcid)
    if dataset_id:
        conditions.append("dataset_id = ?")
        params.append(dataset_id)
    if category:
        conditions.append("category = ?")
        params.append(category)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM benchmark_annotations WHERE {where_clause} ORDER BY created_at DESC, id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return cursor.fetchall()


def insert_benchmark_prediction(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    dataset_id: str | None = None,
    predicted_label: str | None = None,
    confidence: float | None = None,
    model: str | None = None,
    provider: str | None = None,
    run_at: str | None = None,
) -> int:
    """Insert a benchmark prediction record.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str, optional
        Article DOI.
    pmid : str, optional
        Article PMID.
    pmcid : str, optional
        Article PMCID.
    dataset_id : str, optional
        Associated dataset identifier.
    predicted_label : str, optional
        The predicted label.
    confidence : float, optional
        Confidence score for the prediction.
    model : str, optional
        Model that produced the prediction.
    provider : str, optional
        Provider of the model (e.g., "azure").
    run_at : str, optional
        Timestamp when the prediction was produced. If None, the row's
        ``created_at`` still records insertion time.

    Returns
    -------
    int
        The ID of the inserted prediction.
    """
    cursor = conn.execute(
        """
        INSERT INTO benchmark_predictions (
            doi, pmid, pmcid, dataset_id, predicted_label, confidence,
            model, provider, run_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, dataset_id, predicted_label, confidence, model, provider, run_at),
    )
    conn.commit()
    return cursor.lastrowid


def get_benchmark_predictions(
    conn: sqlite3.Connection,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    dataset_id: str | None = None,
    model: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Retrieve benchmark predictions, optionally filtered.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str, optional
        Filter by article DOI.
    pmid : str, optional
        Filter by article PMID.
    pmcid : str, optional
        Filter by article PMCID.
    dataset_id : str, optional
        Filter by dataset identifier.
    model : str, optional
        Filter by model.
    limit : int, optional
        Maximum number of rows to return.

    Returns
    -------
    list[sqlite3.Row]
        Matching prediction rows, newest first.
    """
    conditions = []
    params: list = []
    if doi:
        conditions.append("doi = ?")
        params.append(doi)
    if pmid:
        conditions.append("pmid = ?")
        params.append(pmid)
    if pmcid:
        conditions.append("pmcid = ?")
        params.append(pmcid)
    if dataset_id:
        conditions.append("dataset_id = ?")
        params.append(dataset_id)
    if model:
        conditions.append("model = ?")
        params.append(model)

    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM benchmark_predictions WHERE {where_clause} ORDER BY created_at DESC, id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    cursor = conn.execute(query, params)
    return cursor.fetchall()
