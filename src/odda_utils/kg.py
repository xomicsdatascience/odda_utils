"""Knowledge graph functions for querying article and dataset information.

This module provides functions for:
- Finding datasets associated with articles
- Building knowledge graph JSON representations from the database
- The knowledge graph includes articles, authors, keywords, MeSH terms, datasets, and journals
"""

import json
import logging
from pathlib import Path
from typing import Any

from odda_utils.database import get_datasets, init_db

logger = logging.getLogger(__name__)


def find_datasets(
    db_path: str | Path,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> str:
    """Find datasets associated with an article and return as JSON.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    doi : str, optional
        Article DOI.
    pmid : str, optional
        Article PMID.
    pmcid : str, optional
        Article PMCID.

    Returns
    -------
    str
        JSON string containing a list of dataset objects.
    """
    if not any([doi, pmid, pmcid]):
        return json.dumps([])

    conn = init_db(db_path)
    try:
        datasets = get_datasets(conn, doi=doi, pmid=pmid, pmcid=pmcid)
        return json.dumps(datasets, indent=2)
    finally:
        conn.close()


def build_knowledge_graph(
    db_path: str | Path,
    include_authors: bool = True,
    include_keywords: bool = True,
    include_mesh_terms: bool = True,
    include_datasets: bool = True,
    include_journals: bool = True,
    limit_articles: int | None = None,
) -> dict[str, Any]:
    """Build a knowledge graph JSON representation from the database.

    Creates a graph structure with nodes and edges representing articles and their
    relationships to authors, keywords, MeSH terms, datasets, and journals.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    include_authors : bool, optional
        Whether to include author nodes and edges. Default is True.
    include_keywords : bool, optional
        Whether to include keyword nodes and edges. Default is True.
    include_mesh_terms : bool, optional
        Whether to include MeSH term nodes and edges. Default is True.
    include_datasets : bool, optional
        Whether to include dataset nodes and edges. Default is True.
    include_journals : bool, optional
        Whether to include journal nodes and edges. Default is True.
    limit_articles : int, optional
        Maximum number of articles to include. If None, includes all articles.

    Returns
    -------
    dict
        Knowledge graph dictionary with the following structure:
        {
            "nodes": [
                {"id": "...", "type": "article|author|keyword|mesh|dataset|journal",
                 "label": "...", "attributes": {...}},
                ...
            ],
            "edges": [
                {"source": "...", "target": "...", "type": "...", "attributes": {...}},
                ...
            ],
            "metadata": {
                "article_count": int,
                "node_count": int,
                "edge_count": int,
                "node_types": {"article": int, "author": int, ...}
            }
        }
    """
    conn = init_db(db_path)
    try:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        node_type_counts: dict[str, int] = {}

        # Query articles
        query = "SELECT * FROM articles"
        if limit_articles:
            query += f" LIMIT {limit_articles}"
        cursor = conn.execute(query)
        articles = cursor.fetchall()

        for article in articles:
            doi = article["doi"]
            pmid = article["pmid"]
            pmcid = article["pmcid"]

            # Use DOI as primary ID, fallback to PMID or PMCID
            article_id = doi or pmid or pmcid
            if not article_id:
                continue

            node_key = f"article:{article_id}"
            if node_key in node_ids:
                continue
            node_ids.add(node_key)

            # Create article node
            article_node = {
                "id": node_key,
                "type": "article",
                "label": article["title"][:80] + "..." if article["title"] and len(article["title"]) > 80 else article["title"],
                "attributes": {
                    "doi": doi,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "title": article["title"],
                    "publication_date": article["publication_date"],
                    "article_type": article["article_type"],
                },
            }
            nodes.append(article_node)
            node_type_counts["article"] = node_type_counts.get("article", 0) + 1

            # Add journal relationship
            if include_journals and article["journal_id"]:
                journal_cursor = conn.execute(
                    "SELECT * FROM journals WHERE journal_id = ?",
                    (article["journal_id"],),
                )
                journal = journal_cursor.fetchone()
                if journal:
                    journal_key = f"journal:{journal['journal_id']}"
                    if journal_key not in node_ids:
                        node_ids.add(journal_key)
                        journal_node = {
                            "id": journal_key,
                            "type": "journal",
                            "label": journal["name"],
                            "attributes": {
                                "journal_id": journal["journal_id"],
                                "name": journal["name"],
                                "issn": journal["issn"],
                                "iso_abbreviation": journal["iso_abbreviation"],
                            },
                        }
                        nodes.append(journal_node)
                        node_type_counts["journal"] = node_type_counts.get("journal", 0) + 1

                    edges.append({
                        "source": node_key,
                        "target": journal_key,
                        "type": "published_in",
                        "attributes": {},
                    })

            # Add author relationships
            if include_authors:
                author_cursor = conn.execute(
                    """
                    SELECT ai.*, aa.author_position, aa.is_corresponding
                    FROM article_authors aa
                    JOIN author_info ai ON aa.author_id = ai.author_id
                    WHERE aa.doi = ? OR aa.pmid = ? OR aa.pmcid = ?
                    ORDER BY aa.author_position
                    """,
                    (doi, pmid, pmcid),
                )
                for author in author_cursor:
                    author_key = f"author:{author['author_id']}"
                    if author_key not in node_ids:
                        node_ids.add(author_key)
                        author_name = f"{author['first_name'] or ''} {author['last_name'] or ''}".strip()
                        author_node = {
                            "id": author_key,
                            "type": "author",
                            "label": author_name,
                            "attributes": {
                                "author_id": author["author_id"],
                                "first_name": author["first_name"],
                                "last_name": author["last_name"],
                                "orcid": author["author_orcid"],
                                "is_collective": bool(author["is_collective"]),
                            },
                        }
                        nodes.append(author_node)
                        node_type_counts["author"] = node_type_counts.get("author", 0) + 1

                    edges.append({
                        "source": author_key,
                        "target": node_key,
                        "type": "authored",
                        "attributes": {
                            "position": author["author_position"],
                            "is_corresponding": bool(author["is_corresponding"]),
                        },
                    })

            # Add keyword relationships
            if include_keywords:
                keyword_cursor = conn.execute(
                    """
                    SELECT k.*
                    FROM article_keywords ak
                    JOIN keywords k ON ak.keyword_id = k.keyword_id
                    WHERE ak.doi = ? OR ak.pmid = ? OR ak.pmcid = ?
                    """,
                    (doi, pmid, pmcid),
                )
                for keyword in keyword_cursor:
                    keyword_key = f"keyword:{keyword['keyword_id']}"
                    if keyword_key not in node_ids:
                        node_ids.add(keyword_key)
                        keyword_node = {
                            "id": keyword_key,
                            "type": "keyword",
                            "label": keyword["keyword"],
                            "attributes": {
                                "keyword_id": keyword["keyword_id"],
                                "keyword": keyword["keyword"],
                            },
                        }
                        nodes.append(keyword_node)
                        node_type_counts["keyword"] = node_type_counts.get("keyword", 0) + 1

                    edges.append({
                        "source": node_key,
                        "target": keyword_key,
                        "type": "has_keyword",
                        "attributes": {},
                    })

            # Add MeSH term relationships
            if include_mesh_terms:
                mesh_cursor = conn.execute(
                    """
                    SELECT mt.*, amt.is_major_topic
                    FROM article_mesh_terms amt
                    JOIN mesh_terms mt ON amt.mesh_term_id = mt.mesh_term_id
                    WHERE amt.doi = ? OR amt.pmid = ? OR amt.pmcid = ?
                    """,
                    (doi, pmid, pmcid),
                )
                for mesh in mesh_cursor:
                    mesh_key = f"mesh:{mesh['mesh_term_id']}"
                    if mesh_key not in node_ids:
                        node_ids.add(mesh_key)
                        mesh_node = {
                            "id": mesh_key,
                            "type": "mesh",
                            "label": mesh["descriptor_name"],
                            "attributes": {
                                "mesh_term_id": mesh["mesh_term_id"],
                                "descriptor_name": mesh["descriptor_name"],
                                "descriptor_ui": mesh["descriptor_ui"],
                            },
                        }
                        nodes.append(mesh_node)
                        node_type_counts["mesh"] = node_type_counts.get("mesh", 0) + 1

                    edges.append({
                        "source": node_key,
                        "target": mesh_key,
                        "type": "has_mesh_term",
                        "attributes": {
                            "is_major_topic": bool(mesh["is_major_topic"]),
                        },
                    })

            # Add dataset relationships
            if include_datasets:
                dataset_list = get_datasets(conn, doi=doi, pmid=pmid, pmcid=pmcid)
                for dataset in dataset_list:
                    dataset_key = f"dataset:{dataset['id']}"
                    if dataset_key not in node_ids:
                        node_ids.add(dataset_key)
                        dataset_node = {
                            "id": dataset_key,
                            "type": "dataset",
                            "label": dataset["id"],
                            "attributes": {
                                "dataset_id": dataset["id"],
                                "repository": dataset["repository"],
                                "url": dataset["url"],
                            },
                        }
                        nodes.append(dataset_node)
                        node_type_counts["dataset"] = node_type_counts.get("dataset", 0) + 1

                    edges.append({
                        "source": node_key,
                        "target": dataset_key,
                        "type": "uses_dataset",
                        "attributes": {},
                    })

        # Build result
        result = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "article_count": node_type_counts.get("article", 0),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_types": node_type_counts,
            },
        }

        return result

    finally:
        conn.close()


def build_knowledge_graph_json(
    db_path: str | Path,
    include_authors: bool = True,
    include_keywords: bool = True,
    include_mesh_terms: bool = True,
    include_datasets: bool = True,
    include_journals: bool = True,
    limit_articles: int | None = None,
) -> str:
    """Build a knowledge graph and return it as a JSON string.

    This is a convenience wrapper around build_knowledge_graph that returns
    the result as a formatted JSON string.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    include_authors : bool, optional
        Whether to include author nodes and edges. Default is True.
    include_keywords : bool, optional
        Whether to include keyword nodes and edges. Default is True.
    include_mesh_terms : bool, optional
        Whether to include MeSH term nodes and edges. Default is True.
    include_datasets : bool, optional
        Whether to include dataset nodes and edges. Default is True.
    include_journals : bool, optional
        Whether to include journal nodes and edges. Default is True.
    limit_articles : int, optional
        Maximum number of articles to include. If None, includes all articles.

    Returns
    -------
    str
        JSON string representation of the knowledge graph.
    """
    graph = build_knowledge_graph(
        db_path=db_path,
        include_authors=include_authors,
        include_keywords=include_keywords,
        include_mesh_terms=include_mesh_terms,
        include_datasets=include_datasets,
        include_journals=include_journals,
        limit_articles=limit_articles,
    )
    return json.dumps(graph, indent=2)
