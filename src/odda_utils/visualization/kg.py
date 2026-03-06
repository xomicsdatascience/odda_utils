# Knowledge graph JSON generator for article-dataset relationships.
#
# This module provides functionality to generate a knowledge graph JSON representation
# of articles and their associated datasets from the SQLite database. The output format
# is compatible with common visualization libraries such as D3.js, vis.js, and Cytoscape.js.
#
# The graph consists of:
# - Nodes: articles (identified by doi/pmid/pmcid) and datasets (identified by dataset_id)
# - Edges: directional "has_dataset" edges from articles to datasets

import json
import sqlite3
from pathlib import Path
from typing import Any


# Default database path relative to project root
# From src/knowledge_graph/visualization/ go up to mcp/ project root
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent.parent / "articles.sqlite"


def get_article_identifier(doi: str | None, pmid: str | None, pmcid: str | None) -> str:
    """
    Get the primary identifier for an article.

    Prioritizes DOI, then PMID, then PMCID.

    Parameters
    ----------
    doi : str or None
        Digital Object Identifier.
    pmid : str or None
        PubMed ID.
    pmcid : str or None
        PubMed Central ID.

    Returns
    -------
    str
        The primary identifier for the article.

    Raises
    ------
    ValueError
        If all identifiers are None.
    """
    if doi:
        return f"article:doi:{doi}"
    elif pmid:
        return f"article:pmid:{pmid}"
    elif pmcid:
        return f"article:pmcid:{pmcid}"
    else:
        raise ValueError("At least one identifier (doi, pmid, pmcid) must be provided")


def fetch_articles(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Fetch all articles from the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    list of dict
        List of article dictionaries with keys: doi, pmid, pmcid, title,
        publication_date, journal_id.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT doi, pmid, pmcid, title, publication_date, journal_id
        FROM articles
    """)
    columns = ["doi", "pmid", "pmcid", "title", "publication_date", "journal_id"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def fetch_datasets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Fetch all datasets from the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    list of dict
        List of dataset dictionaries with keys: dataset_id, title, description,
        species, repository, doi, pmid, pmcid.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT dataset_id, title, description, species, repository, doi, pmid, pmcid
        FROM datasets
    """)
    columns = ["dataset_id", "title", "description", "species", "repository",
               "doi", "pmid", "pmcid"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def fetch_llm_dataset_links(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """
    Fetch article-dataset links from LLM extraction tables.

    Combines links from both llm_raw_data and llm_processed_data tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    list of dict
        List of link dictionaries with keys: doi, pmid, pmcid, dataset_id,
        data_repository.
    """
    cursor = conn.cursor()

    # Query both tables and union the results (deduplicated)
    cursor.execute("""
        SELECT DISTINCT doi, pmid, pmcid, dataset_id, data_repository
        FROM llm_raw_data
        WHERE dataset_id IS NOT NULL
        UNION
        SELECT DISTINCT doi, pmid, pmcid, dataset_id, data_repository
        FROM llm_processed_data
        WHERE dataset_id IS NOT NULL
    """)
    columns = ["doi", "pmid", "pmcid", "dataset_id", "data_repository"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def build_knowledge_graph(
    db_path: str | Path = DEFAULT_DB_PATH,
    include_orphan_articles: bool = True,
    include_orphan_datasets: bool = True
) -> dict[str, Any]:
    """
    Build a knowledge graph JSON structure from the database.

    The output format is compatible with common visualization libraries:
    - D3.js: Can be used directly with force-directed graph layouts
    - vis.js: Nodes and edges arrays can be passed to Network
    - Cytoscape.js: Can be converted to elements format with minimal transformation

    Parameters
    ----------
    db_path : str or Path, optional
        Path to the SQLite database. Defaults to the project's articles.sqlite.
    include_orphan_articles : bool, optional
        Whether to include articles that have no associated datasets.
        Default is True.
    include_orphan_datasets : bool, optional
        Whether to include datasets that have no associated articles.
        Default is True.

    Returns
    -------
    dict
        Knowledge graph with structure:
        {
            "nodes": [
                {
                    "id": str,           # Unique node identifier
                    "type": str,         # "article" or "dataset"
                    "label": str,        # Display label
                    "properties": dict   # Additional metadata
                },
                ...
            ],
            "edges": [
                {
                    "source": str,       # Source node id (article)
                    "target": str,       # Target node id (dataset)
                    "type": str,         # Edge type ("has_dataset")
                    "properties": dict   # Additional edge metadata
                },
                ...
            ],
            "metadata": {
                "node_count": int,
                "edge_count": int,
                "article_count": int,
                "dataset_count": int
            }
        }

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    sqlite3.Error
        If there is an error querying the database.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        # Fetch data from database
        articles = fetch_articles(conn)
        datasets = fetch_datasets(conn)
        llm_links = fetch_llm_dataset_links(conn)
    finally:
        conn.close()

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # Track which articles and datasets have connections
    connected_articles: set[str] = set()
    connected_datasets: set[str] = set()

    # Build a mapping of article identifiers to article data
    article_map: dict[str, dict[str, Any]] = {}
    for article in articles:
        try:
            article_id = get_article_identifier(
                article["doi"], article["pmid"], article["pmcid"]
            )
            article_map[article_id] = article
            # Also create lookup by individual identifiers for edge matching
            if article["doi"]:
                article_map[f"doi:{article['doi']}"] = article
            if article["pmid"]:
                article_map[f"pmid:{article['pmid']}"] = article
            if article["pmcid"]:
                article_map[f"pmcid:{article['pmcid']}"] = article
        except ValueError:
            # Skip articles with no identifiers
            continue

    # Build dataset map
    dataset_map: dict[str, dict[str, Any]] = {d["dataset_id"]: d for d in datasets}

    # Process direct links from datasets table (doi, pmid, pmcid columns)
    edge_set: set[tuple[str, str]] = set()  # Track unique edges

    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        target_id = f"dataset:{dataset_id}"

        # Try to find the linked article
        article_data = None
        if dataset["doi"]:
            article_data = article_map.get(f"doi:{dataset['doi']}")
        if not article_data and dataset["pmid"]:
            article_data = article_map.get(f"pmid:{dataset['pmid']}")
        if not article_data and dataset["pmcid"]:
            article_data = article_map.get(f"pmcid:{dataset['pmcid']}")

        if article_data:
            try:
                source_id = get_article_identifier(
                    article_data["doi"], article_data["pmid"], article_data["pmcid"]
                )
                edge_key = (source_id, target_id)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "type": "has_dataset",
                        "properties": {
                            "source_table": "datasets"
                        }
                    })
                    connected_articles.add(source_id)
                    connected_datasets.add(dataset_id)
            except ValueError:
                continue

    # Process links from LLM extraction tables
    for link in llm_links:
        dataset_id = link["dataset_id"]
        target_id = f"dataset:{dataset_id}"

        # Find the article
        article_data = None
        if link["doi"]:
            article_data = article_map.get(f"doi:{link['doi']}")
        if not article_data and link["pmid"]:
            article_data = article_map.get(f"pmid:{link['pmid']}")
        if not article_data and link["pmcid"]:
            article_data = article_map.get(f"pmcid:{link['pmcid']}")

        if article_data:
            try:
                source_id = get_article_identifier(
                    article_data["doi"], article_data["pmid"], article_data["pmcid"]
                )
                edge_key = (source_id, target_id)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "type": "has_dataset",
                        "properties": {
                            "source_table": "llm_extraction",
                            "data_repository": link.get("data_repository")
                        }
                    })
                    connected_articles.add(source_id)
                    connected_datasets.add(dataset_id)
            except ValueError:
                continue

    # Build article nodes
    for article in articles:
        try:
            article_id = get_article_identifier(
                article["doi"], article["pmid"], article["pmcid"]
            )
        except ValueError:
            continue

        is_connected = article_id in connected_articles
        if not is_connected and not include_orphan_articles:
            continue

        # Create a display label (truncated title or identifier)
        label = article["title"]
        if label and len(label) > 50:
            label = label[:47] + "..."
        elif not label:
            label = article_id.replace("article:", "")

        nodes.append({
            "id": article_id,
            "type": "article",
            "label": label,
            "properties": {
                "doi": article["doi"],
                "pmid": article["pmid"],
                "pmcid": article["pmcid"],
                "title": article["title"],
                "publication_date": article["publication_date"],
                "journal_id": article["journal_id"]
            }
        })

    # Build dataset nodes
    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        is_connected = dataset_id in connected_datasets
        if not is_connected and not include_orphan_datasets:
            continue

        node_id = f"dataset:{dataset_id}"

        # Create a display label
        label = dataset["title"]
        if label and len(label) > 50:
            label = label[:47] + "..."
        elif not label:
            label = dataset_id

        nodes.append({
            "id": node_id,
            "type": "dataset",
            "label": label,
            "properties": {
                "dataset_id": dataset_id,
                "title": dataset["title"],
                "description": dataset["description"],
                "species": dataset["species"],
                "repository": dataset["repository"]
            }
        })

    # Calculate counts
    article_count = sum(1 for n in nodes if n["type"] == "article")
    dataset_count = sum(1 for n in nodes if n["type"] == "dataset")

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "article_count": article_count,
            "dataset_count": dataset_count
        }
    }


def export_knowledge_graph_json(
    output_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    include_orphan_articles: bool = True,
    include_orphan_datasets: bool = True,
    indent: int | None = 2
) -> dict[str, Any]:
    """
    Generate and export the knowledge graph to a JSON file.

    Parameters
    ----------
    output_path : str or Path
        Path where the JSON file will be written.
    db_path : str or Path, optional
        Path to the SQLite database. Defaults to the project's articles.sqlite.
    include_orphan_articles : bool, optional
        Whether to include articles that have no associated datasets.
        Default is True.
    include_orphan_datasets : bool, optional
        Whether to include datasets that have no associated articles.
        Default is True.
    indent : int or None, optional
        JSON indentation level. Use None for compact output. Default is 2.

    Returns
    -------
    dict
        The generated knowledge graph dictionary.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    sqlite3.Error
        If there is an error querying the database.
    IOError
        If there is an error writing the output file.
    """
    graph = build_knowledge_graph(
        db_path=db_path,
        include_orphan_articles=include_orphan_articles,
        include_orphan_datasets=include_orphan_datasets
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=indent, ensure_ascii=False)

    return graph


def convert_to_cytoscape_format(graph: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the knowledge graph to Cytoscape.js elements format.

    Parameters
    ----------
    graph : dict
        Knowledge graph in the standard format (from build_knowledge_graph).

    Returns
    -------
    dict
        Cytoscape.js compatible format with "elements" containing "nodes" and "edges".

    Examples
    --------
    >>> graph = build_knowledge_graph()
    >>> cytoscape_data = convert_to_cytoscape_format(graph)
    >>> # cytoscape_data["elements"]["nodes"] and cytoscape_data["elements"]["edges"]
    """
    cytoscape_nodes = []
    for node in graph["nodes"]:
        cytoscape_nodes.append({
            "data": {
                "id": node["id"],
                "label": node["label"],
                "type": node["type"],
                **node["properties"]
            }
        })

    cytoscape_edges = []
    for edge in graph["edges"]:
        cytoscape_edges.append({
            "data": {
                "source": edge["source"],
                "target": edge["target"],
                "type": edge["type"],
                **edge.get("properties", {})
            }
        })

    return {
        "elements": {
            "nodes": cytoscape_nodes,
            "edges": cytoscape_edges
        },
        "metadata": graph["metadata"]
    }


def convert_to_visjs_format(graph: dict[str, Any]) -> dict[str, Any]:
    """
    Convert the knowledge graph to vis.js Network format.

    Parameters
    ----------
    graph : dict
        Knowledge graph in the standard format (from build_knowledge_graph).

    Returns
    -------
    dict
        vis.js Network compatible format with "nodes" and "edges" arrays.
        Nodes include visual styling hints (group, title).
        Edges include directional arrow configuration.

    Examples
    --------
    >>> graph = build_knowledge_graph()
    >>> visjs_data = convert_to_visjs_format(graph)
    >>> # Pass visjs_data["nodes"] and visjs_data["edges"] to vis.Network
    """
    visjs_nodes = []
    for node in graph["nodes"]:
        visjs_nodes.append({
            "id": node["id"],
            "label": node["label"],
            "group": node["type"],  # vis.js uses groups for styling
            "title": node["properties"].get("title") or node["label"],  # tooltip
            **{k: v for k, v in node["properties"].items() if v is not None}
        })

    visjs_edges = []
    for i, edge in enumerate(graph["edges"]):
        visjs_edges.append({
            "id": f"edge_{i}",
            "from": edge["source"],
            "to": edge["target"],
            "label": edge["type"],
            "arrows": "to",  # Directional arrow
            **{k: v for k, v in edge.get("properties", {}).items() if v is not None}
        })

    return {
        "nodes": visjs_nodes,
        "edges": visjs_edges,
        "metadata": graph["metadata"]
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate knowledge graph JSON from articles database"
    )
    parser.add_argument(
        "-o", "--output",
        default="odda_utils.json",
        help="Output JSON file path (default: odda.json)"
    )
    parser.add_argument(
        "-d", "--database",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--no-orphan-articles",
        action="store_true",
        help="Exclude articles without dataset connections"
    )
    parser.add_argument(
        "--no-orphan-datasets",
        action="store_true",
        help="Exclude datasets without article connections"
    )
    parser.add_argument(
        "--format",
        choices=["standard", "cytoscape", "visjs"],
        default="standard",
        help="Output format (default: standard)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation)"
    )

    args = parser.parse_args()

    try:
        graph = build_knowledge_graph(
            db_path=args.database,
            include_orphan_articles=not args.no_orphan_articles,
            include_orphan_datasets=not args.no_orphan_datasets
        )

        # Convert format if needed
        if args.format == "cytoscape":
            graph = convert_to_cytoscape_format(graph)
        elif args.format == "visjs":
            graph = convert_to_visjs_format(graph)

        # Write output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        indent = None if args.compact else 2
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=indent, ensure_ascii=False)

        print(f"Knowledge graph exported to: {output_path}")
        print(f"  Nodes: {graph.get('metadata', {}).get('node_count', len(graph.get('nodes', graph.get('elements', {}).get('nodes', []))))}")
        print(f"  Edges: {graph.get('metadata', {}).get('edge_count', len(graph.get('edges', graph.get('elements', {}).get('edges', []))))}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        exit(1)
