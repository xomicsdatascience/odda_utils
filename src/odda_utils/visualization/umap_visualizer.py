# UMAP visualization of article embeddings.
#
# This module provides functionality to create 2D UMAP visualizations of article
# embeddings from the SQLite database. Points can optionally be highlighted by
# providing a list of DOIs, which will be displayed with larger markers.

import sqlite3
import struct
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray
from umap import UMAP


# Default database path relative to project root
# From src/knowledge_graph/visualization/ go up to mcp/ project root
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent.parent / "articles.sqlite"


def _blob_to_embedding(blob: bytes) -> list[float]:
    """
    Convert a binary blob back to an embedding list.

    Parameters
    ----------
    blob : bytes
        Binary representation of the embedding.

    Returns
    -------
    list of float
        List of floats representing the embedding vector.
    """
    count = len(blob) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", blob))


def fetch_embeddings_with_metadata(
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """
    Fetch all embeddings with article metadata from the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.

    Returns
    -------
    list of dict
        List of dictionaries containing embedding and metadata with keys:
        doi, pmid, pmcid, title, embedding (as list of floats).
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.doi, e.pmid, e.pmcid, e.embedding, a.title
        FROM embeddings e
        LEFT JOIN articles a ON (
            (e.doi IS NOT NULL AND e.doi = a.doi) OR
            (e.pmid IS NOT NULL AND e.pmid = a.pmid) OR
            (e.pmcid IS NOT NULL AND e.pmcid = a.pmcid)
        )
        WHERE e.embedding IS NOT NULL
    """)

    results = []
    for row in cursor.fetchall():
        embedding = _blob_to_embedding(row[3])
        results.append({
            "doi": row[0],
            "pmid": row[1],
            "pmcid": row[2],
            "embedding": embedding,
            "title": row[4],
        })

    return results


def compute_umap(
    embeddings: NDArray[np.float32],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int | None = 42,
) -> NDArray[np.float32]:
    """
    Compute 2D UMAP projection of embeddings.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, n_features)
        Input embeddings matrix.
    n_neighbors : int, optional
        Number of neighbors for UMAP. Default is 15.
    min_dist : float, optional
        Minimum distance for UMAP. Default is 0.1.
    metric : str, optional
        Distance metric to use. Default is "cosine".
    random_state : int or None, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    ndarray of shape (n_samples, 2)
        2D UMAP coordinates.
    """
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def create_umap_visualization(
    db_path: str | Path = DEFAULT_DB_PATH,
    highlight_dois: list[str] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int | None = 42,
    figsize: tuple[float, float] = (12, 10),
    base_marker_size: float = 30,
    highlight_scale: float = 2.0,
    title: str = "UMAP of Article Embeddings",
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    """
    Create a UMAP visualization of article embeddings.

    Generates a 2D scatter plot of article embeddings using UMAP dimensionality
    reduction. Optionally highlights specific articles by DOI with larger markers.

    Parameters
    ----------
    db_path : str or Path, optional
        Path to the SQLite database. Defaults to the project's articles.sqlite.
    highlight_dois : list of str, optional
        List of DOIs to highlight with larger markers. If None, all points
        are displayed with the same size.
    n_neighbors : int, optional
        Number of neighbors for UMAP. Default is 15.
    min_dist : float, optional
        Minimum distance for UMAP. Default is 0.1.
    metric : str, optional
        Distance metric for UMAP. Default is "cosine".
    random_state : int or None, optional
        Random seed for reproducibility. Default is 42.
    figsize : tuple of float, optional
        Figure size (width, height) in inches. Default is (12, 10).
    base_marker_size : float, optional
        Base size for scatter plot markers. Default is 30.
    highlight_scale : float, optional
        Scale factor for highlighted points (relative to base_marker_size).
        Default is 2.0.
    title : str, optional
        Plot title. Default is "UMAP of Article Embeddings".
    show_legend : bool, optional
        Whether to show the legend. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure.
    ax : matplotlib.axes.Axes
        The matplotlib axes.
    metadata : dict
        Dictionary containing:
        - umap_coords: ndarray of shape (n_samples, 2) with UMAP coordinates
        - articles: list of article metadata dicts
        - highlight_indices: list of indices for highlighted articles
        - n_total: total number of articles
        - n_highlighted: number of highlighted articles

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    ValueError
        If no embeddings are found in the database.

    Examples
    --------
    >>> fig, ax, meta = create_umap_visualization()
    >>> plt.show()

    >>> fig, ax, meta = create_umap_visualization(
    ...     highlight_dois=["10.1234/example1", "10.1234/example2"],
    ...     title="My Article Embeddings"
    ... )
    >>> fig.savefig("umap.png", dpi=300)
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # Fetch embeddings
    conn = sqlite3.connect(str(db_path))
    try:
        articles = fetch_embeddings_with_metadata(conn)
    finally:
        conn.close()

    if not articles:
        raise ValueError("No embeddings found in the database")

    # Convert to numpy array
    embeddings = np.array([a["embedding"] for a in articles], dtype=np.float32)

    # Compute UMAP
    umap_coords = compute_umap(
        embeddings,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    # Determine which points to highlight
    highlight_set = set(highlight_dois) if highlight_dois else set()
    highlight_indices = []
    regular_indices = []

    for i, article in enumerate(articles):
        if article["doi"] and article["doi"] in highlight_set:
            highlight_indices.append(i)
        else:
            regular_indices.append(i)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot regular points
    if regular_indices:
        regular_coords = umap_coords[regular_indices]
        ax.scatter(
            regular_coords[:, 0],
            regular_coords[:, 1],
            s=base_marker_size,
            c="#4a90d9",
            alpha=0.6,
            label=f"Articles (n={len(regular_indices)})",
            edgecolors="white",
            linewidths=0.5,
        )

    # Plot highlighted points
    if highlight_indices:
        highlight_coords = umap_coords[highlight_indices]
        ax.scatter(
            highlight_coords[:, 0],
            highlight_coords[:, 1],
            s=base_marker_size * highlight_scale,
            c="#e63946",
            alpha=0.9,
            label=f"Highlighted (n={len(highlight_indices)})",
            edgecolors="white",
            linewidths=1.0,
            zorder=10,
        )

        # Draw a single red rectangle around all highlighted points
        x_min, x_max = highlight_coords[:, 0].min(), highlight_coords[:, 0].max()
        y_min, y_max = highlight_coords[:, 1].min(), highlight_coords[:, 1].max()
        padding_x = (x_max - x_min) * 0.05 + 0.3
        padding_y = (y_max - y_min) * 0.05 + 0.3
        rect = Rectangle(
            (x_min - padding_x, y_min - padding_y),
            (x_max - x_min) + 2 * padding_x,
            (y_max - y_min) + 2 * padding_y,
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
            zorder=11,
        )
        ax.add_patch(rect)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14)

    if show_legend:
        ax.legend(loc="best", fontsize=10)

    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    metadata = {
        "umap_coords": umap_coords,
        "articles": articles,
        "highlight_indices": highlight_indices,
        "n_total": len(articles),
        "n_highlighted": len(highlight_indices),
    }

    return fig, ax, metadata


def save_umap_visualization(
    output_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    highlight_dois: list[str] | None = None,
    dpi: int = 300,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generate and save a UMAP visualization to a file.

    Parameters
    ----------
    output_path : str or Path
        Path where the figure will be saved. Format is inferred from extension.
    db_path : str or Path, optional
        Path to the SQLite database. Defaults to the project's articles.sqlite.
    highlight_dois : list of str, optional
        List of DOIs to highlight with larger markers.
    dpi : int, optional
        Resolution for raster formats. Default is 300.
    **kwargs : Any
        Additional arguments passed to create_umap_visualization.

    Returns
    -------
    dict
        Metadata dictionary from create_umap_visualization.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    ValueError
        If no embeddings are found in the database.
    """
    fig, ax, metadata = create_umap_visualization(
        db_path=db_path,
        highlight_dois=highlight_dois,
        **kwargs,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate UMAP visualization of article embeddings"
    )
    parser.add_argument(
        "-o", "--output",
        default="umap_articles.png",
        help="Output file path (default: umap_articles.png)"
    )
    parser.add_argument(
        "-d", "--database",
        default=str(DEFAULT_DB_PATH),
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--highlight",
        nargs="+",
        help="DOIs to highlight (space-separated)"
    )
    parser.add_argument(
        "--highlight-file",
        type=str,
        help="File containing DOIs to highlight (one per line)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP (default: 15)"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP (default: 0.1)"
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean", "manhattan", "correlation"],
        help="Distance metric for UMAP (default: cosine)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution in DPI (default: 300)"
    )
    parser.add_argument(
        "--title",
        default="UMAP of Article Embeddings",
        help="Plot title"
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide the legend"
    )

    args = parser.parse_args()

    # Collect DOIs to highlight
    highlight_dois = []
    if args.highlight:
        highlight_dois.extend(args.highlight)
    if args.highlight_file:
        with open(args.highlight_file, "r") as f:
            for line in f:
                doi = line.strip()
                if doi:
                    highlight_dois.append(doi)

    try:
        metadata = save_umap_visualization(
            output_path=args.output,
            db_path=args.database,
            highlight_dois=highlight_dois if highlight_dois else None,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
            dpi=args.dpi,
            title=args.title,
            show_legend=not args.no_legend,
        )

        print(f"UMAP visualization saved to: {args.output}")
        print(f"  Total articles: {metadata['n_total']}")
        print(f"  Highlighted: {metadata['n_highlighted']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
