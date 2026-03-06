"""Knowledge graph visualization functions.

This module provides functions for visualizing knowledge graphs built from the
article database. It supports highlighting specific articles by DOI and uses
networkx for graph layout with matplotlib for rendering.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default colors for different node types
NODE_COLORS = {
    "article": "#4A90D9",      # Blue
    "author": "#7CB342",       # Green
    "keyword": "#FFA726",      # Orange
    "mesh": "#AB47BC",         # Purple
    "dataset": "#26A69A",      # Teal
    "journal": "#EF5350",      # Red
}

# Default base sizes for different node types
NODE_BASE_SIZES = {
    "article": 300,
    "author": 150,
    "keyword": 100,
    "mesh": 100,
    "dataset": 200,
    "journal": 250,
}


def visualize_knowledge_graph(
    graph_data: dict[str, Any],
    highlight_dois: list[str] | None = None,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 12),
    show_labels: bool = True,
    layout: str = "spring",
    title: str | None = None,
    node_colors: dict[str, str] | None = None,
    node_base_sizes: dict[str, int] | None = None,
    highlight_scale: float = 2.0,
) -> Any:
    """Visualize a knowledge graph with optional DOI highlighting.

    Creates a visual representation of the knowledge graph using networkx and
    matplotlib. Articles matching the provided DOIs are highlighted by making
    their nodes larger (by default, twice as large).

    Parameters
    ----------
    graph_data : dict
        Knowledge graph dictionary as produced by build_knowledge_graph().
        Must contain "nodes" and "edges" keys.
    highlight_dois : list of str, optional
        List of DOIs to highlight. Articles with these DOIs will have their
        nodes displayed at highlight_scale times the normal size.
    output_path : str or Path, optional
        Path to save the visualization image. If None, the figure is displayed
        interactively (if running in an environment that supports it).
    figsize : tuple of int, optional
        Figure size as (width, height) in inches. Default is (16, 12).
    show_labels : bool, optional
        Whether to show node labels. Default is True. May want to disable for
        large graphs.
    layout : str, optional
        Graph layout algorithm to use. Options are:
        - "spring": Force-directed layout (default)
        - "kamada_kawai": Kamada-Kawai layout (better for smaller graphs)
        - "circular": Circular layout
        - "shell": Shell layout
        - "spectral": Spectral layout
    title : str, optional
        Title for the visualization. If None, a default title is generated.
    node_colors : dict, optional
        Custom colors for node types. Keys are node types (article, author,
        keyword, mesh, dataset, journal), values are color strings.
    node_base_sizes : dict, optional
        Custom base sizes for node types. Keys are node types, values are
        integer sizes.
    highlight_scale : float, optional
        Scale factor for highlighted nodes. Default is 2.0 (twice as large).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the visualization.

    Raises
    ------
    ImportError
        If networkx or matplotlib are not installed.
    ValueError
        If the graph_data is missing required keys or has invalid structure.

    Examples
    --------
    >>> from odda_utils.kg import build_knowledge_graph
    >>> from odda_utils.visualization import visualize_knowledge_graph
    >>> graph = build_knowledge_graph("./articles.sqlite")
    >>> fig = visualize_knowledge_graph(
    ...     graph,
    ...     highlight_dois=["10.1234/example.doi"],
    ...     output_path="odda_utils.png"
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "Visualization requires networkx and matplotlib. "
            "Install them with: pip install networkx matplotlib"
        ) from e

    # Validate input
    if "nodes" not in graph_data or "edges" not in graph_data:
        raise ValueError("graph_data must contain 'nodes' and 'edges' keys")

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    if not nodes:
        logger.warning("Empty graph - no nodes to visualize")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty graph - no nodes to visualize",
                ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # Set up colors and sizes
    colors = {**NODE_COLORS, **(node_colors or {})}
    base_sizes = {**NODE_BASE_SIZES, **(node_base_sizes or {})}

    # Normalize highlight DOIs
    highlight_set = set()
    if highlight_dois:
        for doi in highlight_dois:
            # Normalize DOI format - could be with or without prefix
            normalized = doi.lower().strip()
            highlight_set.add(normalized)
            # Also try without common prefixes
            if normalized.startswith("https://doi.org/"):
                highlight_set.add(normalized[16:])
            elif normalized.startswith("http://doi.org/"):
                highlight_set.add(normalized[15:])
            elif normalized.startswith("doi:"):
                highlight_set.add(normalized[4:])
            # Also add the original
            highlight_set.add(doi)

    # Create networkx graph
    G = nx.Graph()

    # Build node ID to data mapping
    node_map = {node["id"]: node for node in nodes}

    # Add nodes
    for node in nodes:
        G.add_node(node["id"], **node)

    # Add edges
    for edge in edges:
        if edge["source"] in node_map and edge["target"] in node_map:
            G.add_edge(edge["source"], edge["target"],
                      edge_type=edge.get("type", ""),
                      **edge.get("attributes", {}))

    # Determine which articles to highlight
    highlighted_nodes = set()
    for node in nodes:
        if node["type"] == "article":
            node_doi = node.get("attributes", {}).get("doi")
            if node_doi:
                # Check if this DOI should be highlighted
                doi_lower = node_doi.lower()
                if doi_lower in highlight_set or node_doi in highlight_set:
                    highlighted_nodes.add(node["id"])

    # Calculate node colors and sizes
    node_color_list = []
    node_size_list = []

    for node_id in G.nodes():
        node_data = node_map.get(node_id, {})
        node_type = node_data.get("type", "article")

        # Get color
        color = colors.get(node_type, "#888888")
        node_color_list.append(color)

        # Get size - apply highlight scale if node is highlighted
        base_size = base_sizes.get(node_type, 100)
        if node_id in highlighted_nodes:
            node_size_list.append(base_size * highlight_scale)
        else:
            node_size_list.append(base_size)

    # Calculate layout
    logger.info(f"Computing {layout} layout for {len(G.nodes())} nodes...")
    if layout == "spring":
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        logger.warning(f"Unknown layout '{layout}', using spring layout")
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.3,
        edge_color="#CCCCCC",
        width=0.5,
        ax=ax
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color_list,
        node_size=node_size_list,
        alpha=0.8,
        ax=ax
    )

    # Draw highlighted nodes with edge highlight
    if highlighted_nodes:
        highlighted_pos = {k: v for k, v in pos.items() if k in highlighted_nodes}
        highlighted_sizes = [
            base_sizes.get(node_map.get(n, {}).get("type", "article"), 100) * highlight_scale
            for n in highlighted_pos
        ]
        highlighted_colors = [
            colors.get(node_map.get(n, {}).get("type", "article"), "#4A90D9")
            for n in highlighted_pos
        ]
        # Draw a ring around highlighted nodes
        nx.draw_networkx_nodes(
            G, highlighted_pos,
            nodelist=list(highlighted_pos.keys()),
            node_color="none",
            edgecolors="#FF0000",
            linewidths=3,
            node_size=[s * 1.2 for s in highlighted_sizes],
            ax=ax
        )

    # Draw labels if requested
    if show_labels:
        # Create label dict with truncated labels for readability
        labels = {}
        for node_id in G.nodes():
            node_data = node_map.get(node_id, {})
            label = node_data.get("label", node_id)
            if label and len(label) > 25:
                label = label[:22] + "..."
            labels[node_id] = label

        # Adjust font size based on graph size
        font_size = max(6, min(10, 200 / len(G.nodes())))
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=font_size,
            font_weight="normal",
            alpha=0.7,
            ax=ax
        )

    # Add legend
    legend_elements = []
    from matplotlib.patches import Patch
    for node_type, color in colors.items():
        count = sum(1 for n in nodes if n.get("type") == node_type)
        if count > 0:
            legend_elements.append(
                Patch(facecolor=color, edgecolor=color,
                      label=f"{node_type.capitalize()} ({count})")
            )

    if highlighted_nodes:
        legend_elements.append(
            Patch(facecolor="none", edgecolor="#FF0000", linewidth=2,
                  label=f"Highlighted ({len(highlighted_nodes)})")
        )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        metadata = graph_data.get("metadata", {})
        node_count = metadata.get("node_count", len(nodes))
        edge_count = metadata.get("edge_count", len(edges))
        ax.set_title(
            f"Knowledge Graph ({node_count} nodes, {edge_count} edges)",
            fontsize=14, fontweight="bold"
        )

    ax.axis("off")
    plt.tight_layout()

    # Save or display
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")

    return fig


def visualize_knowledge_graph_from_json(
    graph_json: str,
    highlight_dois: list[str] | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Visualize a knowledge graph from a JSON string.

    Convenience function that parses a JSON string and visualizes it.

    Parameters
    ----------
    graph_json : str
        JSON string representation of the knowledge graph.
    highlight_dois : list of str, optional
        List of DOIs to highlight.
    output_path : str or Path, optional
        Path to save the visualization image.
    **kwargs
        Additional arguments passed to visualize_knowledge_graph().

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object containing the visualization.
    """
    import json
    graph_data = json.loads(graph_json)
    return visualize_knowledge_graph(
        graph_data,
        highlight_dois=highlight_dois,
        output_path=output_path,
        **kwargs
    )
