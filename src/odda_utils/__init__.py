"""Knowledge Graph package.

This package provides functionality for building and visualizing knowledge graphs
from scientific article metadata stored in a SQLite database.
"""

from odda_utils.kg import (
    build_knowledge_graph,
    build_knowledge_graph_json,
    find_datasets,
)

__version__ = "0.1.0"

__all__ = [
    "build_knowledge_graph",
    "build_knowledge_graph_json",
    "find_datasets",
]

# Visualization functions are optional - only import if dependencies available
try:
    from odda_utils.visualization import (
        visualize_knowledge_graph,
        visualize_knowledge_graph_from_json,
    )
    __all__.extend([
        "visualize_knowledge_graph",
        "visualize_knowledge_graph_from_json",
    ])
except ImportError:
    # Visualization dependencies not installed
    pass
