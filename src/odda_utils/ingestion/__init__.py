# Data ingestion modules for classifying and cataloging omics datasets.
#
# This package provides tools for analyzing directory contents, classifying
# files by their role in omics analysis workflows, and storing results in
# the knowledge graph database. Supports inspection of archive contents
# (zip, tar.gz, etc.) with files represented as "archive.zip/internal/path.txt".

from odda_utils.ingestion.analyze_directory import (
    ArchiveFileInfo,
    DirectoryAnalysisResult,
    FileCategory,
    FileClassification,
    analyze_directory,
    classify_file_by_heuristics,
    classify_file_deep_llm,
    classify_files_shallow_llm,
    extract_file_from_archive,
    get_classification_summary,
    get_file_header,
    get_file_header_from_archive,
    is_supported_archive,
    list_archive_contents,
)

__all__ = [
    "ArchiveFileInfo",
    "DirectoryAnalysisResult",
    "FileCategory",
    "FileClassification",
    "analyze_directory",
    "classify_file_by_heuristics",
    "classify_file_deep_llm",
    "classify_files_shallow_llm",
    "extract_file_from_archive",
    "get_classification_summary",
    "get_file_header",
    "get_file_header_from_archive",
    "is_supported_archive",
    "list_archive_contents",
]
