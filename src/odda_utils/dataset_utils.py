# Utility functions for checking dataset existence in the filesystem.
# Checks for both directories and archive files in the configured dataset directory.

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Common archive extensions for datasets
ARCHIVE_EXTENSIONS = [".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".zip", ".7z"]


@dataclass
class DatasetExistsResult:
    """Result of checking if a dataset exists.

    Attributes
    ----------
    exists : bool
        True if the dataset directory or archive exists.
    dataset_id : str
        The dataset ID that was checked.
    path : Optional[str]
        Path to the dataset directory or archive if it exists.
    is_directory : bool
        True if the dataset exists as a directory.
    is_archive : bool
        True if the dataset exists as an archive file.
    archive_format : Optional[str]
        Archive extension if is_archive is True (e.g., ".tar.gz").
    error : Optional[str]
        Error message if something went wrong during the check.
    """

    exists: bool = False
    dataset_id: str = ""
    path: Optional[str] = None
    is_directory: bool = False
    is_archive: bool = False
    archive_format: Optional[str] = None
    error: Optional[str] = None


def check_dataset_exists(
    dataset_id: str,
    datasets_dir: str | Path = "/data/datasets/",
) -> DatasetExistsResult:
    """Check if a dataset exists as a directory or archive file.

    Looks for the dataset in the specified directory by checking if a
    subdirectory with the dataset ID exists, or if there is an archive
    file with the dataset ID as the base name.

    Parameters
    ----------
    dataset_id : str
        The dataset identifier to check (e.g., "PXD012345", "GSE12345").
    datasets_dir : str or Path
        Directory where datasets are stored. Defaults to "/data/datasets/".

    Returns
    -------
    DatasetExistsResult
        Result containing:
        - exists: True if found as directory or archive
        - dataset_id: The dataset ID that was checked
        - path: Full path to the dataset if it exists
        - is_directory: True if found as a directory
        - is_archive: True if found as an archive file
        - archive_format: Extension of the archive if applicable
        - error: Error message if something went wrong
    """
    result = DatasetExistsResult(dataset_id=dataset_id)

    if not dataset_id:
        result.error = "dataset_id cannot be empty"
        return result

    datasets_path = Path(datasets_dir)

    if not datasets_path.exists():
        result.error = f"Datasets directory does not exist: {datasets_dir}"
        return result

    if not datasets_path.is_dir():
        result.error = f"Datasets path is not a directory: {datasets_dir}"
        return result

    # Check for directory
    dataset_dir = datasets_path / dataset_id
    if dataset_dir.exists() and dataset_dir.is_dir():
        result.exists = True
        result.path = str(dataset_dir)
        result.is_directory = True
        return result

    # Check for archive files
    for ext in ARCHIVE_EXTENSIONS:
        archive_path = datasets_path / f"{dataset_id}{ext}"
        if archive_path.exists() and archive_path.is_file():
            result.exists = True
            result.path = str(archive_path)
            result.is_archive = True
            result.archive_format = ext
            return result

    # Not found
    return result
