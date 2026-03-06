# Fetch code repositories (GitHub, BitBucket) associated with scientific articles.
#
# This module provides functionality to:
# - Clone code repositories from GitHub and BitBucket URLs
# - Safely retrieve repository metadata without examining code contents (code treated as hostile)
# - Calculate repository size and file count using git commands
# - Fetch primary programming language via external APIs (GitHub/BitBucket)
# - Log repository information to the llm_code database table
#
# Security note: Repository contents are never examined or executed. Metadata is obtained
# either through git commands on the cloned repository or via external APIs.

import logging
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("./articles.sqlite")
DEFAULT_CODE_DIR = Path("/data/code/")

# Regex patterns for parsing repository URLs
GITHUB_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
BITBUCKET_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?bitbucket\.org/([^/]+)/([^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
GITLAB_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?gitlab\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


@dataclass
class RepositoryInfo:
    """Information about a code repository."""

    url: str
    owner: Optional[str] = None
    repo_name: Optional[str] = None
    platform: Optional[str] = None  # "github", "bitbucket", "gitlab", "unknown"
    primary_language: Optional[str] = None
    size_mb: Optional[int] = None
    file_count: Optional[int] = None
    local_path: Optional[str] = None


@dataclass
class FetchRepositoryResult:
    """Result of fetching a code repository."""

    success: bool
    url: Optional[str] = None
    local_path: Optional[str] = None
    size_mb: Optional[int] = None
    file_count: Optional[int] = None
    primary_language: Optional[str] = None
    platform: Optional[str] = None
    owner: Optional[str] = None
    repo_name: Optional[str] = None
    db_record_id: Optional[int] = None
    error: Optional[str] = None


def _parse_repository_url(url: str) -> RepositoryInfo:
    """Parse a repository URL to extract owner, repo name, and platform.

    Parameters
    ----------
    url : str
        The repository URL to parse.

    Returns
    -------
    RepositoryInfo
        Parsed repository information.
    """
    info = RepositoryInfo(url=url)

    # Try GitHub
    match = GITHUB_URL_PATTERN.match(url)
    if match:
        info.owner = match.group(1)
        info.repo_name = match.group(2)
        info.platform = "github"
        return info

    # Try BitBucket
    match = BITBUCKET_URL_PATTERN.match(url)
    if match:
        info.owner = match.group(1)
        info.repo_name = match.group(2)
        info.platform = "bitbucket"
        return info

    # Try GitLab
    match = GITLAB_URL_PATTERN.match(url)
    if match:
        info.owner = match.group(1)
        info.repo_name = match.group(2)
        info.platform = "gitlab"
        return info

    # Unknown platform - try to extract from generic git URL
    parsed = urlparse(url)
    if parsed.path:
        parts = parsed.path.strip("/").rstrip(".git").split("/")
        if len(parts) >= 2:
            info.owner = parts[0]
            info.repo_name = parts[1]
    info.platform = "unknown"

    return info


async def _fetch_github_language(owner: str, repo: str) -> Optional[str]:
    """Fetch the primary language of a GitHub repository via API.

    Parameters
    ----------
    owner : str
        Repository owner/organization.
    repo : str
        Repository name.

    Returns
    -------
    str or None
        Primary programming language, or None if unable to fetch.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "MCP-Repo-Fetcher/1.0",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("language")
            else:
                logger.warning(
                    f"GitHub API returned status {response.status_code} for {owner}/{repo}"
                )
                return None
    except Exception as e:
        logger.warning(f"Failed to fetch GitHub language for {owner}/{repo}: {e}")
        return None


async def _fetch_bitbucket_language(owner: str, repo: str) -> Optional[str]:
    """Fetch the primary language of a BitBucket repository via API.

    Parameters
    ----------
    owner : str
        Repository owner/workspace.
    repo : str
        Repository name (slug).

    Returns
    -------
    str or None
        Primary programming language, or None if unable to fetch.
    """
    url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "MCP-Repo-Fetcher/1.0",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("language")
            else:
                logger.warning(
                    f"BitBucket API returned status {response.status_code} for {owner}/{repo}"
                )
                return None
    except Exception as e:
        logger.warning(f"Failed to fetch BitBucket language for {owner}/{repo}: {e}")
        return None


async def _fetch_gitlab_language(owner: str, repo: str) -> Optional[str]:
    """Fetch the primary language of a GitLab repository via API.

    Parameters
    ----------
    owner : str
        Repository owner/namespace.
    repo : str
        Repository name.

    Returns
    -------
    str or None
        Primary programming language, or None if unable to fetch.
    """
    # GitLab uses URL-encoded project path
    project_path = f"{owner}%2F{repo}"
    url = f"https://gitlab.com/api/v4/projects/{project_path}/languages"
    headers = {
        "Accept": "application/json",
        "User-Agent": "MCP-Repo-Fetcher/1.0",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                # GitLab returns a dict of language -> percentage
                # Return the language with highest percentage
                if data:
                    return max(data, key=data.get)
            else:
                logger.warning(
                    f"GitLab API returned status {response.status_code} for {owner}/{repo}"
                )
                return None
    except Exception as e:
        logger.warning(f"Failed to fetch GitLab language for {owner}/{repo}: {e}")
        return None


async def _fetch_primary_language(info: RepositoryInfo) -> Optional[str]:
    """Fetch the primary language for a repository using the appropriate API.

    Parameters
    ----------
    info : RepositoryInfo
        Repository information with platform, owner, and repo_name.

    Returns
    -------
    str or None
        Primary programming language, or None if unable to fetch.
    """
    if not info.owner or not info.repo_name:
        return None

    if info.platform == "github":
        return await _fetch_github_language(info.owner, info.repo_name)
    elif info.platform == "bitbucket":
        return await _fetch_bitbucket_language(info.owner, info.repo_name)
    elif info.platform == "gitlab":
        return await _fetch_gitlab_language(info.owner, info.repo_name)
    else:
        # Try GitHub first as a fallback for unknown platforms
        lang = await _fetch_github_language(info.owner, info.repo_name)
        if lang:
            return lang
        # Then try BitBucket
        return await _fetch_bitbucket_language(info.owner, info.repo_name)


def _clone_repository(url: str, dest_path: Path) -> bool:
    """Clone a repository to the specified path.

    Uses shallow clone (--depth 1) for efficiency. Does not execute any
    repository hooks or scripts.

    Parameters
    ----------
    url : str
        Repository URL to clone.
    dest_path : Path
        Destination directory for the cloned repository.

    Returns
    -------
    bool
        True if clone was successful, False otherwise.

    Raises
    ------
    subprocess.CalledProcessError
        If git clone fails.
    """
    # Ensure parent directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove destination if it exists
    if dest_path.exists():
        shutil.rmtree(dest_path)

    logger.info(f"Cloning repository {url} to {dest_path}")

    # Clone with depth 1 for efficiency
    # Use --config to disable hooks for security
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--config",
            "core.hooksPath=/dev/null",
            url,
            str(dest_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout
    )

    if result.returncode != 0:
        logger.error(f"Git clone failed: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, "git clone", result.stdout, result.stderr
        )

    return True


def _get_file_count(repo_path: Path) -> int:
    """Get the number of tracked files in a git repository.

    Uses git ls-files to count files without examining their contents.

    Parameters
    ----------
    repo_path : Path
        Path to the git repository.

    Returns
    -------
    int
        Number of tracked files in the repository.
    """
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        logger.warning(f"git ls-files failed: {result.stderr}")
        return 0

    # Count non-empty lines
    files = [f for f in result.stdout.strip().split("\n") if f]
    return len(files)


def _get_directory_size_mb(path: Path) -> int:
    """Calculate the total size of a directory in megabytes, rounded to nearest MB.

    Parameters
    ----------
    path : Path
        Path to the directory.

    Returns
    -------
    int
        Size in megabytes, rounded to the nearest integer.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip symbolic links
            if not os.path.islink(filepath):
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass  # File may have been deleted or inaccessible

    # Convert to MB and round to nearest integer
    size_mb = total_size / (1024 * 1024)
    return round(size_mb)


def _log_to_database(
    conn: sqlite3.Connection,
    doi: Optional[str],
    pmid: Optional[str],
    pmcid: Optional[str],
    url: str,
    size_mb: int,
    file_count: int,
    primary_language: Optional[str],
    local_path: str,
) -> int:
    """Log repository information to the llm_code database table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    doi : str, optional
        Article DOI.
    pmid : str, optional
        Article PubMed ID.
    pmcid : str, optional
        Article PubMed Central ID.
    url : str
        Repository URL.
    size_mb : int
        Repository size in megabytes (rounded).
    file_count : int
        Number of files in the repository.
    primary_language : str, optional
        Primary programming language.
    local_path : str
        Local filesystem path where repository was cloned.

    Returns
    -------
    int
        The ID of the inserted record.
    """
    fetched_at = datetime.utcnow().isoformat()

    cursor = conn.execute(
        """
        INSERT INTO llm_code (doi, pmid, pmcid, url, size_mb, file_count, primary_language, local_path, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doi, pmid, pmcid, url, size_mb, file_count, primary_language, local_path, fetched_at),
    )
    conn.commit()
    return cursor.lastrowid


def get_repository_info(url: str) -> RepositoryInfo:
    """Parse a repository URL and return its information.

    This is a synchronous function that only parses the URL without
    fetching any additional data.

    Parameters
    ----------
    url : str
        Repository URL to parse.

    Returns
    -------
    RepositoryInfo
        Parsed repository information including platform, owner, and repo name.
    """
    return _parse_repository_url(url)


async def fetch_repository(
    url: str,
    output_path: Optional[Path | str] = None,
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    code_dir: Path | str = DEFAULT_CODE_DIR,
    overwrite: bool = False,
) -> FetchRepositoryResult:
    """Fetch a code repository and log its information to the database.

    Clones the repository, calculates size and file count, fetches the primary
    language via external API, and logs all information to the llm_code table.

    The code is treated as hostile: contents are never examined or executed.
    Metadata is obtained through git commands or external APIs only.

    Parameters
    ----------
    url : str
        Repository URL to clone (GitHub, BitBucket, GitLab, or other git URLs).
    output_path : Path or str, optional
        Specific output path for the repository. If not provided, uses
        code_dir/{owner}_{repo_name}/ derived from the URL.
    doi : str, optional
        Article DOI to associate with this repository.
    pmid : str, optional
        Article PubMed ID to associate with this repository.
    pmcid : str, optional
        Article PubMed Central ID to associate with this repository.
    db_path : Path or str
        Path to the SQLite database file.
    code_dir : Path or str
        Base directory for storing cloned repositories.
    overwrite : bool
        If True, re-clone even if the repository already exists locally.

    Returns
    -------
    FetchRepositoryResult
        Result containing repository metadata and database record ID.
    """
    db_path = Path(db_path)
    code_dir = Path(code_dir)

    # Parse the repository URL
    info = _parse_repository_url(url)

    if not info.owner or not info.repo_name:
        return FetchRepositoryResult(
            success=False,
            url=url,
            error=f"Could not parse owner and repository name from URL: {url}",
        )

    # Determine output path
    if output_path:
        dest_path = Path(output_path)
    else:
        # Create a directory name from owner and repo
        dir_name = f"{info.owner}_{info.repo_name}"
        dest_path = code_dir / dir_name

    # Check if already exists
    if dest_path.exists() and not overwrite:
        logger.info(f"Repository already exists at {dest_path}, skipping clone")
    else:
        # Clone the repository
        try:
            _clone_repository(url, dest_path)
        except subprocess.CalledProcessError as e:
            return FetchRepositoryResult(
                success=False,
                url=url,
                platform=info.platform,
                owner=info.owner,
                repo_name=info.repo_name,
                error=f"Git clone failed: {e.stderr}",
            )
        except subprocess.TimeoutExpired:
            return FetchRepositoryResult(
                success=False,
                url=url,
                platform=info.platform,
                owner=info.owner,
                repo_name=info.repo_name,
                error="Git clone timed out after 10 minutes",
            )

    # Calculate size and file count
    size_mb = _get_directory_size_mb(dest_path)
    file_count = _get_file_count(dest_path)

    # Fetch primary language via API (safe, does not examine code)
    primary_language = await _fetch_primary_language(info)

    # Log to database
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Insert record
        record_id = _log_to_database(
            conn=conn,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            url=url,
            size_mb=size_mb,
            file_count=file_count,
            primary_language=primary_language,
            local_path=str(dest_path),
        )

        conn.close()

    except sqlite3.Error as e:
        return FetchRepositoryResult(
            success=False,
            url=url,
            local_path=str(dest_path),
            size_mb=size_mb,
            file_count=file_count,
            primary_language=primary_language,
            platform=info.platform,
            owner=info.owner,
            repo_name=info.repo_name,
            error=f"Database error: {e}",
        )

    return FetchRepositoryResult(
        success=True,
        url=url,
        local_path=str(dest_path),
        size_mb=size_mb,
        file_count=file_count,
        primary_language=primary_language,
        platform=info.platform,
        owner=info.owner,
        repo_name=info.repo_name,
        db_record_id=record_id,
    )
