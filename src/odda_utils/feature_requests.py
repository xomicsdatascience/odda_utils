# Feature request submission, verification, and status management using semantic embeddings.
# Supports status tracking including 'in_progress', 'implemented', and 'incomplete' statuses.
# The 'incomplete' status is used when a feature cannot be fully implemented due to external
# dependencies or other blockers.

import asyncio
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from odda_utils import llm


DEFAULT_DB_PATH = Path("./articles.sqlite")
DEFAULT_ENDPOINT_FILE = Path(".claude/azure.endpoint")
DEFAULT_API_KEY_FILE = Path(".claude/azure.key")


@dataclass
class FeatureRequestResult:
    """Result of submitting a feature request."""

    request_id: int
    agent_name: str
    request: str
    reason_for_request: Optional[str]
    embedding_generated: bool
    embedding_model: Optional[str]
    error: Optional[str] = None


@dataclass
class SimilarRequestResult:
    """Result of finding a similar feature request."""

    found: bool
    request_id: Optional[int] = None
    agent_name: Optional[str] = None
    request: Optional[str] = None
    reason_for_request: Optional[str] = None
    request_status: Optional[str] = None
    similarity_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ApprovedRequestResult:
    """Result of fetching the oldest approved request."""

    found: bool
    request_id: Optional[int] = None
    agent_name: Optional[str] = None
    request: Optional[str] = None
    reason_for_request: Optional[str] = None
    status_reason: Optional[str] = None
    creation_time: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MarkImplementedResult:
    """Result of marking a feature request as implemented."""

    success: bool
    request_id: int
    previous_status: Optional[str] = None
    new_status: str = "implemented"
    agent_name: Optional[str] = None
    request: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MarkInProgressResult:
    """Result of marking a feature request as in progress."""

    success: bool
    request_id: int
    previous_status: Optional[str] = None
    new_status: str = "in_progress"
    assigned_time: Optional[str] = None
    agent_name: Optional[str] = None
    request: Optional[str] = None
    error: Optional[str] = None


@dataclass
class MarkIncompleteResult:
    """Result of marking a feature request as incomplete.

    The incomplete status indicates that a feature could not be fully implemented
    due to external dependencies, missing APIs, or other blockers that prevent
    completion.
    """

    success: bool
    request_id: int
    previous_status: Optional[str] = None
    new_status: str = "incomplete"
    agent_name: Optional[str] = None
    request: Optional[str] = None
    status_reason: Optional[str] = None
    error: Optional[str] = None


def _embedding_to_blob(embedding: list[float]) -> bytes:
    """Convert an embedding list to a binary blob.

    Parameters
    ----------
    embedding : list[float]
        List of floats representing the embedding vector.

    Returns
    -------
    bytes
        Binary representation of the embedding.
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes) -> list[float]:
    """Convert a binary blob back to an embedding list.

    Parameters
    ----------
    blob : bytes
        Binary representation of the embedding.

    Returns
    -------
    list[float]
        List of floats representing the embedding vector.
    """
    count = len(blob) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", blob))


async def get_text_embedding_async(
    text: str,
    endpoint_file: Path | None = None,
    api_key_file: Path | None = None,
    deployment_name: str = "text-embedding-3-small",
    api_version: str = "2024-02-01",
) -> list[float]:
    """Get a text embedding via the configured embedding provider, asynchronously.

    Delegates to the provider-agnostic :mod:`odda_utils.llm` abstraction (run in a
    worker thread since ``llm.embed`` is synchronous). This replaces the module's
    former duplicate Azure-credential reader and direct Azure embeddings URL. The
    ``endpoint_file`` / ``api_key_file`` / ``deployment_name`` / ``api_version``
    arguments are Azure-OpenAI hints, honoured only when the resolved embedding
    provider is ``azure_openai``. For backward compatibility, if no credential
    files are supplied, the default ``.claude/azure.endpoint`` /
    ``.claude/azure.key`` files are used when they exist.

    Parameters
    ----------
    text : str
        The text to embed.
    endpoint_file : Path | None
        Path to file containing the Azure OpenAI endpoint URL.
    api_key_file : Path | None
        Path to file containing the Azure OpenAI API key.
    deployment_name : str
        Name of the embedding model deployment (azure_openai).
    api_version : str
        Azure OpenAI API version.

    Returns
    -------
    list[float]
        List of floats representing the embedding vector.

    Raises
    ------
    odda_utils.llm.ModelConfigError
        If no embedding provider is configured.
    odda_utils.llm.LLMProviderError
        If the embedding request fails.
    """
    # Preserve the historical default of reading credentials from .claude/ files
    # when present, while otherwise deferring to the canonical config/env path.
    if endpoint_file is None and DEFAULT_ENDPOINT_FILE.exists():
        endpoint_file = DEFAULT_ENDPOINT_FILE
    if api_key_file is None and DEFAULT_API_KEY_FILE.exists():
        api_key_file = DEFAULT_API_KEY_FILE

    result = await asyncio.to_thread(
        llm.embed,
        text,
        endpoint_file=endpoint_file,
        api_key_file=api_key_file,
        model=deployment_name,
        api_version=api_version,
    )
    return result.vector


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    vec1 : np.ndarray
        First vector.
    vec2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity score in range [-1, 1].
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


async def submit_feature_request(
    agent_name: str,
    request: str,
    reason_for_request: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    endpoint_file: Path | str | None = None,
    api_key_file: Path | str | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> FeatureRequestResult:
    """Submit a feature request to the database with an embedding.

    Inserts a new agent request into the database and generates a semantic
    embedding for the request text to enable similarity search.

    Parameters
    ----------
    agent_name : str
        Name of the agent submitting the request.
    request : str
        The feature request text.
    reason_for_request : str | None
        Optional reason explaining why the request is being made.
    db_path : Path | str
        Path to the SQLite database file.
    endpoint_file : Path | str | None
        Path to file containing the Azure OpenAI endpoint URL.
    api_key_file : Path | str | None
        Path to file containing the Azure OpenAI API key.
    embedding_model : str
        Name of the embedding model to use.

    Returns
    -------
    FeatureRequestResult
        Result containing the request ID and embedding status.
    """
    db_path = Path(db_path)
    endpoint_file = Path(endpoint_file) if endpoint_file else None
    api_key_file = Path(api_key_file) if api_key_file else None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Insert the request
        cursor = conn.execute(
            """
            INSERT INTO agent_requests (agent_name, request, reason_for_request)
            VALUES (?, ?, ?)
            """,
            (agent_name, request, reason_for_request),
        )
        request_id = cursor.lastrowid
        conn.commit()

        # Generate embedding
        embedding_generated = False
        error = None
        try:
            embedding = await get_text_embedding_async(
                request,
                endpoint_file=endpoint_file,
                api_key_file=api_key_file,
                deployment_name=embedding_model,
            )

            # Store embedding
            embedding_blob = _embedding_to_blob(embedding)
            conn.execute(
                """
                UPDATE agent_requests
                SET embedding = ?, embedding_model = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (embedding_blob, embedding_model, request_id),
            )
            conn.commit()
            embedding_generated = True

        except Exception as e:
            error = f"Embedding generation failed: {e}"

        return FeatureRequestResult(
            request_id=request_id,
            agent_name=agent_name,
            request=request,
            reason_for_request=reason_for_request,
            embedding_generated=embedding_generated,
            embedding_model=embedding_model if embedding_generated else None,
            error=error,
        )

    finally:
        conn.close()


async def verify_feature_request(
    request: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    endpoint_file: Path | str | None = None,
    api_key_file: Path | str | None = None,
    embedding_model: str = "text-embedding-3-small",
    similarity_threshold: float = 0.0,
) -> SimilarRequestResult:
    """Find the most similar existing feature request.

    Generates an embedding for the input request text and compares it against
    all existing requests in the database using cosine similarity. Only the
    request text is used for similarity matching (not the reason).

    Parameters
    ----------
    request : str
        The feature request text to check for duplicates.
    db_path : Path | str
        Path to the SQLite database file.
    endpoint_file : Path | str | None
        Path to file containing the Azure OpenAI endpoint URL.
    api_key_file : Path | str | None
        Path to file containing the Azure OpenAI API key.
    embedding_model : str
        Name of the embedding model to use.
    similarity_threshold : float
        Minimum similarity score to consider a match (0.0 to 1.0).
        If the best match is below this threshold, found will be False.

    Returns
    -------
    SimilarRequestResult
        Result containing the most similar request and its similarity score.
    """
    db_path = Path(db_path)
    endpoint_file = Path(endpoint_file) if endpoint_file else None
    api_key_file = Path(api_key_file) if api_key_file else None

    # Generate embedding for the query
    try:
        query_embedding = await get_text_embedding_async(
            request,
            endpoint_file=endpoint_file,
            api_key_file=api_key_file,
            deployment_name=embedding_model,
        )
        query_vector = np.array(query_embedding)

    except Exception as e:
        return SimilarRequestResult(
            found=False,
            error=f"Failed to generate query embedding: {e}",
        )

    # Search for similar requests
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(
            """
            SELECT id, agent_name, request, reason_for_request,
                   request_status, embedding
            FROM agent_requests
            WHERE embedding IS NOT NULL
            """
        )

        best_match = None
        best_score = -1.0

        for row in cursor:
            embedding = _blob_to_embedding(row["embedding"])
            embedding_vector = np.array(embedding)

            score = _cosine_similarity(query_vector, embedding_vector)

            if score > best_score:
                best_score = score
                best_match = row

        if best_match is None or best_score < similarity_threshold:
            return SimilarRequestResult(
                found=False,
                similarity_score=best_score if best_match else None,
            )

        return SimilarRequestResult(
            found=True,
            request_id=best_match["id"],
            agent_name=best_match["agent_name"],
            request=best_match["request"],
            reason_for_request=best_match["reason_for_request"],
            request_status=best_match["request_status"],
            similarity_score=best_score,
        )

    finally:
        conn.close()


def get_oldest_approved_request(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> ApprovedRequestResult:
    """Fetch the oldest approved feature request from the database.

    Retrieves the approved request with the earliest creation_time from the
    agent_requests table. This can be used to process approved requests in
    chronological order.

    Parameters
    ----------
    db_path : Path | str
        Path to the SQLite database file.

    Returns
    -------
    ApprovedRequestResult
        Result containing the oldest approved request details, or found=False
        if no approved requests exist.
    """
    db_path = Path(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(
            """
            SELECT id, agent_name, request, reason_for_request, status_reason,
                   creation_time, updated_at
            FROM agent_requests
            WHERE request_status = 'approved'
            ORDER BY creation_time ASC
            LIMIT 1
            """
        )

        row = cursor.fetchone()

        if row is None:
            return ApprovedRequestResult(found=False)

        return ApprovedRequestResult(
            found=True,
            request_id=row["id"],
            agent_name=row["agent_name"],
            request=row["request"],
            reason_for_request=row["reason_for_request"],
            status_reason=row["status_reason"],
            creation_time=row["creation_time"],
            updated_at=row["updated_at"],
        )

    except Exception as e:
        return ApprovedRequestResult(
            found=False,
            error=f"Database error: {e}",
        )

    finally:
        conn.close()


def mark_request_in_progress(
    request_id: int,
    status_reason: Optional[str] = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> MarkInProgressResult:
    """Mark a feature request as in progress.

    Updates the request_status of a feature request to 'in_progress' and sets
    the assigned_time to the current timestamp. This is typically called when
    an agent begins working on implementing a feature that was previously approved.

    Parameters
    ----------
    request_id : int
        The ID of the feature request to mark as in progress.
    status_reason : str, optional
        Reason or notes about the status change.
    db_path : Path | str
        Path to the SQLite database file.

    Returns
    -------
    MarkInProgressResult
        Result containing the operation status and request details.
        - success: True if the status was updated successfully
        - request_id: The ID of the request
        - previous_status: The status before the update
        - new_status: Always 'in_progress' if successful
        - assigned_time: The timestamp when the task was assigned
        - agent_name: Name of the agent that created the request
        - request: The feature request text
        - error: Error message if the operation failed
    """
    db_path = Path(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # First, fetch the current request to verify it exists and get its status
        cursor = conn.execute(
            """
            SELECT id, agent_name, request, request_status
            FROM agent_requests
            WHERE id = ?
            """,
            (request_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return MarkInProgressResult(
                success=False,
                request_id=request_id,
                error=f"Feature request with ID {request_id} not found",
            )

        previous_status = row["request_status"]
        agent_name = row["agent_name"]
        request_text = row["request"]

        # Check if already in progress
        if previous_status == "in_progress":
            return MarkInProgressResult(
                success=False,
                request_id=request_id,
                previous_status=previous_status,
                agent_name=agent_name,
                request=request_text,
                error="Feature request is already marked as in_progress",
            )

        # Check if the request is in an appropriate state to be marked as in progress
        # Only approved requests should be marked as in progress
        if previous_status != "approved":
            return MarkInProgressResult(
                success=False,
                request_id=request_id,
                previous_status=previous_status,
                agent_name=agent_name,
                request=request_text,
                error=(
                    f"Cannot mark request as in_progress: current status is "
                    f"'{previous_status}', but only 'approved' requests can be "
                    f"marked as in_progress"
                ),
            )

        # Update the status to in_progress and set assigned_time
        conn.execute(
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

        # Fetch the assigned_time
        cursor = conn.execute(
            "SELECT assigned_time FROM agent_requests WHERE id = ?",
            (request_id,),
        )
        updated_row = cursor.fetchone()
        assigned_time = updated_row["assigned_time"] if updated_row else None

        return MarkInProgressResult(
            success=True,
            request_id=request_id,
            previous_status=previous_status,
            new_status="in_progress",
            assigned_time=assigned_time,
            agent_name=agent_name,
            request=request_text,
        )

    except Exception as e:
        return MarkInProgressResult(
            success=False,
            request_id=request_id,
            error=f"Database error: {e}",
        )

    finally:
        conn.close()


def mark_request_implemented(
    request_id: int,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> MarkImplementedResult:
    """Mark a feature request as implemented.

    Updates the request_status of a feature request to 'implemented'. This is
    typically called after an agent has successfully implemented a feature
    that was previously approved or in progress.

    Parameters
    ----------
    request_id : int
        The ID of the feature request to mark as implemented.
    db_path : Path | str
        Path to the SQLite database file.

    Returns
    -------
    MarkImplementedResult
        Result containing the operation status and request details.
        - success: True if the status was updated successfully
        - request_id: The ID of the request
        - previous_status: The status before the update
        - new_status: Always 'implemented' if successful
        - agent_name: Name of the agent that created the request
        - request: The feature request text
        - error: Error message if the operation failed
    """
    db_path = Path(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # First, fetch the current request to verify it exists and get its status
        cursor = conn.execute(
            """
            SELECT id, agent_name, request, request_status
            FROM agent_requests
            WHERE id = ?
            """,
            (request_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return MarkImplementedResult(
                success=False,
                request_id=request_id,
                error=f"Feature request with ID {request_id} not found",
            )

        previous_status = row["request_status"]
        agent_name = row["agent_name"]
        request_text = row["request"]

        # Check if already implemented
        if previous_status == "implemented":
            return MarkImplementedResult(
                success=False,
                request_id=request_id,
                previous_status=previous_status,
                agent_name=agent_name,
                request=request_text,
                error="Feature request is already marked as implemented",
            )

        # Check if the request is in an appropriate state to be marked as implemented
        # Both 'approved' and 'in_progress' requests can be marked as implemented
        valid_previous_statuses = ("approved", "in_progress")
        if previous_status not in valid_previous_statuses:
            return MarkImplementedResult(
                success=False,
                request_id=request_id,
                previous_status=previous_status,
                agent_name=agent_name,
                request=request_text,
                error=(
                    f"Cannot mark request as implemented: current status is "
                    f"'{previous_status}', but only 'approved' or 'in_progress' "
                    f"requests can be marked as implemented"
                ),
            )

        # Update the status to implemented
        conn.execute(
            """
            UPDATE agent_requests
            SET request_status = 'implemented', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (request_id,),
        )
        conn.commit()

        return MarkImplementedResult(
            success=True,
            request_id=request_id,
            previous_status=previous_status,
            new_status="implemented",
            agent_name=agent_name,
            request=request_text,
        )

    except Exception as e:
        return MarkImplementedResult(
            success=False,
            request_id=request_id,
            error=f"Database error: {e}",
        )

    finally:
        conn.close()


def mark_request_incomplete(
    request_id: int,
    status_reason: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> MarkIncompleteResult:
    """Mark a feature request as incomplete.

    Updates the request_status of a feature request to 'incomplete'. This is
    used when a feature cannot be fully implemented due to external dependencies
    such as missing APIs, unavailable external services, or other blockers that
    prevent completion.

    The status_reason parameter is required to document why the feature could
    not be completed, which helps with future triage and re-evaluation.

    Parameters
    ----------
    request_id : int
        The ID of the feature request to mark as incomplete.
    status_reason : str
        Required explanation of why the feature could not be completed.
        This should describe the external dependency or blocker.
    db_path : Path | str
        Path to the SQLite database file.

    Returns
    -------
    MarkIncompleteResult
        Result containing the operation status and request details.
        - success: True if the status was updated successfully
        - request_id: The ID of the request
        - previous_status: The status before the update
        - new_status: Always 'incomplete' if successful
        - agent_name: Name of the agent that created the request
        - request: The feature request text
        - status_reason: The reason provided for marking as incomplete
        - error: Error message if the operation failed
    """
    db_path = Path(db_path)

    # Validate that status_reason is provided
    if not status_reason or not status_reason.strip():
        return MarkIncompleteResult(
            success=False,
            request_id=request_id,
            error="status_reason is required when marking a request as incomplete",
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # First, fetch the current request to verify it exists and get its status
        cursor = conn.execute(
            """
            SELECT id, agent_name, request, request_status
            FROM agent_requests
            WHERE id = ?
            """,
            (request_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return MarkIncompleteResult(
                success=False,
                request_id=request_id,
                error=f"Feature request with ID {request_id} not found",
            )

        previous_status = row["request_status"]
        agent_name = row["agent_name"]
        request_text = row["request"]

        # Check if already incomplete
        if previous_status == "incomplete":
            return MarkIncompleteResult(
                success=False,
                request_id=request_id,
                previous_status=previous_status,
                agent_name=agent_name,
                request=request_text,
                error="Feature request is already marked as incomplete",
            )

        # Check if the request is in an appropriate state to be marked as incomplete
        # Both 'approved' and 'in_progress' requests can be marked as incomplete
        valid_previous_statuses = ("approved", "in_progress")
        if previous_status not in valid_previous_statuses:
            return MarkIncompleteResult(
                success=False,
                request_id=request_id,
                previous_status=previous_status,
                agent_name=agent_name,
                request=request_text,
                error=(
                    f"Cannot mark request as incomplete: current status is "
                    f"'{previous_status}', but only 'approved' or 'in_progress' "
                    f"requests can be marked as incomplete"
                ),
            )

        # Update the status to incomplete with the reason
        conn.execute(
            """
            UPDATE agent_requests
            SET request_status = 'incomplete',
                status_reason = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status_reason.strip(), request_id),
        )
        conn.commit()

        return MarkIncompleteResult(
            success=True,
            request_id=request_id,
            previous_status=previous_status,
            new_status="incomplete",
            agent_name=agent_name,
            request=request_text,
            status_reason=status_reason.strip(),
        )

    except Exception as e:
        return MarkIncompleteResult(
            success=False,
            request_id=request_id,
            error=f"Database error: {e}",
        )

    finally:
        conn.close()
