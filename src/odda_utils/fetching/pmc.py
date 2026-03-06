"""PMC Open Access article fetching, with Europe PMC fallback for non-OA articles.

Provides functions to search PubMed, fetch article metadata, download full text
and supplemental materials from the PMC Open Access subset, and fall back to
Europe PMC's rendering service for articles that have a PMCID but are not in
the OA subset.
"""

import ftplib
import io
import logging
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import requests

from odda_utils.metadata import FullArticleMetadata, extract_full_metadata
from odda_utils.utils import ArticleIds, convert_ids

logger = logging.getLogger(__name__)

PMC_OA_SERVICE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPEPMC_RENDER_URL = "https://europepmc.org/backend/ptpmcrender.fcgi"

DateType = Literal["edat", "pdat", "mdat"]


@dataclass
class ArticleMetadata:
    """Metadata for a PubMed article."""

    pmid: str
    title: str | None = None
    abstract: str | None = None
    doi: str | None = None
    pmcid: str | None = None


def fetch_article_metadata(pmid: str) -> FullArticleMetadata:
    """Fetch full article metadata from PubMed.

    Args:
        pmid: PubMed ID.

    Returns:
        FullArticleMetadata with all available metadata fields including
        authors, journal info, keywords, MeSH terms, grants, etc.
    """
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
    }

    response = requests.get(PUBMED_EFETCH_URL, params=params, timeout=30)
    response.raise_for_status()

    return extract_full_metadata(response.content, pmid)


def search_pubmed(
    query: str,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    date_type: DateType = "edat",
    max_results: int = 10000,
) -> list[str]:
    """Search PubMed for articles matching a query.

    Args:
        query: PubMed articles query string.
        start_date: Start date for filtering (inclusive). Can be date object or
            string in YYYY/MM/DD or YYYY format.
        end_date: End date for filtering (inclusive). Can be date object or
            string in YYYY/MM/DD or YYYY format.
        date_type: Type of date to filter on:
            - "edat": Entrez date (date added to PubMed)
            - "pdat": Publication date
            - "mdat": Modification date
        max_results: Maximum number of results to return (default 10000).

    Returns:
        List of PubMed IDs (PMIDs) matching the query.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
    }

    if start_date is not None or end_date is not None:
        params["datetype"] = date_type

        if start_date is not None:
            if isinstance(start_date, date):
                params["mindate"] = start_date.strftime("%Y/%m/%d")
            else:
                params["mindate"] = start_date

        if end_date is not None:
            if isinstance(end_date, date):
                params["maxdate"] = end_date.strftime("%Y/%m/%d")
            else:
                params["maxdate"] = end_date

    response = requests.get(PUBMED_ESEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    esearch_result = data.get("esearchresult", {})
    id_list = esearch_result.get("idlist", [])

    return id_list


@dataclass
class DownloadResult:
    """Result of a PMC article download."""

    article_ids: ArticleIds
    text_filepath: Path | None = None
    supplementals_filepath: Path | None = None
    source: str | None = None
    error: str | None = None


@dataclass
class ArchiveFileInfo:
    """Information about a file within an archive."""

    name: str
    size: int
    is_dir: bool


@dataclass
class ArchiveContentsResult:
    """Result of listing archive contents."""

    archive_path: str
    files: list[ArchiveFileInfo]
    total_files: int
    total_size: int
    error: str | None = None


def list_archive_contents(archive_path: str | Path) -> ArchiveContentsResult:
    """List the contents of a .tar.gz archive without extracting.

    Examines a PMC supplementals archive (or any .tar.gz file) and returns
    information about all files and directories within it.

    Args:
        archive_path: Path to the .tar.gz archive file.

    Returns:
        ArchiveContentsResult containing:
        - archive_path: The path that was examined
        - files: List of ArchiveFileInfo with name, size, and is_dir
        - total_files: Count of files (excluding directories)
        - total_size: Total uncompressed size in bytes
        - error: Error message if the operation failed
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        return ArchiveContentsResult(
            archive_path=str(archive_path),
            files=[],
            total_files=0,
            total_size=0,
            error=f"Archive not found: {archive_path}",
        )

    if not archive_path.suffix == ".gz" and not str(archive_path).endswith(".tar.gz"):
        return ArchiveContentsResult(
            archive_path=str(archive_path),
            files=[],
            total_files=0,
            total_size=0,
            error=f"Not a .tar.gz archive: {archive_path}",
        )

    try:
        files = []
        total_size = 0
        file_count = 0

        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                files.append(
                    ArchiveFileInfo(
                        name=member.name,
                        size=member.size,
                        is_dir=member.isdir(),
                    )
                )
                if not member.isdir():
                    file_count += 1
                    total_size += member.size

        return ArchiveContentsResult(
            archive_path=str(archive_path),
            files=files,
            total_files=file_count,
            total_size=total_size,
        )

    except tarfile.TarError as e:
        return ArchiveContentsResult(
            archive_path=str(archive_path),
            files=[],
            total_files=0,
            total_size=0,
            error=f"Failed to read archive: {e}",
        )
    except Exception as e:
        return ArchiveContentsResult(
            archive_path=str(archive_path),
            files=[],
            total_files=0,
            total_size=0,
            error=f"Unexpected error: {e}",
        )


def download_pmc_article(
    output_dir: str | Path,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
) -> DownloadResult:
    """Download full text and supplementals from PMC Open Access, with Europe PMC fallback.

    First attempts to download from the PMC Open Access subset. If the article
    has a PMCID but is not available in the OA subset, falls back to Europe PMC's
    rendering service to retrieve the full text PDF and convert it to text.

    Provide exactly one of doi, pmid, or pmcid.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save downloaded files.
    doi : str, optional
        Digital Object Identifier.
    pmid : str, optional
        PubMed ID.
    pmcid : str, optional
        PubMed Central ID.

    Returns
    -------
    DownloadResult
        Result with paths to downloaded files. The ``source`` field indicates
        whether the article was obtained from ``"pmc_oa"`` (PMC Open Access
        archive) or ``"europepmc"`` (Europe PMC rendering service).

    Raises
    ------
    ValueError
        If no ID or multiple IDs provided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert IDs to get PMCID (required for PMC OA)
    article_ids = convert_ids(doi=doi, pmid=pmid, pmcid=pmcid)

    if not article_ids.pmcid:
        return DownloadResult(
            article_ids=article_ids,
            error="Could not obtain PMCID. Article may not be in PMC.",
        )

    # Query PMC OA service for download links
    oa_links = _get_oa_links(article_ids.pmcid)
    if oa_links is not None:
        result = DownloadResult(article_ids=article_ids, source="pmc_oa")

        # Download and extract article text from PMC OA archive
        if oa_links.get("tgz"):
            text_path, suppl_path = _download_and_extract(
                oa_links["tgz"],
                output_dir,
                article_ids.pmcid,
            )
            result.text_filepath = text_path
            result.supplementals_filepath = suppl_path

        return result

    # PMC OA not available -- try Europe PMC as fallback
    logger.info(
        "Article %s not in PMC OA subset, trying Europe PMC fallback",
        article_ids.pmcid,
    )
    return _download_europepmc_article(output_dir, article_ids)


def _download_europepmc_article(
    output_dir: Path,
    article_ids: ArticleIds,
) -> DownloadResult:
    """Download article full text PDF from Europe PMC and convert to text.

    Uses the Europe PMC rendering service endpoint to retrieve a PDF for
    articles that have a PMCID but are not in the PMC Open Access subset.
    The PDF is converted to plain text using pdfplumber.

    Parameters
    ----------
    output_dir : Path
        Directory to save the extracted text file.
    article_ids : ArticleIds
        Article identifiers; must have a non-None ``pmcid``.

    Returns
    -------
    DownloadResult
        Result with the path to the extracted text file, or an error message
        if the download or conversion failed.
    """
    pmcid = article_ids.pmcid

    # Build the Europe PMC PDF rendering URL
    pdf_url = f"{EUROPEPMC_RENDER_URL}?accid={pmcid}&blobtype=pdf"

    try:
        pdf_bytes = _download_europepmc_pdf(pdf_url)
    except requests.HTTPError as e:
        error_msg = f"Europe PMC PDF download failed for {pmcid}: HTTP {e.response.status_code}"
        logger.warning(error_msg)
        return DownloadResult(
            article_ids=article_ids,
            error=error_msg,
        )
    except requests.RequestException as e:
        error_msg = f"Europe PMC PDF download failed for {pmcid}: {e}"
        logger.warning(error_msg)
        return DownloadResult(
            article_ids=article_ids,
            error=error_msg,
        )

    # Validate that we actually received a PDF (check magic bytes)
    if not pdf_bytes or not pdf_bytes[:5] == b"%PDF-":
        error_msg = (
            f"Europe PMC did not return a valid PDF for {pmcid} "
            f"(received {len(pdf_bytes)} bytes)"
        )
        logger.warning(error_msg)
        return DownloadResult(
            article_ids=article_ids,
            error=error_msg,
        )

    # Convert PDF to text
    try:
        text_content = _extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        error_msg = f"Failed to extract text from Europe PMC PDF for {pmcid}: {e}"
        logger.warning(error_msg)
        return DownloadResult(
            article_ids=article_ids,
            error=error_msg,
        )

    # Check that we got meaningful text content
    if not text_content or len(text_content.strip()) < 100:
        error_msg = (
            f"Extracted text from Europe PMC PDF for {pmcid} is too short "
            f"({len(text_content.strip())} chars), PDF may be image-based"
        )
        logger.warning(error_msg)
        return DownloadResult(
            article_ids=article_ids,
            error=error_msg,
        )

    # Save the text file in the same format as PMC OA articles
    text_filepath = output_dir / f"{pmcid}.txt"
    text_filepath.write_text(text_content, encoding="utf-8")

    logger.info(
        "Downloaded article %s from Europe PMC (%d chars extracted from PDF)",
        pmcid,
        len(text_content),
    )

    return DownloadResult(
        article_ids=article_ids,
        text_filepath=text_filepath,
        source="europepmc",
    )


def _download_europepmc_pdf(pdf_url: str, timeout: int = 120) -> bytes:
    """Download a PDF from Europe PMC's rendering service.

    Parameters
    ----------
    pdf_url : str
        Full URL to the Europe PMC PDF rendering endpoint.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    bytes
        Raw PDF content.

    Raises
    ------
    requests.HTTPError
        If the HTTP request returns a non-2xx status code.
    requests.RequestException
        If the request fails due to network or other issues.
    """
    response = requests.get(pdf_url, timeout=timeout, stream=True)
    response.raise_for_status()

    # Read the full response content
    chunks = []
    for chunk in response.iter_content(chunk_size=8192):
        chunks.append(chunk)

    return b"".join(chunks)


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pdfplumber.

    Processes each page of the PDF sequentially and concatenates the
    extracted text with page separators.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.

    Returns
    -------
    str
        Extracted plain text from all pages of the PDF.

    Raises
    ------
    ImportError
        If pdfplumber is not installed.
    Exception
        If PDF parsing fails.
    """
    import pdfplumber

    text_parts = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    return "\n\n".join(text_parts)


def _get_oa_links(pmcid: str) -> dict[str, str] | None:
    """Get Open Access download links for an article.

    Args:
        pmcid: PubMed Central ID.

    Returns:
        Dictionary with format -> URL mappings, or None if not available.
        Prefers HTTPS links over FTP when both are available.
    """
    params = {"id": pmcid}
    response = requests.get(PMC_OA_SERVICE_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    # Check for errors
    error = root.find(".//error")
    if error is not None:
        return None

    # Extract links, collecting HTTPS and FTP separately
    https_links = {}
    ftp_links = {}
    for link in root.findall(".//link"):
        format_type = link.get("format")
        href = link.get("href")
        if format_type and href:
            if href.startswith("https://") or href.startswith("http://"):
                https_links[format_type] = href
            elif href.startswith("ftp://"):
                ftp_links[format_type] = href

    # Prefer HTTPS links, fall back to FTP
    links = {}
    all_formats = set(https_links.keys()) | set(ftp_links.keys())
    for fmt in all_formats:
        links[fmt] = https_links.get(fmt) or ftp_links.get(fmt)

    return links if links else None


def _download_file(url: str, dest_path: str, timeout: int = 120) -> None:
    """Download a file from HTTP(S) or FTP URL.

    Args:
        url: URL to download from (supports http, https, ftp).
        dest_path: Local path to save the file.
        timeout: Timeout in seconds.
    """
    if url.startswith("ftp://"):
        # Use ftplib for FTP with timeout support
        parsed = urlparse(url)
        with ftplib.FTP(timeout=timeout) as ftp:
            ftp.connect(parsed.hostname, parsed.port or 21)
            ftp.login()
            with open(dest_path, "wb") as f:
                ftp.retrbinary(f"RETR {parsed.path}", f.write)
    else:
        # Use requests for HTTP(S) for better timeout handling
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def _download_and_extract(
    tgz_url: str,
    output_dir: Path,
    pmcid: str,
) -> tuple[Path | None, Path | None]:
    """Download and extract article archive.

    Args:
        tgz_url: URL to the .tar.gz archive (supports http, https, ftp).
        output_dir: Directory to save extracted files.
        pmcid: PubMed Central ID for naming files.

    Returns:
        Tuple of (text_filepath, supplementals_filepath).
    """
    text_filepath = None
    supplementals = []

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    # Download the archive
    _download_file(tgz_url, tmp_path)

    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    name_lower = member.name.lower()

                    # Main article text (NXML or XML format)
                    if name_lower.endswith(".nxml") or (
                        name_lower.endswith(".xml") and "supp" not in name_lower
                    ):
                        extracted = tar.extractfile(member)
                        if extracted:
                            text_content = _extract_text_from_nxml(extracted.read())
                            text_filepath = output_dir / f"{pmcid}.txt"
                            text_filepath.write_text(text_content, encoding="utf-8")

                    # Supplemental files
                    elif _is_supplemental(member.name):
                        supplementals.append(member)

            # Extract supplementals to archive
            supplementals_filepath = None
            if supplementals:
                supplementals_filepath = output_dir / f"{pmcid}_supplementals.tar.gz"
                with tarfile.open(supplementals_filepath, "w:gz") as suppl_tar:
                    for member in supplementals:
                        extracted = tar.extractfile(member)
                        if extracted:
                            suppl_tar.addfile(member, extracted)

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return text_filepath, supplementals_filepath


def _is_supplemental(filename: str) -> bool:
    """Check if a file is a supplemental material."""
    name_lower = filename.lower()
    suppl_indicators = ["supp", "supplement", "si_", "s1_", "s2_", "s3_", "s4_"]
    suppl_extensions = [".pdf", ".xlsx", ".xls", ".csv", ".zip", ".doc", ".docx"]

    # Check for supplemental indicators in name
    for indicator in suppl_indicators:
        if indicator in name_lower:
            return True

    # Check for common supplemental file extensions (excluding main article formats)
    for ext in suppl_extensions:
        if name_lower.endswith(ext):
            return True

    return False


def _extract_text_from_nxml(nxml_content: bytes) -> str:
    """Extract plain text from NXML/JATS format.

    Args:
        nxml_content: Raw NXML/XML content.

    Returns:
        Extracted plain text.
    """
    try:
        root = ET.fromstring(nxml_content)
    except ET.ParseError:
        return nxml_content.decode("utf-8", errors="replace")

    sections = []

    # Extract front matter (title, abstract)
    front = root.find(".//front")
    if front is not None:
        # Title
        title = front.find(".//article-title")
        if title is not None:
            sections.append(f"TITLE: {_get_text(title)}\n")

        # Abstract
        abstract = front.find(".//abstract")
        if abstract is not None:
            sections.append(f"ABSTRACT:\n{_get_text(abstract)}\n")

    # Extract body sections
    body = root.find(".//body")
    if body is not None:
        for sec in body.findall(".//sec"):
            title = sec.find("title")
            if title is not None:
                sections.append(f"\n{_get_text(title).upper()}\n")

            for p in sec.findall("p"):
                sections.append(_get_text(p))
                sections.append("")

        # If no sections, just get all paragraphs
        if not body.findall(".//sec"):
            for p in body.findall(".//p"):
                sections.append(_get_text(p))
                sections.append("")

    # Extract back matter (references summary)
    back = root.find(".//back")
    if back is not None:
        ref_list = back.find(".//ref-list")
        if ref_list is not None:
            ref_count = len(ref_list.findall("ref"))
            sections.append(f"\nREFERENCES: {ref_count} references\n")

    return "\n".join(sections).strip()


def _get_text(element: ET.Element) -> str:
    """Recursively extract text from an XML element."""
    texts = []
    if element.text:
        texts.append(element.text)
    for child in element:
        texts.append(_get_text(child))
        if child.tail:
            texts.append(child.tail)
    return "".join(texts).strip()
