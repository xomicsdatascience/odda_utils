# Validate article metadata consistency across identifiers (DOI, PMID, PMCID).

import asyncio
import httpx
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional
from datetime import date
import xml.etree.ElementTree as ET


class RateLimiter:
    """Async rate limiter using token bucket algorithm.

    Parameters
    ----------
    requests_per_second : float
        Maximum number of requests allowed per second.
    """

    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_request_time = 0.0

    async def acquire(self) -> None:
        """Wait until a request can be made within rate limits."""
        async with self._lock:
            now = time.monotonic()
            time_since_last = now - self._last_request_time
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self._last_request_time = time.monotonic()


@dataclass
class ArticleMetadata:
    """Metadata retrieved from an external source for an article."""
    title: Optional[str] = None
    publication_date: Optional[date] = None  # Print publication date
    electronic_publication_date: Optional[date] = None  # Electronic/online publication date
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    source: Optional[str] = None  # e.g., "crossref", "pubmed"
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating article metadata consistency."""
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    stored_title: Optional[str] = None
    stored_publication_date: Optional[date] = None  # Print publication date
    stored_electronic_publication_date: Optional[date] = None  # Electronic publication date

    # Fetched metadata from each source
    crossref_metadata: Optional[ArticleMetadata] = None
    pubmed_metadata: Optional[ArticleMetadata] = None

    # Consistency flags
    title_matches_crossref: Optional[bool] = None
    title_matches_pubmed: Optional[bool] = None
    date_matches_crossref: Optional[bool] = None
    date_matches_pubmed: Optional[bool] = None
    electronic_date_matches_crossref: Optional[bool] = None
    electronic_date_matches_pubmed: Optional[bool] = None
    ids_consistent: Optional[bool] = None

    # Detailed issues found
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if no issues were found."""
        return len(self.issues) == 0


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Parameters
    ----------
    text : str
        Text potentially containing HTML tags.

    Returns
    -------
    str
        Text with HTML tags removed.
    """
    # Remove HTML tags like <sup>, </sup>, <sub>, </sub>, <i>, </i>, etc.
    return re.sub(r"<[^>]+>", "", text)


def _normalize_title(title: Optional[str]) -> str:
    """Normalize a title for comparison.

    Applies the following transformations:
    - Strip HTML tags
    - Lowercase
    - Remove accents and diacritics (é -> e, ü -> u, etc.)
    - Remove punctuation and dashes (including en-dash, em-dash, hyphens)
    - Normalize whitespace

    Parameters
    ----------
    title : str, optional
        The title to normalize.

    Returns
    -------
    str
        Normalized title for comparison.
    """
    if not title:
        return ""

    # Strip HTML tags first
    normalized = _strip_html_tags(title)

    # Normalize Unicode to decomposed form (separates base chars from accents)
    normalized = unicodedata.normalize("NFKD", normalized)

    # Remove accents/diacritics (combining characters)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))

    # Lowercase
    normalized = normalized.lower()

    # Remove punctuation, dashes (including en-dash, em-dash, hyphens), and special chars
    # Keep only alphanumeric and whitespace
    normalized = re.sub(r"[^\w\s]", " ", normalized)

    # Normalize whitespace
    return " ".join(normalized.split())


def _titles_similar(title1: Optional[str], title2: Optional[str], threshold: float = 0.85) -> bool:
    """Check if two titles are similar enough to be considered matching.

    Uses word overlap ratio for comparison, with special handling for
    truncated titles (e.g., from PubMed API).

    Parameters
    ----------
    title1 : str, optional
        First title to compare.
    title2 : str, optional
        Second title to compare.
    threshold : float
        Minimum Jaccard similarity for titles to match.

    Returns
    -------
    bool
        True if titles are similar enough to be considered matching.
    """
    if not title1 or not title2:
        return False

    norm1 = _normalize_title(title1)
    norm2 = _normalize_title(title2)

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Check for substring/prefix match (handles truncated titles)
    # If the shorter title is a prefix of the longer one, consider it a match
    shorter, longer = (norm1, norm2) if len(norm1) <= len(norm2) else (norm2, norm1)
    if longer.startswith(shorter) and len(shorter) >= 20:
        # Require at least 20 chars to avoid false positives on very short matches
        return True

    # Word overlap ratio (Jaccard similarity)
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if not words1 or not words2:
        return False

    intersection = words1 & words2
    union = words1 | words2
    jaccard = len(intersection) / len(union)

    return jaccard >= threshold


def _parse_crossref_date(date_parts: list) -> Optional[date]:
    """Parse CrossRef date-parts format into a date object."""
    if not date_parts or not date_parts[0]:
        return None
    parts = date_parts[0]
    year = parts[0] if len(parts) > 0 else None
    month = parts[1] if len(parts) > 1 else 1
    day = parts[2] if len(parts) > 2 else 1
    if year:
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


async def fetch_crossref_metadata(
    doi: str,
    timeout: float = 10.0,
    rate_limiter: Optional[RateLimiter] = None
) -> ArticleMetadata:
    """Fetch article metadata from CrossRef API using DOI.

    Parameters
    ----------
    doi : str
        The DOI to look up (without https://doi.org/ prefix).
    timeout : float
        Request timeout in seconds.
    rate_limiter : RateLimiter, optional
        Rate limiter to control request frequency.

    Returns
    -------
    ArticleMetadata
        Fetched information or error.
    """
    if rate_limiter:
        await rate_limiter.acquire()

    url = f"https://api.crossref.org/works/{doi}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            return ArticleMetadata(source="crossref", error=f"HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            return ArticleMetadata(source="crossref", error=str(e))
        except Exception as e:
            return ArticleMetadata(source="crossref", error=str(e))

    message = data.get("message", {})

    # Extract title
    titles = message.get("title", [])
    title = titles[0] if titles else None

    # Extract print publication date
    print_pub_date = None
    if "published-print" in message:
        date_parts = message["published-print"].get("date-parts")
        print_pub_date = _parse_crossref_date(date_parts)

    # Extract electronic publication date
    electronic_pub_date = None
    if "published-online" in message:
        date_parts = message["published-online"].get("date-parts")
        electronic_pub_date = _parse_crossref_date(date_parts)

    # Fall back to "issued" if neither is available
    if not print_pub_date and not electronic_pub_date and "issued" in message:
        date_parts = message["issued"].get("date-parts")
        # "issued" is ambiguous, treat as print date
        print_pub_date = _parse_crossref_date(date_parts)

    return ArticleMetadata(
        title=title,
        publication_date=print_pub_date,
        electronic_publication_date=electronic_pub_date,
        doi=doi,
        source="crossref"
    )


async def fetch_pubmed_metadata(
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    timeout: float = 10.0,
    rate_limiter: Optional[RateLimiter] = None
) -> ArticleMetadata:
    """Fetch article metadata from PubMed/NCBI API using PMID or PMCID.

    Parameters
    ----------
    pmid : str, optional
        PubMed ID to look up.
    pmcid : str, optional
        PubMed Central ID to look up.
    timeout : float
        Request timeout in seconds.
    rate_limiter : RateLimiter, optional
        Rate limiter to control request frequency.

    Returns
    -------
    ArticleMetadata
        Fetched information or error.
    """
    if not pmid and not pmcid:
        return ArticleMetadata(source="pubmed", error="No PMID or PMCID provided")

    # Use efetch API to get article metadata
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "retmode": "xml",
    }

    if pmid:
        params["id"] = pmid
    elif pmcid:
        # First convert PMCID to PMID using ID converter
        if rate_limiter:
            await rate_limiter.acquire()
        converter_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    converter_url,
                    params={"ids": pmcid, "format": "json"},
                    timeout=timeout
                )
                resp.raise_for_status()
                conv_data = resp.json()
                records = conv_data.get("records", [])
                if records and "pmid" in records[0]:
                    params["id"] = records[0]["pmid"]
                else:
                    return ArticleMetadata(source="pubmed", error=f"Could not convert {pmcid} to PMID")
            except Exception as e:
                return ArticleMetadata(source="pubmed", error=f"PMCID conversion failed: {e}")

    if rate_limiter:
        await rate_limiter.acquire()

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(base_url, params=params, timeout=timeout)
            response.raise_for_status()
            xml_content = response.text
        except httpx.HTTPStatusError as e:
            return ArticleMetadata(source="pubmed", error=f"HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            return ArticleMetadata(source="pubmed", error=str(e))

    # Parse XML response
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return ArticleMetadata(source="pubmed", error=f"XML parse error: {e}")

    article = root.find(".//PubmedArticle")
    if article is None:
        return ArticleMetadata(source="pubmed", error="No article found in response")

    # Extract title
    title_elem = article.find(".//ArticleTitle")
    title = title_elem.text if title_elem is not None else None

    # Helper to parse PubMed date elements
    def _parse_pubmed_date_elem(date_elem) -> Optional[date]:
        if date_elem is None:
            return None
        year = date_elem.findtext("Year")
        month_str = date_elem.findtext("Month")
        day = date_elem.findtext("Day")
        if not year:
            return None
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        try:
            month = int(month_str) if month_str and month_str.isdigit() else month_map.get(month_str.lower()[:3], 1) if month_str else 1
            day_int = int(day) if day else 1
            return date(int(year), month, day_int)
        except (ValueError, AttributeError):
            return None

    # Extract print publication date (PubDate in JournalIssue)
    print_pub_date = _parse_pubmed_date_elem(article.find(".//JournalIssue/PubDate"))

    # Extract electronic publication date (ArticleDate with DateType='Electronic')
    electronic_pub_date = _parse_pubmed_date_elem(
        article.find(".//ArticleDate[@DateType='Electronic']")
    )

    # Extract IDs
    extracted_pmid = None
    extracted_pmcid = None
    extracted_doi = None

    pmid_elem = article.find(".//PMID")
    if pmid_elem is not None:
        extracted_pmid = pmid_elem.text

    # Look for DOI and PMCID in ArticleIdList
    for article_id in article.findall(".//ArticleId"):
        id_type = article_id.get("IdType")
        if id_type == "doi":
            extracted_doi = article_id.text
        elif id_type == "pmc":
            extracted_pmcid = article_id.text

    return ArticleMetadata(
        title=title,
        publication_date=print_pub_date,
        electronic_publication_date=electronic_pub_date,
        doi=extracted_doi,
        pmid=extracted_pmid,
        pmcid=extracted_pmcid,
        source="pubmed"
    )


async def validate_article(
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
    stored_title: Optional[str] = None,
    stored_publication_date: Optional[date] = None,
    stored_electronic_publication_date: Optional[date] = None,
    title_similarity_threshold: float = 0.85,
    rate_limiter: Optional[RateLimiter] = None
) -> ValidationResult:
    """Validate article metadata consistency across identifiers.

    Checks:
    1. If DOI is provided, fetches CrossRef metadata and compares title/date.
    2. If PMID/PMCID is provided, fetches PubMed metadata and compares title/date.
    3. If multiple IDs are provided, checks that they all point to the same article.

    Date comparison is done type-to-type: electronic dates are compared to
    electronic dates, and print dates are compared to print dates.

    Parameters
    ----------
    doi : str, optional
        DOI of the article.
    pmid : str, optional
        PubMed ID of the article.
    pmcid : str, optional
        PubMed Central ID of the article.
    stored_title : str, optional
        Title stored in the database.
    stored_publication_date : date, optional
        Print publication date stored in the database.
    stored_electronic_publication_date : date, optional
        Electronic publication date stored in the database.
    title_similarity_threshold : float
        Minimum Jaccard similarity for titles to match.
    rate_limiter : RateLimiter, optional
        Rate limiter to control API request frequency.

    Returns
    -------
    ValidationResult
        Detailed consistency information.
    """
    result = ValidationResult(
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        stored_title=stored_title,
        stored_publication_date=stored_publication_date,
        stored_electronic_publication_date=stored_electronic_publication_date
    )

    # Fetch metadata from CrossRef if DOI provided
    if doi:
        result.crossref_metadata = await fetch_crossref_metadata(doi, rate_limiter=rate_limiter)

        if result.crossref_metadata.error:
            result.issues.append(f"CrossRef lookup failed: {result.crossref_metadata.error}")
        else:
            # Check title match
            if stored_title and result.crossref_metadata.title:
                result.title_matches_crossref = _titles_similar(
                    stored_title,
                    result.crossref_metadata.title,
                    title_similarity_threshold
                )
                if not result.title_matches_crossref:
                    result.issues.append(
                        f"Title mismatch with CrossRef: stored='{stored_title[:50]}...' vs "
                        f"crossref='{result.crossref_metadata.title[:50]}...'"
                    )

            # Check date match (compare like-with-like: print to print, electronic to electronic)
            # Print publication date comparison
            if stored_publication_date and result.crossref_metadata.publication_date:
                result.date_matches_crossref = (
                    stored_publication_date == result.crossref_metadata.publication_date
                )
                if not result.date_matches_crossref:
                    result.issues.append(
                        f"Print date mismatch with CrossRef: stored={stored_publication_date} vs "
                        f"crossref={result.crossref_metadata.publication_date}"
                    )

            # Electronic publication date comparison
            if stored_electronic_publication_date and result.crossref_metadata.electronic_publication_date:
                result.electronic_date_matches_crossref = (
                    stored_electronic_publication_date == result.crossref_metadata.electronic_publication_date
                )
                if not result.electronic_date_matches_crossref:
                    result.issues.append(
                        f"Electronic date mismatch with CrossRef: stored={stored_electronic_publication_date} vs "
                        f"crossref={result.crossref_metadata.electronic_publication_date}"
                    )

    # Fetch metadata from PubMed if PMID or PMCID provided
    if pmid or pmcid:
        result.pubmed_metadata = await fetch_pubmed_metadata(pmid=pmid, pmcid=pmcid, rate_limiter=rate_limiter)

        if result.pubmed_metadata.error:
            result.issues.append(f"PubMed lookup failed: {result.pubmed_metadata.error}")
        else:
            # Check title match
            if stored_title and result.pubmed_metadata.title:
                result.title_matches_pubmed = _titles_similar(
                    stored_title,
                    result.pubmed_metadata.title,
                    title_similarity_threshold
                )
                if not result.title_matches_pubmed:
                    result.issues.append(
                        f"Title mismatch with PubMed: stored='{stored_title[:50]}...' vs "
                        f"pubmed='{result.pubmed_metadata.title[:50]}...'"
                    )

            # Check date match (compare like-with-like: print to print, electronic to electronic)
            # Print publication date comparison
            if stored_publication_date and result.pubmed_metadata.publication_date:
                result.date_matches_pubmed = (
                    stored_publication_date == result.pubmed_metadata.publication_date
                )
                if not result.date_matches_pubmed:
                    result.issues.append(
                        f"Print date mismatch with PubMed: stored={stored_publication_date} vs "
                        f"pubmed={result.pubmed_metadata.publication_date}"
                    )

            # Electronic publication date comparison
            if stored_electronic_publication_date and result.pubmed_metadata.electronic_publication_date:
                result.electronic_date_matches_pubmed = (
                    stored_electronic_publication_date == result.pubmed_metadata.electronic_publication_date
                )
                if not result.electronic_date_matches_pubmed:
                    result.issues.append(
                        f"Electronic date mismatch with PubMed: stored={stored_electronic_publication_date} vs "
                        f"pubmed={result.pubmed_metadata.electronic_publication_date}"
                    )

    # Check ID consistency across sources
    if result.crossref_metadata and result.pubmed_metadata:
        if not result.crossref_metadata.error and not result.pubmed_metadata.error:
            # Check if titles from both sources match each other
            crossref_title = result.crossref_metadata.title
            pubmed_title = result.pubmed_metadata.title

            if crossref_title and pubmed_title:
                titles_match = _titles_similar(crossref_title, pubmed_title, title_similarity_threshold)
                if not titles_match:
                    result.ids_consistent = False
                    result.issues.append(
                        f"IDs point to different articles: CrossRef title='{crossref_title[:50]}...' vs "
                        f"PubMed title='{pubmed_title[:50]}...'"
                    )
                else:
                    result.ids_consistent = True

            # Check if DOI from PubMed matches provided DOI
            if doi and result.pubmed_metadata.doi:
                if doi.lower() != result.pubmed_metadata.doi.lower():
                    result.ids_consistent = False
                    result.issues.append(
                        f"DOI mismatch: provided={doi} vs PubMed extracted={result.pubmed_metadata.doi}"
                    )

    return result


async def validate_article_batch(
    articles: list[dict],
    title_similarity_threshold: float = 0.85,
    requests_per_second: float = 1.0
) -> list[ValidationResult]:
    """Validate a batch of articles with rate limiting.

    Parameters
    ----------
    articles : list[dict]
        List of dicts with keys: doi, pmid, pmcid, title, publication_date,
        electronic_publication_date.
    title_similarity_threshold : float
        Minimum similarity for titles to match.
    requests_per_second : float
        Maximum API requests per second (default 1.0).

    Returns
    -------
    list[ValidationResult]
        Validation results for each article.
    """
    rate_limiter = RateLimiter(requests_per_second=requests_per_second)

    tasks = [
        validate_article(
            doi=a.get("doi"),
            pmid=a.get("pmid"),
            pmcid=a.get("pmcid"),
            stored_title=a.get("title"),
            stored_publication_date=a.get("publication_date"),
            stored_electronic_publication_date=a.get("electronic_publication_date"),
            title_similarity_threshold=title_similarity_threshold,
            rate_limiter=rate_limiter
        )
        for a in articles
    ]

    return await asyncio.gather(*tasks)
