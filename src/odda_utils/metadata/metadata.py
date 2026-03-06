# Metadata extraction from PubMed efetch XML responses

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class JournalInfo:
    """Journal metadata."""

    name: str | None = None
    issn: str | None = None
    eissn: str | None = None
    volume: str | None = None
    issue: str | None = None
    iso_abbreviation: str | None = None


@dataclass
class Author:
    """Author information."""

    last_name: str | None = None
    first_name: str | None = None
    initials: str | None = None
    orcid: str | None = None
    affiliations: list[str] = field(default_factory=list)
    is_collective: bool = False


@dataclass
class Grant:
    """Grant/funding information."""

    grant_id: str | None = None
    acronym: str | None = None
    agency: str | None = None
    country: str | None = None


@dataclass
class MeshTerm:
    """MeSH (Medical Subject Headings) term."""

    descriptor_name: str
    descriptor_ui: str | None = None
    is_major_topic: bool = False
    qualifiers: list[str] = field(default_factory=list)


@dataclass
class FullArticleMetadata:
    """Complete article metadata extracted from PubMed XML."""

    # Basic identifiers
    pmid: str
    doi: str | None = None
    pmcid: str | None = None

    # Core content
    title: str | None = None
    abstract: str | None = None

    # Publication info
    journal: JournalInfo | None = None
    publication_date: date | None = None
    electronic_publication_date: date | None = None
    article_type: str | None = None
    language: str | None = None

    # People and organizations
    authors: list[Author] = field(default_factory=list)

    # Subject classification
    keywords: list[str] = field(default_factory=list)
    mesh_terms: list[MeshTerm] = field(default_factory=list)

    # Funding
    grants: list[Grant] = field(default_factory=list)

    # Legal
    copyright: str | None = None


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


def _parse_pubmed_date(date_elem: ET.Element) -> date | None:
    """Parse a PubMed date element into a Python date object.

    Args:
        date_elem: XML element containing Year, Month, Day subelements.

    Returns:
        date object or None if parsing fails.
    """
    year_elem = date_elem.find("Year")
    if year_elem is None or not year_elem.text:
        return None

    try:
        year = int(year_elem.text)
    except ValueError:
        return None

    month = 1
    month_elem = date_elem.find("Month")
    if month_elem is not None and month_elem.text:
        try:
            month = int(month_elem.text)
        except ValueError:
            # Month might be a name like "Jan"
            month_names = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "may": 5, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            month = month_names.get(month_elem.text.lower()[:3], 1)

    day = 1
    day_elem = date_elem.find("Day")
    if day_elem is not None and day_elem.text:
        try:
            day = int(day_elem.text)
        except ValueError:
            pass

    try:
        return date(year, month, day)
    except ValueError:
        # Invalid date, try with day=1
        try:
            return date(year, month, 1)
        except ValueError:
            return date(year, 1, 1)


def extract_journal_info(article: ET.Element) -> JournalInfo | None:
    """Extract journal information from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        JournalInfo or None if no journal information found.
    """
    journal_elem = article.find(".//Journal")
    if journal_elem is None:
        return None

    info = JournalInfo()

    # Journal title
    title_elem = journal_elem.find("Title")
    if title_elem is not None:
        info.name = title_elem.text

    # ISO abbreviation
    iso_elem = journal_elem.find("ISOAbbreviation")
    if iso_elem is not None:
        info.iso_abbreviation = iso_elem.text

    # ISSN (can be print or electronic)
    issn_elem = journal_elem.find("ISSN")
    if issn_elem is not None:
        issn_type = issn_elem.get("IssnType", "")
        if issn_type == "Electronic":
            info.eissn = issn_elem.text
        else:
            info.issn = issn_elem.text

    # Volume and issue from JournalIssue
    journal_issue = journal_elem.find("JournalIssue")
    if journal_issue is not None:
        volume_elem = journal_issue.find("Volume")
        if volume_elem is not None:
            info.volume = volume_elem.text

        issue_elem = journal_issue.find("Issue")
        if issue_elem is not None:
            info.issue = issue_elem.text

    return info


def extract_publication_date(article: ET.Element) -> date | None:
    """Extract print publication date from a PubMed article XML element.

    Tries multiple date sources in order of preference:
    1. PubDate in JournalIssue (print publication date)
    2. PubMedPubDate with PubStatus="pubmed"

    Note: Electronic publication date should be extracted separately using
    extract_electronic_publication_date().

    Args:
        article: PubmedArticle XML element.

    Returns:
        date object or None if no valid date found.
    """
    # Try PubDate in JournalIssue (print publication date)
    pub_date_elem = article.find(".//JournalIssue/PubDate")
    if pub_date_elem is not None:
        # PubDate can have MedlineDate instead of Year/Month/Day
        medline_date = pub_date_elem.find("MedlineDate")
        if medline_date is not None and medline_date.text:
            # MedlineDate format varies, try to extract year
            parts = medline_date.text.split()
            if parts:
                try:
                    year = int(parts[0][:4])
                    return date(year, 1, 1)
                except ValueError:
                    pass
        else:
            pub_date = _parse_pubmed_date(pub_date_elem)
            if pub_date:
                return pub_date

    # Try PubMedPubDate as fallback
    pubmed_pub_date = article.find(".//PubMedPubDate[@PubStatus='pubmed']")
    if pubmed_pub_date is not None:
        return _parse_pubmed_date(pubmed_pub_date)

    return None


def extract_electronic_publication_date(article: ET.Element) -> date | None:
    """Extract electronic publication date from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        date object or None if no electronic publication date found.
    """
    article_date = article.find(".//ArticleDate[@DateType='Electronic']")
    if article_date is not None:
        return _parse_pubmed_date(article_date)
    return None


def extract_authors(article: ET.Element) -> list[Author]:
    """Extract author information from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        List of Author objects.
    """
    authors = []
    author_list = article.find(".//AuthorList")

    if author_list is None:
        return authors

    for author_elem in author_list.findall("Author"):
        author = Author()

        # Check if this is a collective name (organization)
        collective_name = author_elem.find("CollectiveName")
        if collective_name is not None:
            author.last_name = collective_name.text
            author.is_collective = True
        else:
            # Individual author
            last_name_elem = author_elem.find("LastName")
            if last_name_elem is not None:
                author.last_name = last_name_elem.text

            fore_name_elem = author_elem.find("ForeName")
            if fore_name_elem is not None:
                author.first_name = fore_name_elem.text

            initials_elem = author_elem.find("Initials")
            if initials_elem is not None:
                author.initials = initials_elem.text

        # ORCID - check Identifier elements
        for identifier in author_elem.findall("Identifier"):
            source = identifier.get("Source", "")
            if source == "ORCID" and identifier.text:
                # ORCID may be full URL or just the ID
                orcid = identifier.text
                if orcid.startswith("http"):
                    orcid = orcid.split("/")[-1]
                author.orcid = orcid
                break

        # Affiliations
        for affiliation in author_elem.findall("AffiliationInfo/Affiliation"):
            if affiliation.text:
                author.affiliations.append(affiliation.text)

        authors.append(author)

    return authors


def extract_keywords(article: ET.Element) -> list[str]:
    """Extract keywords from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        List of keyword strings.
    """
    keywords = []

    for keyword_list in article.findall(".//KeywordList"):
        for keyword in keyword_list.findall("Keyword"):
            if keyword.text:
                keywords.append(keyword.text)

    return keywords


def extract_mesh_terms(article: ET.Element) -> list[MeshTerm]:
    """Extract MeSH terms from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        List of MeshTerm objects.
    """
    mesh_terms = []

    mesh_heading_list = article.find(".//MeshHeadingList")
    if mesh_heading_list is None:
        return mesh_terms

    for mesh_heading in mesh_heading_list.findall("MeshHeading"):
        descriptor = mesh_heading.find("DescriptorName")
        if descriptor is None:
            continue

        term = MeshTerm(
            descriptor_name=descriptor.text or "",
            descriptor_ui=descriptor.get("UI"),
            is_major_topic=descriptor.get("MajorTopicYN") == "Y",
        )

        # Extract qualifiers
        for qualifier in mesh_heading.findall("QualifierName"):
            if qualifier.text:
                term.qualifiers.append(qualifier.text)

        mesh_terms.append(term)

    return mesh_terms


def extract_article_type(article: ET.Element) -> str | None:
    """Extract article type from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        Article type string or None.
    """
    # Try PublicationTypeList first
    pub_type_list = article.find(".//PublicationTypeList")
    if pub_type_list is not None:
        pub_types = []
        for pub_type in pub_type_list.findall("PublicationType"):
            if pub_type.text:
                pub_types.append(pub_type.text)
        if pub_types:
            # Return primary type (usually first) or join all
            return pub_types[0]

    return None


def extract_grants(article: ET.Element) -> list[Grant]:
    """Extract grant/funding information from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        List of Grant objects.
    """
    grants = []

    grant_list = article.find(".//GrantList")
    if grant_list is None:
        return grants

    for grant_elem in grant_list.findall("Grant"):
        grant = Grant()

        grant_id_elem = grant_elem.find("GrantID")
        if grant_id_elem is not None:
            grant.grant_id = grant_id_elem.text

        acronym_elem = grant_elem.find("Acronym")
        if acronym_elem is not None:
            grant.acronym = acronym_elem.text

        agency_elem = grant_elem.find("Agency")
        if agency_elem is not None:
            grant.agency = agency_elem.text

        country_elem = grant_elem.find("Country")
        if country_elem is not None:
            grant.country = country_elem.text

        grants.append(grant)

    return grants


def extract_language(article: ET.Element) -> str | None:
    """Extract language from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        Language code (e.g., "eng") or None.
    """
    language_elem = article.find(".//Language")
    if language_elem is not None:
        return language_elem.text
    return None


def extract_copyright(article: ET.Element) -> str | None:
    """Extract copyright information from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        Copyright string or None.
    """
    # Check Abstract for CopyrightInformation
    copyright_elem = article.find(".//Abstract/CopyrightInformation")
    if copyright_elem is not None and copyright_elem.text:
        return copyright_elem.text

    # Check CoiStatement (Conflict of Interest, sometimes has copyright)
    coi_elem = article.find(".//CoiStatement")
    if coi_elem is not None and coi_elem.text:
        text = coi_elem.text.lower()
        if "copyright" in text or "©" in text:
            return coi_elem.text

    return None


def extract_identifiers(article: ET.Element) -> tuple[str | None, str | None]:
    """Extract DOI and PMCID from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        Tuple of (doi, pmcid).
    """
    doi = None
    pmcid = None

    # Use specific path to PubmedData/ArticleIdList to avoid picking up
    # ArticleIds from referenced papers in the ReferenceList section
    for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.get("IdType")
        if id_type == "doi":
            doi = article_id.text
        elif id_type == "pmc":
            pmcid = article_id.text

    return doi, pmcid


def extract_title(article: ET.Element) -> str | None:
    """Extract article title from a PubMed article XML element.

    Args:
        article: PubmedArticle XML element.

    Returns:
        Title string or None.
    """
    title_elem = article.find(".//ArticleTitle")
    if title_elem is not None:
        return _get_text(title_elem)
    return None


def extract_abstract(article: ET.Element) -> str | None:
    """Extract abstract from a PubMed article XML element.

    Handles structured abstracts with labeled sections.

    Args:
        article: PubmedArticle XML element.

    Returns:
        Abstract text or None.
    """
    abstract_elem = article.find(".//Abstract")
    if abstract_elem is None:
        return None

    abstract_texts = []
    for abstract_text in abstract_elem.findall(".//AbstractText"):
        label = abstract_text.get("Label")
        text = _get_text(abstract_text)
        if label:
            abstract_texts.append(f"{label}: {text}")
        else:
            abstract_texts.append(text)

    return " ".join(abstract_texts) if abstract_texts else None


def extract_full_metadata(xml_content: bytes | str, pmid: str) -> FullArticleMetadata:
    """Extract all available metadata from PubMed efetch XML response.

    Args:
        xml_content: Raw XML content from PubMed efetch API.
        pmid: PubMed ID of the article.

    Returns:
        FullArticleMetadata with all extracted fields.
    """
    if isinstance(xml_content, str):
        xml_content = xml_content.encode("utf-8")

    root = ET.fromstring(xml_content)
    article = root.find(".//PubmedArticle")

    if article is None:
        return FullArticleMetadata(pmid=pmid)

    doi, pmcid = extract_identifiers(article)

    return FullArticleMetadata(
        pmid=pmid,
        doi=doi,
        pmcid=pmcid,
        title=extract_title(article),
        abstract=extract_abstract(article),
        journal=extract_journal_info(article),
        publication_date=extract_publication_date(article),
        electronic_publication_date=extract_electronic_publication_date(article),
        article_type=extract_article_type(article),
        language=extract_language(article),
        authors=extract_authors(article),
        keywords=extract_keywords(article),
        mesh_terms=extract_mesh_terms(article),
        grants=extract_grants(article),
        copyright=extract_copyright(article),
    )


def extract_full_metadata_from_element(
    article: ET.Element, pmid: str
) -> FullArticleMetadata:
    """Extract all available metadata from a PubmedArticle XML element.

    Args:
        article: PubmedArticle XML element.
        pmid: PubMed ID of the article.

    Returns:
        FullArticleMetadata with all extracted fields.
    """
    doi, pmcid = extract_identifiers(article)

    return FullArticleMetadata(
        pmid=pmid,
        doi=doi,
        pmcid=pmcid,
        title=extract_title(article),
        abstract=extract_abstract(article),
        journal=extract_journal_info(article),
        publication_date=extract_publication_date(article),
        electronic_publication_date=extract_electronic_publication_date(article),
        article_type=extract_article_type(article),
        language=extract_language(article),
        authors=extract_authors(article),
        keywords=extract_keywords(article),
        mesh_terms=extract_mesh_terms(article),
        grants=extract_grants(article),
        copyright=extract_copyright(article),
    )
