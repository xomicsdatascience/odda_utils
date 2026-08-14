# Unit tests for the JATS/NXML text extraction in odda_utils.fetching.pmc and
# for identifier normalisation in odda_utils.utils. These cover a class of
# silent data loss: publishers place the data availability statement in three
# different locations, and an extractor that reads only <body> drops the
# statement (and every accession that appears only there) without erroring.
# The fixtures below reproduce all three real layouts observed in PubMed
# Central. No network or credentials are required.

import unittest

from odda_utils.fetching.pmc import _extract_text_from_nxml
from odda_utils.utils import _as_str


def _article(body: str = "", back: str = "") -> bytes:
    """Build a minimal JATS article around the supplied body and back matter."""
    return (
        "<article>"
        "<front><article-meta>"
        "<article-title>A study</article-title>"
        "<abstract><p>An abstract.</p></abstract>"
        "</article-meta></front>"
        f"<body><sec><title>Methods</title><p>We did things.</p></sec>{body}</body>"
        f"<back>{back}</back>"
        "</article>"
    ).encode("utf-8")


# Layout 1: ACS, Elsevier and Oxford place the statement inside <body>.
BODY_SEC = _article(
    body=(
        '<sec sec-type="data-availability">'
        "<title>Data availability</title>"
        "<p>Deposited to PRIDE with the identifier PXD069212.</p>"
        "</sec>"
    )
)

# Layout 2: ASM places the statement in <back> as a <sec>.
BACK_SEC = _article(
    back=(
        '<sec sec-type="data-availability">'
        "<title>Data availability</title>"
        "<p>Available through PeptideAtlas data set PASS05975.</p>"
        "</sec>"
        "<ref-list><ref/><ref/></ref-list>"
    )
)

# Layout 3: Springer, Nature and BMC use <back> with <notes>.
BACK_NOTES = _article(
    back=(
        '<notes notes-type="data-availability">'
        "<title>Data availability</title>"
        "<p>Deposited with the identifier PXD059818 and GSE287770.</p>"
        "</notes>"
        '<notes notes-type="data-availability">'
        "<title>Code availability</title>"
        "<p>Code is at github.com/example/repo.</p>"
        "</notes>"
        "<ref-list><ref/></ref-list>"
    )
)

# Springer nests declarations inside a parent note, which must not cause the
# nested paragraphs to be emitted twice.
NESTED_NOTES = _article(
    back=(
        "<notes>"
        "<title>Declarations</title>"
        '<notes notes-type="COI-statement">'
        "<title>Competing interests</title>"
        "<p>The authors declare no competing interests.</p>"
        "</notes>"
        "</notes>"
    )
)

# Some publishers omit the title and carry the meaning in the attribute only.
UNTITLED_NOTES = _article(
    back='<notes notes-type="funding-statement"><p>Funded by NIGMS.</p></notes>'
)


class TestDataAvailabilityExtraction(unittest.TestCase):
    """The data availability statement must survive from all three layouts."""

    def test_statement_in_body_section(self):
        text = _extract_text_from_nxml(BODY_SEC)
        self.assertIn("PXD069212", text)
        self.assertIn("DATA AVAILABILITY", text.upper())

    def test_statement_in_back_section(self):
        text = _extract_text_from_nxml(BACK_SEC)
        self.assertIn("PASS05975", text)
        self.assertIn("DATA AVAILABILITY", text.upper())

    def test_statement_in_back_notes(self):
        text = _extract_text_from_nxml(BACK_NOTES)
        self.assertIn("PXD059818", text)
        self.assertIn("GSE287770", text)
        self.assertIn("DATA AVAILABILITY", text.upper())

    def test_sibling_code_availability_note_also_kept(self):
        # Two notes share notes-type="data-availability"; taking only the first
        # would silently drop the code availability statement.
        text = _extract_text_from_nxml(BACK_NOTES)
        self.assertIn("CODE AVAILABILITY", text.upper())
        self.assertIn("github.com/example/repo", text)


class TestBackMatterFormatting(unittest.TestCase):
    """Back matter must be emitted once, titled, without disturbing the rest."""

    def test_nested_notes_are_not_duplicated(self):
        text = _extract_text_from_nxml(NESTED_NOTES)
        self.assertEqual(text.count("The authors declare no competing interests."), 1)
        self.assertIn("COMPETING INTERESTS", text.upper())

    def test_heading_falls_back_to_type_attribute(self):
        text = _extract_text_from_nxml(UNTITLED_NOTES)
        self.assertIn("FUNDING STATEMENT", text)
        self.assertIn("Funded by NIGMS.", text)

    def test_reference_count_still_reported(self):
        self.assertIn("REFERENCES: 2 references", _extract_text_from_nxml(BACK_SEC))
        self.assertIn("REFERENCES: 1 references", _extract_text_from_nxml(BACK_NOTES))

    def test_front_and_body_still_extracted(self):
        text = _extract_text_from_nxml(BACK_NOTES)
        self.assertIn("A study", text)
        self.assertIn("An abstract.", text)
        self.assertIn("We did things.", text)

    def test_article_without_back_matter_is_unaffected(self):
        text = _extract_text_from_nxml(_article())
        self.assertIn("We did things.", text)

    def test_malformed_xml_falls_back_to_raw_decode(self):
        text = _extract_text_from_nxml(b"<article><body>unclosed")
        self.assertIn("unclosed", text)


class TestIdentifierNormalisation(unittest.TestCase):
    """The current NCBI converter returns PMIDs as JSON numbers, not strings."""

    def test_integer_pmid_is_coerced_to_string(self):
        self.assertEqual(_as_str(42117716), "42117716")

    def test_string_identifier_is_unchanged(self):
        self.assertEqual(_as_str("PMC13228066"), "PMC13228066")

    def test_missing_identifier_stays_none(self):
        self.assertIsNone(_as_str(None))


if __name__ == "__main__":
    unittest.main()
