preamble = """
Extract the following information from the text to process below. The text is a scientific article that details experimental procedures and results; we are interested in extracting structured data from the unstructured article. Return the results as a single JSON file, keyed as specified by each following section.
"""

keyword_data_prompt = """
=== KEYWORDS ===
For the key, "keywords", identify keywords that are relevant to the research topic, data acquisition, analysis, and conclusions of article. Return a list of strings.
"""


raw_data_prompt = """
=== RAW DATA ===
For the key, "raw_data", identify the parts of the text related to raw data. Return a list of dictionaries, each with the following keys:
- dataset_id
-- ID for the associated data repository
- data_repository
-- Name of the data repository (e.g. PXD, MassIVE, etc.)
- url (optional)
-- URL to the data repository, if available. For standard data repositories, this can be reconstructed from the ID.
- evidence_text
-- Text from the article that was used to determine the dataset.
"""

processed_data_prompt = """
=== PROCESSED DATA ===
For the key, "processed_data", identify the parts of the text related to processed data, specifically omic quantities such as protein abundance or transcript counts. Return a list of dictionaries, each with the following keys:
- dataset ID
-- ID for the associated data repository
- data repository
-- Name of the data repository (e.g. PXD, MassIVE, etc.)
- url (optional)
-- URL to the data repository, if available.
- file (optional)
-- If the URL points to an archive, specify
- evidence_text
-- Text from the article that was used to determine the dataset.
"""

analysis_methods = """
=== ANALYSIS METHODS ===
For the key, "analysis_methods", identify the parts of the text related to the analysis methods used to generate the processed data. Return a list of dictionaries, each with the following keys:
- method_name
-- Name of the analysis method
- method_description
-- Description of the analysis method
"""

code_prompt = """
=== CODE ===
For the key, "code", identify the parts of the text related to the code used to generate the processed data. Return a list of dictionaries, each with the following keys:
- repository
-- Name of the repository
- url
-- URL to the repository
- description
-- Description of the code's purpose.
"""

postamble = """
BEGIN TEXT TO PROCESS
=====================
"""

supplemental_classification_prompt = """
Classify the following files from a scientific publication into one of five categories:

1. **raw_data**: Raw instrument data from data collection
   - Mass spectrometry: .raw, .mzML, .wiff files
   - Sequencing: .fastq, .bam, .sam files
   - Any unprocessed data directly from instruments

2. **quantitative_data**: Processed omic quantities derived from raw data
   - Protein abundances, gene expression levels, metabolite concentrations
   - Typically in spreadsheets (Excel, CSV) with numerical values
   - Contains processed measurements ready for analysis

3. **summary_data**: Summary tables, figures, and documents presenting findings
   - Supplementary PDFs, Word documents
   - Tables summarizing results
   - Figures and visualizations

4. **supporting**: Files needed for processing but not derived from data collection
   - Reference databases (FASTA files)
   - Configuration files, spectral libraries
   - Search results, annotation files

5. **instrument_parameter**: Instrument acquisition parameters, methods, and settings
   - Mass spectrometer method files (.method, .m)
   - Acquisition parameters and settings
   - Instrument calibration files
   - Tune files, run parameters
   - Any file describing how instruments were configured during data collection

## File Information

**Filenames:** {filename}

**File Content Preview:**
```
{file_preview}
```

## Article Context (for understanding experimental design):
```
{article_context}
```

## Instructions

Based on the filename, content preview, and article context (if any), classify this file.
Return your answer as JSON with two fields:
- "category": one of "raw_data", "quantitative_data", "summary_data", "supporting", or "instrument_parameter"
- "justification": a brief explanation (under 15 words) for your classification

Example response:
{{"category": "quantitative_data", "justification": "Excel file with protein abundance measurements"}}
"""
