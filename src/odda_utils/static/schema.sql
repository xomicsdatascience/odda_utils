-- Journals table
CREATE TABLE IF NOT EXISTS journals (
    journal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    issn VARCHAR(20),
    eissn VARCHAR(20),
    iso_abbreviation VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_journals_name ON journals(name);
CREATE INDEX IF NOT EXISTS idx_journals_issn ON journals(issn);
CREATE INDEX IF NOT EXISTS idx_journals_iso_abbrev ON journals(iso_abbreviation);

-- Articles table
CREATE TABLE IF NOT EXISTS articles (
    doi VARCHAR(40) UNIQUE,
    pmid VARCHAR(30) UNIQUE,
    pmcid VARCHAR(30) UNIQUE,
    title TEXT,
    abstract TEXT,
    publication_date DATE,
    electronic_publication_date DATE,
    journal_id INTEGER REFERENCES journals(journal_id),
    volume VARCHAR(20),
    issue VARCHAR(20),
    article_type VARCHAR(100),
    language VARCHAR(10),
    copyright TEXT,
    article_filepath TEXT,
    supplementals_filepath TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_articles_pmid ON articles(pmid);
CREATE INDEX IF NOT EXISTS idx_articles_pmcid ON articles(pmcid);
CREATE INDEX IF NOT EXISTS idx_articles_journal ON articles(journal_id);
CREATE INDEX IF NOT EXISTS idx_articles_publication_date ON articles(publication_date);
CREATE INDEX IF NOT EXISTS idx_articles_article_type ON articles(article_type);
CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language);

-- Author info table
CREATE TABLE IF NOT EXISTS author_info (
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    initials VARCHAR(10),
    author_orcid VARCHAR(50) UNIQUE,
    is_collective BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_author_info_orcid ON author_info(author_orcid);
CREATE INDEX IF NOT EXISTS idx_author_info_last_name ON author_info(last_name);
CREATE INDEX IF NOT EXISTS idx_author_info_collective ON author_info(is_collective);

-- Affiliations table
CREATE TABLE IF NOT EXISTS affiliations (
    affiliation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    affiliation_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_affiliations_text ON affiliations(affiliation_text);

-- Author-affiliation linking table
CREATE TABLE IF NOT EXISTS author_affiliations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id INTEGER REFERENCES author_info(author_id),
    affiliation_id INTEGER REFERENCES affiliations(affiliation_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(author_id, affiliation_id)
);

CREATE INDEX IF NOT EXISTS idx_author_affiliations_author ON author_affiliations(author_id);
CREATE INDEX IF NOT EXISTS idx_author_affiliations_affiliation ON author_affiliations(affiliation_id);

-- Article-author linking table
CREATE TABLE IF NOT EXISTS article_authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    author_id INTEGER REFERENCES author_info(author_id),
    author_position INTEGER,
    is_corresponding BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, author_id),
    UNIQUE(pmid, author_id),
    UNIQUE(pmcid, author_id)
);

CREATE INDEX IF NOT EXISTS idx_article_authors_doi ON article_authors(doi);
CREATE INDEX IF NOT EXISTS idx_article_authors_pmid ON article_authors(pmid);
CREATE INDEX IF NOT EXISTS idx_article_authors_pmcid ON article_authors(pmcid);
CREATE INDEX IF NOT EXISTS idx_article_authors_author_id ON article_authors(author_id);

-- Keywords table
CREATE TABLE IF NOT EXISTS keywords (
    keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON keywords(keyword);

-- Article-keyword linking table
CREATE TABLE IF NOT EXISTS article_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    keyword_id INTEGER REFERENCES keywords(keyword_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, keyword_id),
    UNIQUE(pmid, keyword_id),
    UNIQUE(pmcid, keyword_id)
);

CREATE INDEX IF NOT EXISTS idx_article_keywords_doi ON article_keywords(doi);
CREATE INDEX IF NOT EXISTS idx_article_keywords_pmid ON article_keywords(pmid);
CREATE INDEX IF NOT EXISTS idx_article_keywords_pmcid ON article_keywords(pmcid);
CREATE INDEX IF NOT EXISTS idx_article_keywords_keyword_id ON article_keywords(keyword_id);

-- LLM-generated keywords table
CREATE TABLE IF NOT EXISTS llm_keywords (
    llm_keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(keyword, model)
);

CREATE INDEX IF NOT EXISTS idx_llm_keywords_keyword ON llm_keywords(keyword);
CREATE INDEX IF NOT EXISTS idx_llm_keywords_model ON llm_keywords(model);

-- Article-LLM keyword linking table
CREATE TABLE IF NOT EXISTS article_llm_keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    llm_keyword_id INTEGER REFERENCES llm_keywords(llm_keyword_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, llm_keyword_id),
    UNIQUE(pmid, llm_keyword_id),
    UNIQUE(pmcid, llm_keyword_id)
);

CREATE INDEX IF NOT EXISTS idx_article_llm_keywords_doi ON article_llm_keywords(doi);
CREATE INDEX IF NOT EXISTS idx_article_llm_keywords_pmid ON article_llm_keywords(pmid);
CREATE INDEX IF NOT EXISTS idx_article_llm_keywords_pmcid ON article_llm_keywords(pmcid);
CREATE INDEX IF NOT EXISTS idx_article_llm_keywords_keyword_id ON article_llm_keywords(llm_keyword_id);

-- Grants table
CREATE TABLE IF NOT EXISTS grants (
    grant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    grant_number VARCHAR(100),
    acronym VARCHAR(20),
    agency TEXT,
    country VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_grants_grant_number ON grants(grant_number);
CREATE INDEX IF NOT EXISTS idx_grants_agency ON grants(agency);
CREATE INDEX IF NOT EXISTS idx_grants_country ON grants(country);

-- Article-grant linking table
CREATE TABLE IF NOT EXISTS article_grants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    grant_id INTEGER REFERENCES grants(grant_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, grant_id),
    UNIQUE(pmid, grant_id),
    UNIQUE(pmcid, grant_id)
);

CREATE INDEX IF NOT EXISTS idx_article_grants_doi ON article_grants(doi);
CREATE INDEX IF NOT EXISTS idx_article_grants_pmid ON article_grants(pmid);
CREATE INDEX IF NOT EXISTS idx_article_grants_pmcid ON article_grants(pmcid);
CREATE INDEX IF NOT EXISTS idx_article_grants_grant_id ON article_grants(grant_id);

-- MeSH terms table
CREATE TABLE IF NOT EXISTS mesh_terms (
    mesh_term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor_name TEXT NOT NULL,
    descriptor_ui VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mesh_terms_ui ON mesh_terms(descriptor_ui);
CREATE INDEX IF NOT EXISTS idx_mesh_terms_name ON mesh_terms(descriptor_name);

-- MeSH qualifiers table
CREATE TABLE IF NOT EXISTS mesh_qualifiers (
    qualifier_id INTEGER PRIMARY KEY AUTOINCREMENT,
    qualifier_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mesh_qualifiers_name ON mesh_qualifiers(qualifier_name);

-- Article-MeSH term linking table
CREATE TABLE IF NOT EXISTS article_mesh_terms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    mesh_term_id INTEGER REFERENCES mesh_terms(mesh_term_id),
    is_major_topic BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, mesh_term_id),
    UNIQUE(pmid, mesh_term_id),
    UNIQUE(pmcid, mesh_term_id)
);

CREATE INDEX IF NOT EXISTS idx_article_mesh_terms_doi ON article_mesh_terms(doi);
CREATE INDEX IF NOT EXISTS idx_article_mesh_terms_pmid ON article_mesh_terms(pmid);
CREATE INDEX IF NOT EXISTS idx_article_mesh_terms_pmcid ON article_mesh_terms(pmcid);
CREATE INDEX IF NOT EXISTS idx_article_mesh_terms_mesh_id ON article_mesh_terms(mesh_term_id);
CREATE INDEX IF NOT EXISTS idx_article_mesh_terms_major ON article_mesh_terms(is_major_topic);

-- Article-MeSH term qualifier linking table
CREATE TABLE IF NOT EXISTS article_mesh_qualifiers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_mesh_term_id INTEGER REFERENCES article_mesh_terms(id),
    qualifier_id INTEGER REFERENCES mesh_qualifiers(qualifier_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_mesh_term_id, qualifier_id)
);

CREATE INDEX IF NOT EXISTS idx_article_mesh_qualifiers_amt ON article_mesh_qualifiers(article_mesh_term_id);
CREATE INDEX IF NOT EXISTS idx_article_mesh_qualifiers_qual ON article_mesh_qualifiers(qualifier_id);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    doi VARCHAR(40) UNIQUE,
    pmid VARCHAR(30) UNIQUE,
    pmcid VARCHAR(30) UNIQUE,
    embedding BLOB NOT NULL,
    model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_doi ON embeddings(doi);
CREATE INDEX IF NOT EXISTS idx_embeddings_pmid ON embeddings(pmid);
CREATE INDEX IF NOT EXISTS idx_embeddings_pmcid ON embeddings(pmcid);

-- LLM-extracted raw data table
CREATE TABLE IF NOT EXISTS llm_raw_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    dataset_id VARCHAR(100) NOT NULL,
    data_repository VARCHAR(100),
    url TEXT,
    evidence_text TEXT,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, dataset_id, model),
    UNIQUE(pmid, dataset_id, model),
    UNIQUE(pmcid, dataset_id, model)
);

CREATE INDEX IF NOT EXISTS idx_llm_raw_data_doi ON llm_raw_data(doi);
CREATE INDEX IF NOT EXISTS idx_llm_raw_data_pmid ON llm_raw_data(pmid);
CREATE INDEX IF NOT EXISTS idx_llm_raw_data_pmcid ON llm_raw_data(pmcid);
CREATE INDEX IF NOT EXISTS idx_llm_raw_data_dataset_id ON llm_raw_data(dataset_id);
CREATE INDEX IF NOT EXISTS idx_llm_raw_data_repository ON llm_raw_data(data_repository);
CREATE INDEX IF NOT EXISTS idx_llm_raw_data_model ON llm_raw_data(model);

-- LLM-extracted processed data table
CREATE TABLE IF NOT EXISTS llm_processed_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    dataset_id VARCHAR(100) NOT NULL,
    data_repository VARCHAR(100),
    url TEXT,
    file TEXT,
    evidence_text TEXT,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, dataset_id, model),
    UNIQUE(pmid, dataset_id, model),
    UNIQUE(pmcid, dataset_id, model)
);

CREATE INDEX IF NOT EXISTS idx_llm_processed_data_doi ON llm_processed_data(doi);
CREATE INDEX IF NOT EXISTS idx_llm_processed_data_pmid ON llm_processed_data(pmid);
CREATE INDEX IF NOT EXISTS idx_llm_processed_data_pmcid ON llm_processed_data(pmcid);
CREATE INDEX IF NOT EXISTS idx_llm_processed_data_dataset_id ON llm_processed_data(dataset_id);
CREATE INDEX IF NOT EXISTS idx_llm_processed_data_repository ON llm_processed_data(data_repository);
CREATE INDEX IF NOT EXISTS idx_llm_processed_data_model ON llm_processed_data(model);

-- LLM-extracted analysis methods table
CREATE TABLE IF NOT EXISTS llm_analysis_methods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    method_name VARCHAR(200) NOT NULL,
    method_description TEXT,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, method_name, model),
    UNIQUE(pmid, method_name, model),
    UNIQUE(pmcid, method_name, model)
);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_methods_doi ON llm_analysis_methods(doi);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_methods_pmid ON llm_analysis_methods(pmid);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_methods_pmcid ON llm_analysis_methods(pmcid);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_methods_name ON llm_analysis_methods(method_name);
CREATE INDEX IF NOT EXISTS idx_llm_analysis_methods_model ON llm_analysis_methods(model);

-- LLM-extracted code table
-- Stores code repository references associated with articles, including:
-- - LLM-extracted repository references (repository, description, model)
-- - Fetched repository metadata (size_mb, file_count, primary_language, local_path)
CREATE TABLE IF NOT EXISTS llm_code (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    repository VARCHAR(200),
    url TEXT,
    description TEXT,
    model VARCHAR(100),
    -- Repository fetch metadata (populated by repo_fetcher)
    size_mb INTEGER,
    file_count INTEGER,
    primary_language VARCHAR(50),
    local_path TEXT,
    fetched_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, repository, model),
    UNIQUE(pmid, repository, model),
    UNIQUE(pmcid, repository, model)
);

CREATE INDEX IF NOT EXISTS idx_llm_code_doi ON llm_code(doi);
CREATE INDEX IF NOT EXISTS idx_llm_code_pmid ON llm_code(pmid);
CREATE INDEX IF NOT EXISTS idx_llm_code_pmcid ON llm_code(pmcid);
CREATE INDEX IF NOT EXISTS idx_llm_code_repository ON llm_code(repository);
CREATE INDEX IF NOT EXISTS idx_llm_code_model ON llm_code(model);
CREATE INDEX IF NOT EXISTS idx_llm_code_url ON llm_code(url);
CREATE INDEX IF NOT EXISTS idx_llm_code_primary_language ON llm_code(primary_language);

-- LLM extraction records table (stores prompts and responses)
CREATE TABLE IF NOT EXISTS llm_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_llm_extractions_doi ON llm_extractions(doi);
CREATE INDEX IF NOT EXISTS idx_llm_extractions_pmid ON llm_extractions(pmid);
CREATE INDEX IF NOT EXISTS idx_llm_extractions_pmcid ON llm_extractions(pmcid);
CREATE INDEX IF NOT EXISTS idx_llm_extractions_model ON llm_extractions(model);

-- Supplemental file classifications table
CREATE TABLE IF NOT EXISTS supplemental_file_classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    archive_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    classification VARCHAR(30) NOT NULL CHECK (classification IN ('raw_data', 'quantitative_data', 'summary', 'processing_parameters', 'instrument_parameters', 'unknown')),
    justification TEXT,
    model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(archive_path, file_path)
);

CREATE INDEX IF NOT EXISTS idx_suppl_class_doi ON supplemental_file_classifications(doi);
CREATE INDEX IF NOT EXISTS idx_suppl_class_pmid ON supplemental_file_classifications(pmid);
CREATE INDEX IF NOT EXISTS idx_suppl_class_pmcid ON supplemental_file_classifications(pmcid);
CREATE INDEX IF NOT EXISTS idx_suppl_class_classification ON supplemental_file_classifications(classification);

-- Agent requests table for human review
CREATE TABLE IF NOT EXISTS agent_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request TEXT NOT NULL,
    reason_for_request TEXT,
    request_status TEXT DEFAULT 'pending' CHECK (request_status IN ('pending', 'approved', 'rejected', 'in_progress', 'implemented', 'incomplete', 'completed', 'cancelled')),
    status_reason TEXT,
    assigned_time TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB,
    embedding_model VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_agent_requests_status ON agent_requests(request_status);
CREATE INDEX IF NOT EXISTS idx_agent_requests_creation ON agent_requests(creation_time);
CREATE INDEX IF NOT EXISTS idx_agent_requests_agent ON agent_requests(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_requests_assigned ON agent_requests(assigned_time);

-- Datasets table for ProteomeXchange and other repositories
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR(50) PRIMARY KEY,
    title TEXT,
    description TEXT,
    species TEXT,
    instrument TEXT,
    repository VARCHAR(50),
    submission_date DATE,
    publication_date DATE,
    local_filepath TEXT,
    doi VARCHAR(100),
    pmid VARCHAR(30),
    pmcid VARCHAR(30),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_datasets_repository ON datasets(repository);
CREATE INDEX IF NOT EXISTS idx_datasets_submission_date ON datasets(submission_date);
CREATE INDEX IF NOT EXISTS idx_datasets_doi ON datasets(doi);
CREATE INDEX IF NOT EXISTS idx_datasets_pmid ON datasets(pmid);
CREATE INDEX IF NOT EXISTS idx_datasets_pmcid ON datasets(pmcid);

-- Dataset files table for cataloging individual files within datasets
CREATE TABLE IF NOT EXISTS dataset_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id VARCHAR(50) NOT NULL REFERENCES datasets(dataset_id),
    filename TEXT NOT NULL,
    file_type VARCHAR(50),
    file_type_reason TEXT,
    method VARCHAR(20) CHECK (method IN ('heuristic', 'shallow_llm', 'llm') OR method IS NULL),
    model VARCHAR(100),
    size_bytes INTEGER,
    local_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_id, filename)
);

CREATE INDEX IF NOT EXISTS idx_dataset_files_dataset_id ON dataset_files(dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_files_file_type ON dataset_files(file_type);
CREATE INDEX IF NOT EXISTS idx_dataset_files_method ON dataset_files(method);

-- UniProt FASTA reference proteomes table
-- Stores metadata for reference proteomes from UniProt, used to construct
-- FTP URLs for downloading FASTA files.
CREATE TABLE IF NOT EXISTS uniprot_fasta (
    proteome_id VARCHAR(20) PRIMARY KEY,
    tax_id INTEGER NOT NULL,
    oscode VARCHAR(10),
    superregnum VARCHAR(20) NOT NULL,
    "#(1)" INTEGER,
    "#(2)" INTEGER,
    "#(3)" INTEGER,
    species_name TEXT NOT NULL,
    local_filepath TEXT,
    sha256sum TEXT
);

CREATE INDEX IF NOT EXISTS idx_uniprot_fasta_tax_id ON uniprot_fasta(tax_id);
CREATE INDEX IF NOT EXISTS idx_uniprot_fasta_oscode ON uniprot_fasta(oscode);
CREATE INDEX IF NOT EXISTS idx_uniprot_fasta_superregnum ON uniprot_fasta(superregnum);
CREATE INDEX IF NOT EXISTS idx_uniprot_fasta_species ON uniprot_fasta(species_name);

-- ===========================================================================
-- Provenance / research-object layer (Phase 2)
-- These tables make every quantification/analysis result a reproducible
-- research object by stamping tool versions, container/parameter hashes,
-- commands, hosts, and model/provider provenance. All tables are additive
-- (CREATE TABLE IF NOT EXISTS) and do not modify existing tables.
--
-- NOTE (migration): the existing llm_* tables (llm_raw_data, llm_processed_data,
-- llm_analysis_methods, llm_code, llm_keywords, llm_extractions) predate the
-- provider/run_at provenance columns used below. Because a CREATE TABLE
-- IF NOT EXISTS is a no-op against an already-populated database, adding
-- provider/run_at to those tables here would NOT take effect on the live
-- articles.sqlite. Backfilling those columns requires an explicit ALTER TABLE
-- migration and is intentionally left out of this schema.
-- ===========================================================================

-- Quantification runs table
-- One row per execution of an omic quantification tool (e.g., DIA-NN, MaxQuant)
-- against a dataset. Captures full provenance for reproducibility: tool version,
-- container image + digest, parameter file + hash, command line, input files,
-- output directory, exit status, wall time, host, and (optional) LLM
-- model/provider used to derive parameters.
CREATE TABLE IF NOT EXISTS quantification_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id VARCHAR(50),
    tool VARCHAR(100),
    tool_version VARCHAR(100),
    container_image TEXT,
    container_sha256 VARCHAR(64),
    param_file_path TEXT,
    param_file_sha256 VARCHAR(64),
    command TEXT,
    input_files_json TEXT,
    output_dir TEXT,
    exit_status INTEGER,
    wall_time_sec REAL,
    host TEXT,
    extraction_model VARCHAR(100),
    provider VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_quant_runs_dataset_id ON quantification_runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_quant_runs_tool ON quantification_runs(tool);
CREATE INDEX IF NOT EXISTS idx_quant_runs_created ON quantification_runs(created_at);

-- Analysis runs table
-- One row per downstream analysis (QC, differential expression (DE),
-- enrichment, etc.) performed on quantified data. Optionally links back to the
-- quantification_run that produced its inputs. Captures method/library
-- versions, parameters, code hash, random seed, input/output paths, and
-- model/provider provenance.
CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quantification_run_id INTEGER REFERENCES quantification_runs(id),
    analysis_type VARCHAR(50),
    method VARCHAR(200),
    library VARCHAR(200),
    library_version VARCHAR(100),
    parameters_json TEXT,
    code_sha256 VARCHAR(64),
    random_seed INTEGER,
    input_paths_json TEXT,
    output_paths_json TEXT,
    provider VARCHAR(100),
    model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_quant_run ON analysis_runs(quantification_run_id);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_type ON analysis_runs(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_created ON analysis_runs(created_at);

-- Differential expression (DE/DEP) results table
-- One row per feature (protein/peptide/gene) per analysis run, storing the
-- effect size and significance. Linked to the analysis_run that produced it.
CREATE TABLE IF NOT EXISTS dep_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id INTEGER REFERENCES analysis_runs(id),
    feature_id TEXT,
    log2fc REAL,
    pvalue REAL,
    padj REAL,
    direction VARCHAR(10),
    significant BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dep_results_analysis_run ON dep_results(analysis_run_id);
CREATE INDEX IF NOT EXISTS idx_dep_results_feature ON dep_results(feature_id);
CREATE INDEX IF NOT EXISTS idx_dep_results_significant ON dep_results(significant);

-- Benchmark annotations table
-- Human/ground-truth labels used to evaluate predictions. Linked to an article
-- (doi/pmid/pmcid) and/or a dataset, with an annotator, label, category, and
-- supporting evidence text.
CREATE TABLE IF NOT EXISTS benchmark_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40),
    pmid VARCHAR(30),
    pmcid VARCHAR(30),
    dataset_id VARCHAR(50),
    annotator TEXT,
    label TEXT,
    category VARCHAR(100),
    evidence_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmark_annotations_doi ON benchmark_annotations(doi);
CREATE INDEX IF NOT EXISTS idx_benchmark_annotations_pmid ON benchmark_annotations(pmid);
CREATE INDEX IF NOT EXISTS idx_benchmark_annotations_pmcid ON benchmark_annotations(pmcid);
CREATE INDEX IF NOT EXISTS idx_benchmark_annotations_dataset ON benchmark_annotations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_annotations_category ON benchmark_annotations(category);

-- Benchmark predictions table
-- Model-generated predictions to be compared against benchmark_annotations.
-- Linked to an article (doi/pmid/pmcid) and/or a dataset, with a predicted
-- label, confidence, and model/provider provenance plus a run timestamp.
CREATE TABLE IF NOT EXISTS benchmark_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40),
    pmid VARCHAR(30),
    pmcid VARCHAR(30),
    dataset_id VARCHAR(50),
    predicted_label TEXT,
    confidence REAL,
    model VARCHAR(100),
    provider VARCHAR(100),
    run_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmark_predictions_doi ON benchmark_predictions(doi);
CREATE INDEX IF NOT EXISTS idx_benchmark_predictions_pmid ON benchmark_predictions(pmid);
CREATE INDEX IF NOT EXISTS idx_benchmark_predictions_pmcid ON benchmark_predictions(pmcid);
CREATE INDEX IF NOT EXISTS idx_benchmark_predictions_dataset ON benchmark_predictions(dataset_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_predictions_model ON benchmark_predictions(model);

-- ===========================================================================
-- Question-conditioned relevance gate (feature request #53)
-- Two additive tables that let cross-study aggregation pool only studies that
-- directly measure the analyte of interest in the correct biological
-- system/compartment under the correct contrast. Both are new tables, so
-- CREATE TABLE IF NOT EXISTS takes effect on the live articles.sqlite.
-- ===========================================================================

-- Ingestion-time measurement descriptor.
-- Captured as extra fields on the EXISTING LLM extraction pass (near-zero
-- marginal cost -- same LLM call). Describes WHAT/WHERE/HOW a study measures so
-- a question-time relevance score can be computed cheaply against this cached
-- descriptor and reused across many questions. One row per article per model.
CREATE TABLE IF NOT EXISTS llm_measurement_descriptors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40) REFERENCES articles(doi),
    pmid VARCHAR(30) REFERENCES articles(pmid),
    pmcid VARCHAR(30) REFERENCES articles(pmcid),
    biological_system TEXT,          -- biological system / cell type
    measured_compartment VARCHAR(50),-- whole-cell | EV/exosome | secretome | tissue | nuclei | cell-type-specific in vivo | other/unknown
    species TEXT,
    perturbations TEXT,              -- perturbations / contrasts studied
    omics_assay TEXT,                -- omics / assay modality
    evidence_text TEXT,
    model VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doi, model),
    UNIQUE(pmid, model),
    UNIQUE(pmcid, model)
);

CREATE INDEX IF NOT EXISTS idx_llm_meas_desc_doi ON llm_measurement_descriptors(doi);
CREATE INDEX IF NOT EXISTS idx_llm_meas_desc_pmid ON llm_measurement_descriptors(pmid);
CREATE INDEX IF NOT EXISTS idx_llm_meas_desc_pmcid ON llm_measurement_descriptors(pmcid);
CREATE INDEX IF NOT EXISTS idx_llm_meas_desc_compartment ON llm_measurement_descriptors(measured_compartment);
CREATE INDEX IF NOT EXISTS idx_llm_meas_desc_model ON llm_measurement_descriptors(model);

-- Question-conditioned study relevance scores.
-- One row per (study, question) judgement, persisted for provenance so no study
-- is ever silently dropped from a cross-study comparison. Records the minimal
-- LLM judgement (score, directly_measures, reason), the derived gating verdict,
-- how much context was sent (descriptor/excerpt/full_text), whether the input
-- was escalated to full text, and the model/provider provenance.
CREATE TABLE IF NOT EXISTS study_relevance_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doi VARCHAR(40),
    pmid VARCHAR(30),
    pmcid VARCHAR(30),
    study_label TEXT,                 -- label for supplied-text studies with no stored id
    question TEXT NOT NULL,
    question_sha256 VARCHAR(64),
    score REAL,
    directly_measures BOOLEAN,
    reason TEXT,
    verdict VARCHAR(10),              -- include | exclude | flag | error
    escalated BOOLEAN DEFAULT FALSE,
    context_level VARCHAR(20),        -- descriptor | excerpt | full_text
    model VARCHAR(100),
    provider VARCHAR(100),
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_study_rel_doi ON study_relevance_scores(doi);
CREATE INDEX IF NOT EXISTS idx_study_rel_pmid ON study_relevance_scores(pmid);
CREATE INDEX IF NOT EXISTS idx_study_rel_pmcid ON study_relevance_scores(pmcid);
CREATE INDEX IF NOT EXISTS idx_study_rel_question ON study_relevance_scores(question_sha256);
CREATE INDEX IF NOT EXISTS idx_study_rel_verdict ON study_relevance_scores(verdict);
CREATE INDEX IF NOT EXISTS idx_study_rel_created ON study_relevance_scores(created_at);
