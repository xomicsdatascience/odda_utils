#!/usr/bin/env bash
# Builds the ODDA analysis-sandbox Apptainer image from analysis.def.
# The image version is taken from the first positional argument, then the
# ANALYSIS_VERSION environment variable, then the ANALYSIS_VERSION default
# declared in analysis.def. Produces analysis_v${version}.sif in this directory,
# which the odda_utils `run_analysis` tool discovers automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/analysis.def"

if [[ ! -f "$DEF_FILE" ]]; then
    echo "Error: Definition file not found: ${DEF_FILE}" >&2
    exit 1
fi

# Determine the version: explicit arg > env var > default in analysis.def.
version="${1:-${ANALYSIS_VERSION:-}}"
if [[ -z "$version" ]]; then
    version="$(grep -oP 'ANALYSIS_VERSION\s*=\s*\K[0-9][0-9A-Za-z.\-]*' "$DEF_FILE" | head -1)"
fi

if [[ -z "$version" ]]; then
    echo "Error: Could not determine analysis image version. Pass it as the first argument, set ANALYSIS_VERSION, or declare it in ${DEF_FILE}." >&2
    exit 1
fi

echo "Analysis image version: ${version}"

output="${SCRIPT_DIR}/analysis_v${version}.sif"
if [[ -f "$output" ]]; then
    echo "Skipping ${version}: ${output} already exists"
    exit 0
fi

echo "Building analysis sandbox image ${version}..."
cd "${SCRIPT_DIR}"
if apptainer build \
    --build-arg "ANALYSIS_VERSION=${version}" \
    "$output" \
    "$DEF_FILE" > /dev/null; then
    echo "Built: ${output}"
else
    echo "Error: Build failed for analysis image ${version}" >&2
    exit 1
fi

echo "Done."
