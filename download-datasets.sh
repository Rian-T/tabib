#!/bin/bash
# Download FRACCO and MedDialog datasets from HuggingFace for offline use
# Must run on a partition with internet access (e.g., compil)

set -e

# Disable offline mode for downloads
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=0

SCRATCH=/lustre/fsn1/projects/rech/rua/uvb79kr
DATA_DIR=$SCRATCH/tabib/data

# Create data directory
mkdir -p "$DATA_DIR"

echo "============================================"
echo "Downloading datasets to $DATA_DIR"
echo "============================================"

# FRACCO ICD Classification
echo ""
echo ">>> Downloading rntc/tabib-fracco-icd..."
huggingface-cli download rntc/tabib-fracco-icd \
    --local-dir "$DATA_DIR/rntc--tabib-fracco-icd" \
    --repo-type dataset

# FRACCO NER
echo ""
echo ">>> Downloading rntc/tabib-fracco-ner..."
huggingface-cli download rntc/tabib-fracco-ner \
    --local-dir "$DATA_DIR/rntc--tabib-fracco-ner" \
    --repo-type dataset

# MedDialog FR
echo ""
echo ">>> Downloading rntc/tabib-meddialog-fr..."
huggingface-cli download rntc/tabib-meddialog-fr \
    --local-dir "$DATA_DIR/rntc--tabib-meddialog-fr" \
    --repo-type dataset

echo ""
echo "============================================"
echo "Download complete!"
echo "============================================"
echo ""
echo "Downloaded to:"
ls -la "$DATA_DIR"
