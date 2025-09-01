#!/bin/bash

# ==============================================================================
# SCRIPT: download_data.sh
#
# DESCRIPTION:
# This script automates the download of the Beijing PM2.5 Air Quality dataset
# from the UCI Machine Learning Repository. It creates the necessary directory
# structure, includes a progress bar for the download, checks if the file
# already exists, and performs a basic validation on the downloaded file.
#
# USAGE:
# Run from the root of the project: ./scripts/download_data.sh
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
OUTPUT_DIR="data/raw"
OUTPUT_FILE="$OUTPUT_DIR/beijing_pm25_raw.csv"
# --- End Configuration ---

# 1. Create the Directory
echo "Ensuring data directory exists at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 2. Check if File Already Exists to Avoid Re-downloading
if [ -f "$OUTPUT_FILE" ]; then
    echo "-> File already exists at $OUTPUT_FILE"
    # The `du -h` command shows disk usage in a human-readable format.
    # `cut -f1` extracts just the size information.
    echo "   Current file size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    read -p "   Do you want to overwrite it? (y/n): " -n 1 -r
    echo # Move to a new line after the user's input

    # Check the user's reply. If it's not 'y' or 'Y', exit the script.
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download. Using existing file."
        exit 0
    fi
    echo "Proceeding to overwrite existing file..."
fi

# 3. Download the Data with a Progress Bar
echo "Downloading Beijing PM2.5 dataset from UCI repository..."

# curl flags:
# -L: Follow redirects to the final URL.
# --progress-bar: Display a clean, single-line progress bar instead of the default meter.
# -o: Specify the output file name.
curl -L --progress-bar -o "$OUTPUT_FILE" "$URL"

# 4. Validate the Downloaded File
if [ ! -s "$OUTPUT_FILE" ]; then
    # The `-s` test checks if the file exists and is not empty.
    echo "ERROR: Download failed or the downloaded file is empty."
    rm -f "$OUTPUT_FILE" # Clean up the empty file
    exit 1
fi

# 5. Show Success Message and File Info
echo
echo "Download complete."
echo "   Successfully saved to: $OUTPUT_FILE"
echo "   File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo "--- First 3 lines of the file ---"
head -n 3 "$OUTPUT_FILE"
echo "---------------------------------"
echo
echo "Script finished successfully. Data is ready for the next step."

