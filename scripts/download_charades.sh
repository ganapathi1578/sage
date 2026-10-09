#!/bin/bash

# Exit on any error
set -e

echo "=========================================="
echo "Downloading Charades Dataset"
echo "=========================================="

# Define target directory relative to the script location
# Assumes script is run from project root or inside scripts/
if [ -d "scripts" ]; then
    PROJECT_ROOT="."
else
    PROJECT_ROOT=".."
fi

TARGET_DIR="${PROJECT_ROOT}/data/datasets/charades"
ZIP_FILE="${PROJECT_ROOT}/data/datasets/Charades_v1_480.zip"

# Create directories if they don't exist
mkdir -p "${TARGET_DIR}"

# Charades dataset URL
URL="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"

echo "Downloading Charades..."
echo "URL: ${URL}"
echo "This may take a while..."

# Download the ZIP
if command -v curl &> /dev/null; then
    curl -L --fail --progress-bar "${URL}" -o "${ZIP_FILE}"
elif command -v wget &> /dev/null; then
    wget --show-progress "${URL}" -O "${ZIP_FILE}"
else
    echo "Error: Neither curl nor wget is installed."
    exit 1
fi

# Check that the ZIP exists
if [ ! -f "${ZIP_FILE}" ]; then
    echo "Error: Download failed."
    exit 1
fi

echo "Download completed."

# Extract
echo "=========================================="
echo "Extracting Charades..."
echo "=========================================="

unzip -q -o "${ZIP_FILE}" -d "${TARGET_DIR}"

# Remove ZIP to save disk space
echo "Cleaning up ZIP file..."
rm "${ZIP_FILE}"

echo "=========================================="
echo "Done!"
echo "Charades dataset is available at:"
echo "${TARGET_DIR}"
echo "=========================================="
