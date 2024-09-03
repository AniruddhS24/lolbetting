#!/bin/bash

# Directory to store downloaded CSV files
DOWNLOAD_DIR="data"

# Create the directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Associative array of file IDs and custom names
declare -A FILES
FILES=(
    ["1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN"]="2023_LoL_esports_match_data_from_OraclesElixir.csv"
    ["1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2"]="2024_LoL_esports_match_data_from_OraclesElixir.csv"
)

# Base URL for Google Drive file download
BASE_URL="https://drive.google.com/uc?export=download"

for FILE_ID in "${!FILES[@]}"; do
    CUSTOM_NAME="${FILES[$FILE_ID]}"
    # Construct the download URL
    DOWNLOAD_URL="$BASE_URL&id=$FILE_ID"
    
    # Use curl to download the file
    echo "Downloading file $CUSTOM_NAME..."
    curl -L -o "$DOWNLOAD_DIR/$CUSTOM_NAME" "$DOWNLOAD_URL"
done

echo "All files have been downloaded to $DOWNLOAD_DIR."
