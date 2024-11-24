#!/bin/bash

DOWNLOAD_DIR="data/imagenet"

mkdir -p $DOWNLOAD_DIR

cd $DOWNLOAD_DIR

echo "Downloading ImageNet data from Kaggle..."
kaggle competitions download -c imagenet-object-localization-challenge

# Unzip the downloaded files
echo "Extracting downloaded files..."
unzip '*.zip'

# Remove the zip files after extraction
echo "Removing zip files..."
rm *.zip

echo "Download and extraction complete!"

