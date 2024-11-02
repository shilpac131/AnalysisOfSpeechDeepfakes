#!/bin/bash

# Check if the user wants to download the dataset
if [ "$DOWNLOAD_DATASET" = "yes" ]; then
    echo "Downloading dataset..."
    python download_dataset.py
else
    echo "Skipping dataset download."
fi

# Run the main program
python main.py 
