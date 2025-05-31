#!/bin/bash
# Usage: ./download_lidar_and_camera_subset.sh <split> <num_files> <dest_dir>
# Example: ./download_lidar_and_camera_subset.sh training 5000 scratch/womd/uncompressed/lidar_and_camera/training

set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <split> <num_files> <dest_dir>"
  exit 1
fi

SPLIT=$1
N=$2
DEST=$3
DEST_DIR=womd
BUCKET=gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/lidar_and_camera/$SPLIT

mkdir -p "$DEST"

# Get list of all available files
gsutil ls $BUCKET > $DEST_DIR/all_lidar_and_camera_${SPLIT}.txt

# If we have existing downloaded files, create a list of them
if [ -f "$DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt" ]; then
    echo "Found existing downloaded files. Will skip these files."
    # Create a list of existing files with full paths
    while read -r scenario_id; do
        echo "$BUCKET/${scenario_id}.tfrecord"
    done < "$DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt" > "$DEST_DIR/existing_files_${SPLIT}.txt"
    
    # Get only new files by removing existing ones
    grep -v -f "$DEST_DIR/existing_files_${SPLIT}.txt" "$DEST_DIR/all_lidar_and_camera_${SPLIT}.txt" > "$DEST_DIR/available_new_files_${SPLIT}.txt"
else
    # If no existing files, use all available files
    cp "$DEST_DIR/all_lidar_and_camera_${SPLIT}.txt" "$DEST_DIR/available_new_files_${SPLIT}.txt"
fi

# Randomly select N files from the available new files
shuf "$DEST_DIR/available_new_files_${SPLIT}.txt" | head -n $N > "$DEST_DIR/subset_${SPLIT}.txt"

# Download the selected files in parallel
gsutil -m cp -I "$DEST" < "$DEST_DIR/subset_${SPLIT}.txt"

# Extract and append new scenario IDs
awk -F'/' '{print $NF}' "$DEST_DIR/subset_${SPLIT}.txt" | sed 's/\.tfrecord$//' > "$DEST_DIR/new_scenario_ids_${SPLIT}.txt"

# Append new IDs to existing list or create new list
if [ -f "$DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt" ]; then
    cat "$DEST_DIR/new_scenario_ids_${SPLIT}.txt" >> "$DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt"
else
    mv "$DEST_DIR/new_scenario_ids_${SPLIT}.txt" "$DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt"
fi

echo "Downloaded $N new files to $DEST. Total scenario IDs saved to $DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt." 