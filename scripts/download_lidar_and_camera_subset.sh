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

gsutil ls $BUCKET > $DEST_DIR/all_lidar_and_camera_${SPLIT}.txt

# Randomly select N files from the existing list
shuf $DEST_DIR/all_lidar_and_camera_${SPLIT}.txt | head -n $N > $DEST_DIR/subset_${SPLIT}.txt

# Download the selected files in parallel
gsutil -m cp -I "$DEST" < $DEST_DIR/subset_${SPLIT}.txt

# Save the scenario IDs for reproducibility
awk -F'/' '{print $NF}' $DEST_DIR/subset_${SPLIT}.txt | sed 's/\.tfrecord$//' > $DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt

echo "Downloaded $N files to $DEST. Scenario IDs saved to $DEST_DIR/downloaded_scenario_ids_${SPLIT}.txt." 