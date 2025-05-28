#!/bin/bash
# Usage: ./download_lidar_and_camera_subset.sh <split> <num_files> <dest_dir>
# Example: ./download_lidar_and_camera_subset.sh training 5000 /data/lidar_and_camera_subset/training

set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <split> <num_files> <dest_dir>"
  exit 1
fi

SPLIT=$1
N=$2
DEST=$3
BUCKET=gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/lidar_and_camera/$SPLIT

mkdir -p "$DEST"

# Randomly select N files from the existing list
shuf womd/all_lidar_and_camera_training.txt | head -n $N > subset_${SPLIT}.txt

# Download the selected files in parallel
gsutil -m cp -I "$DEST" < subset_${SPLIT}.txt

# Save the scenario IDs for reproducibility
awk -F'/' '{print $NF}' subset_${SPLIT}.txt | sed 's/\.tfrecord$//' > downloaded_scenario_ids_${SPLIT}.txt

echo "Downloaded $N files to $DEST. Scenario IDs saved to downloaded_scenario_ids_${SPLIT}.txt." 