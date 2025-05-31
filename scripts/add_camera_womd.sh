#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=testing # training, validation, testing

# Create a temporary directory for new files
TEMP_DIR="/tmp/new_camera_files_${DATA_SPLIT}"
mkdir -p "$TEMP_DIR"

# Get list of newly downloaded files
echo "Getting list of newly downloaded files..."
NEW_FILES="$TEMP_DIR/new_files.txt"

# Create or load the list of processed files
PROCESSED_FILES="womd/processed_files_${DATA_SPLIT}.txt"
if [ ! -f "$PROCESSED_FILES" ]; then
    # If no processed files list exists, create an empty one
    touch "$PROCESSED_FILES"
fi

# Get the list of all downloaded files
if [ -f "womd/downloaded_scenario_ids_${DATA_SPLIT}.txt" ]; then
    # Create a temporary file with scenario IDs without .tfrecord extension
    TEMP_DOWNLOADED="$TEMP_DIR/downloaded_without_ext.txt"
    sed 's/\.tfrecord$//' "womd/downloaded_scenario_ids_${DATA_SPLIT}.txt" > "$TEMP_DOWNLOADED"
    
    # Find files that haven't been processed yet
    grep -v -f "$PROCESSED_FILES" "$TEMP_DOWNLOADED" > "$NEW_FILES"
else
    echo "Error: No downloaded scenario IDs file found!"
    exit 1
fi

# Create a temporary directory with only the new files
echo "Creating temporary directory with new files..."
while read -r scenario_id; do
    # Copy the pickle file if it exists
    if [ -f "/workspace/scratch/cache/SMART/$DATA_SPLIT/${scenario_id}.pkl" ]; then
        cp "/workspace/scratch/cache/SMART/$DATA_SPLIT/${scenario_id}.pkl" "$TEMP_DIR/"
    fi
    # Copy the camera file if it exists
    if [ -f "/workspace/scratch/data/womd/uncompressed/lidar_and_camera/$DATA_SPLIT/${scenario_id}.tfrecord" ]; then
        cp "/workspace/scratch/data/womd/uncompressed/lidar_and_camera/$DATA_SPLIT/${scenario_id}.tfrecord" "$TEMP_DIR/"
    fi
done < "$NEW_FILES"

# Add debugging information
echo "Number of new files to process: $(wc -l < "$NEW_FILES")"
echo "Example new files:"
head -n 3 "$NEW_FILES"

# Process only the new files
python \
  -m src.data_preprocess \
  --mode add_camera \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --pickle_dir "$TEMP_DIR" \
  --camera_data_dir "$TEMP_DIR" \
  --output_dir /workspace/scratch/cache/SMART_with_camera

# Update the list of processed files
cat "$NEW_FILES" >> "$PROCESSED_FILES"

# Clean up temporary directory
rm -rf "$TEMP_DIR"