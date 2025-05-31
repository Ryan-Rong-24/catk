#!/usr/bin/env python3
"""Debug script to inspect real camera embedding data format."""

import pickle
from pathlib import Path
import numpy as np

# Path to camera-aware data
data_dir = Path("/workspace/scratch/cache/SMART_with_camera/training")

if not data_dir.exists():
    print(f"❌ Data directory does not exist: {data_dir}")
    print("Available directories:")
    parent = data_dir.parent
    if parent.exists():
        for p in parent.iterdir():
            if p.is_dir():
                print(f"  {p}")
    exit(1)

# Get first pickle file
pickle_files = list(data_dir.glob("*.pkl"))
if not pickle_files:
    print(f"❌ No pickle files found in {data_dir}")
    exit(1)

print(f"Found {len(pickle_files)} pickle files")
first_file = pickle_files[0]
print(f"Inspecting: {first_file}")

# Load and inspect
try:
    with open(first_file, "rb") as f:
        data = pickle.load(f)
    
    print(f"\nData keys: {list(data.keys())}")
    
    if "camera_embeddings" in data:
        camera_embeddings = data["camera_embeddings"]
        print(f"\nCamera embeddings found!")
        print(f"Type: {type(camera_embeddings)}")
        
        if camera_embeddings is None:
            print("❌ Camera embeddings is None")
        elif isinstance(camera_embeddings, list):
            print(f"Length: {len(camera_embeddings)}")
            if len(camera_embeddings) > 0:
                first_frame = camera_embeddings[0]
                print(f"First frame type: {type(first_frame)}")
                
                if isinstance(first_frame, dict):
                    print(f"First frame keys: {list(first_frame.keys())}")
                    for camera_name, embedding in first_frame.items():
                        print(f"  {camera_name}: {type(embedding)}, shape: {getattr(embedding, 'shape', 'no shape')}")
                        break  # Just check first camera
                elif isinstance(first_frame, (list, np.ndarray)):
                    print(f"❌ ERROR: First frame is {type(first_frame)}, expected dict")
                    print(f"First frame shape/length: {getattr(first_frame, 'shape', len(first_frame) if hasattr(first_frame, '__len__') else 'unknown')}")
                    
                    # Check if this is a list of camera embeddings instead of dict
                    if isinstance(first_frame, list) and len(first_frame) > 0:
                        print(f"First element of first frame: {type(first_frame[0])}")
                        if hasattr(first_frame[0], 'shape'):
                            print(f"Shape: {first_frame[0].shape}")
                else:
                    print(f"❌ ERROR: Unexpected first frame type: {type(first_frame)}")
                    
                # Check a few more frames to see if pattern is consistent
                for i in range(min(3, len(camera_embeddings))):
                    frame = camera_embeddings[i]
                    print(f"Frame {i}: {type(frame)}")
        else:
            print(f"❌ ERROR: Camera embeddings is not a list: {type(camera_embeddings)}")
            
    else:
        print("❌ No camera_embeddings key found in data")
        
except Exception as e:
    print(f"❌ Error loading file: {e}")
    import traceback
    traceback.print_exc()