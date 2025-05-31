#!/usr/bin/env python3
"""Simple debug script to test camera embedding processing."""

import numpy as np

# Test camera embeddings creation
print("Creating camera embeddings...")
camera_embeddings = [
    {
        'FRONT': np.random.randn(256, 32).astype(np.float32),
        'FRONT_LEFT': np.random.randn(256, 32).astype(np.float32),
        'FRONT_RIGHT': np.random.randn(256, 32).astype(np.float32),
    }
    for _ in range(5)  # 5 frames
]

print(f"camera_embeddings type: {type(camera_embeddings)}")
print(f"camera_embeddings is None: {camera_embeddings is None}")
print(f"camera_embeddings length: {len(camera_embeddings)}")
print(f"First frame type: {type(camera_embeddings[0])}")
print(f"First frame keys: {list(camera_embeddings[0].keys())}")
print(f"FRONT shape: {camera_embeddings[0]['FRONT'].shape}")

# Test basic iteration
print("\nTesting iteration:")
for i, frame in enumerate(camera_embeddings):
    print(f"Frame {i}: {type(frame)}, keys: {list(frame.keys())}")
    if i >= 2:  # Just test first few
        break

print("✓ Basic camera embeddings work fine")