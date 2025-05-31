#!/usr/bin/env python3
"""Debug script to test decoder camera embedding processing."""

import numpy as np
import torch
from omegaconf import DictConfig

# Create camera embeddings
print("Creating camera embeddings...")
camera_embeddings = [
    {
        'FRONT': np.random.randn(256, 32).astype(np.float32),
        'FRONT_LEFT': np.random.randn(256, 32).astype(np.float32),
        'FRONT_RIGHT': np.random.randn(256, 32).astype(np.float32),
    }
    for _ in range(5)  # 5 frames
]
print(f"✓ Camera embeddings created: {len(camera_embeddings)} frames")

# Test decoder import
print("Testing decoder import...")
try:
    from src.smart.modules.camera_aware_decoder import CameraAwareDecoder
    print("✓ CameraAwareDecoder imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Create minimal decoder config
print("Creating decoder config...")
decoder_config = {
    'hidden_dim': 128,
    'num_historical_steps': 11,
    'num_future_steps': 80,
    'pl2pl_radius': 150,
    'time_span': None,
    'pl2a_radius': 50,
    'a2a_radius': 50,
    'num_freq_bands': 64,
    'num_map_layers': 2,
    'num_agent_layers': 4,
    'num_heads': 8,
    'head_dim': 16,
    'dropout': 0.1,
    'hist_drop_prob': 0.1,
}

print("Creating decoder...")
try:
    decoder = CameraAwareDecoder(
        **decoder_config,
        n_token_agent=256,
        camera_embed_dim=32,
        cross_attn_layers=[2],
    )
    print("✓ Decoder created successfully")
except Exception as e:
    print(f"❌ Decoder creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test the camera embedding processing method directly
print("\nTesting camera embedding processing...")
print(f"Before call - camera_embeddings type: {type(camera_embeddings)}")
print(f"Before call - camera_embeddings is None: {camera_embeddings is None}")

try:
    print("Calling _process_camera_embeddings...")
    processed = decoder._process_camera_embeddings(camera_embeddings)
    print(f"✓ Processing successful! Shape: {processed.shape}")
except Exception as e:
    print(f"❌ Processing failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Debug the method call step by step
    print("\nDebugging method call...")
    print(f"decoder._process_camera_embeddings exists: {hasattr(decoder, '_process_camera_embeddings')}")
    
    # Check what the method receives
    method = decoder._process_camera_embeddings
    print(f"Method: {method}")
    
    # Try calling with explicit debug
    print("Trying step-by-step call...")
    try:
        print("Step 1: Check input")
        print(f"  Input type: {type(camera_embeddings)}")
        print(f"  Input is None: {camera_embeddings is None}")
        
        print("Step 2: Call method")
        result = method(camera_embeddings)
        print(f"  Result shape: {result.shape}")
    except Exception as e2:
        print(f"  Step-by-step failed: {e2}")
        traceback.print_exc()