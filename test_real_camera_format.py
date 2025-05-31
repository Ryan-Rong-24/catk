#!/usr/bin/env python3
"""Test script using real camera embedding format."""

import numpy as np
from omegaconf import DictConfig

def test_real_camera_format():
    """Test with real camera embedding format (numeric keys)."""
    print("Testing real camera embedding format...")
    
    # Create camera embeddings with numeric keys like real data
    camera_embeddings = [
        {
            1: np.random.randn(256, 32).astype(np.float32),
            2: np.random.randn(256, 32).astype(np.float32),
            3: np.random.randn(256, 32).astype(np.float32),
            4: np.random.randn(256, 32).astype(np.float32),
            5: np.random.randn(256, 32).astype(np.float32),
            6: np.random.randn(256, 32).astype(np.float32),
            7: np.random.randn(256, 32).astype(np.float32),
            8: np.random.randn(256, 32).astype(np.float32),
        }
        for _ in range(11)  # 11 frames like real data
    ]
    
    print(f"✓ Real format camera embeddings created: {len(camera_embeddings)} frames, {len(camera_embeddings[0])} cameras each")
    
    # Test decoder import
    from src.smart.modules.camera_aware_decoder import CameraAwareDecoder
    print("✓ CameraAwareDecoder imported")
    
    # Create decoder
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
    
    decoder = CameraAwareDecoder(
        **decoder_config,
        n_token_agent=256,
        camera_embed_dim=32,
        cross_attn_layers=[2],
    )
    print("✓ Decoder created")
    
    # Test processing
    processed = decoder._process_camera_embeddings(camera_embeddings)
    expected_shape = (11, 8, 256, 128)  # frames, cameras, tokens, hidden_dim
    print(f"✓ Processing successful! Shape: {processed.shape}")
    print(f"Expected shape: {expected_shape}")
    
    if processed.shape == expected_shape:
        print("🎉 Real camera format test passed!")
        return True
    else:
        print(f"❌ Shape mismatch: got {processed.shape}, expected {expected_shape}")
        return False

if __name__ == "__main__":
    test_real_camera_format()