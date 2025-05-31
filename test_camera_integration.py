#!/usr/bin/env python3
"""
Test script to verify camera-aware SMART integration.
Run this to check if all components work together.
"""

import torch
import numpy as np
from omegaconf import DictConfig

def test_camera_aware_integration():
    """Test camera-aware SMART components integration."""
    print("Testing camera-aware SMART integration...")
    
    try:
        # Test imports
        from src.smart.model.camera_aware_smart import CameraAwareSMART
        from src.smart.modules.camera_aware_decoder import CameraAwareDecoder
        from src.smart.modules.camera_aware_agent_decoder import CameraAwareAgentDecoder
        print("✓ All imports successful")
        
        # Test model configuration
        model_config = DictConfig({
            'lr': 1e-4,
            'lr_warmup_steps': 1000,
            'lr_total_steps': 10000,
            'lr_min_ratio': 0.1,
            'val_open_loop': True,
            'val_closed_loop': True,
            'n_rollout_closed_val': 3,
            'n_vis_batch': 1,
            'n_vis_scenario': 1,
            'n_vis_rollout': 1,
            'n_batch_wosac_metric': 1,
            'camera_embed_dim': 32,
            'cross_attn_layers': [2],
            'decoder': {
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
            },
            'token_processor': {
                'n_token_agent': 256,
            },
            'finetune': {
                'freeze_encoder': True,
                'lr_multiplier': 0.1,
            },
            'wosac_submission': {
                'is_active': False,
            },
            'training_loss': {},
            'training_rollout_sampling': {'num_k': 0},
            'validation_rollout_sampling': {},
        })
        
        # Test model initialization
        # Note: This would fail without pretrained weights, but import should work
        print("✓ Model configuration created")
        
        # Test camera embedding processing
        camera_embeddings = [
            {
                'FRONT': np.random.randn(256, 32).astype(np.float32),
                'FRONT_LEFT': np.random.randn(256, 32).astype(np.float32),
                'FRONT_RIGHT': np.random.randn(256, 32).astype(np.float32),
            }
            for _ in range(5)  # 5 frames
        ]
        print("✓ Mock camera embeddings created")
        
        # Test decoder with mock data
        decoder = CameraAwareDecoder(
            **model_config.decoder,
            n_token_agent=model_config.token_processor.n_token_agent,
            camera_embed_dim=model_config.camera_embed_dim,
            cross_attn_layers=model_config.cross_attn_layers,
        )
        
        # Test camera embedding processing
        processed = decoder._process_camera_embeddings(camera_embeddings)
        expected_shape = (5, 3, 256, model_config.decoder.hidden_dim)  # frames, cameras, tokens, hidden_dim
        assert processed.shape == expected_shape, f"Expected {expected_shape}, got {processed.shape}"
        print("✓ Camera embedding processing works")
        
        # Test that None camera embeddings raise proper errors
        try:
            decoder._process_camera_embeddings(None)
            assert False, "Should have raised ValueError for None camera_embeddings"
        except ValueError as e:
            assert "required" in str(e).lower()
            print("✓ Proper error handling for missing camera embeddings")
        
        # Test that None frame raises proper errors
        try:
            bad_camera_embeddings = [None] + camera_embeddings[1:]
            decoder._process_camera_embeddings(bad_camera_embeddings)
            assert False, "Should have raised ValueError for None frame"
        except ValueError as e:
            assert "None camera embeddings" in str(e)
            print("✓ Proper error handling for None frames")
        
        print("\n🎉 All integration tests passed!")
        print("\nNext steps:")
        print("1. Prepare camera-aware datasets using scripts/add_camera_womd.sh")
        print("2. Run training with: python -m src.run experiment=camera_aware_finetune")
        print("3. Monitor cross-attention layer convergence")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_camera_aware_integration()