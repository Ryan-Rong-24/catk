# Camera-Aware SMART Model

This document describes the camera-aware SMART implementation that adds cross-attention layers for incorporating camera embeddings into trajectory prediction.

## Overview

The camera-aware SMART model extends the original SMART architecture by:
- Adding cross-attention layers at specified decoder blocks
- Processing camera embeddings from multiple camera views
- Enabling finetuning on pretrained SMART weights with frozen/unfrozen components

## Architecture

### Key Components

1. **CameraAwareSMART** (`src/smart/model/camera_aware_smart.py`)
   - Main model class extending SMART
   - Requires camera embeddings (raises error if missing)
   - Handles pretrained weight loading and finetuning setup

2. **CameraAwareDecoder** (`src/smart/modules/camera_aware_decoder.py`)
   - Replaces SMARTDecoder with camera processing capability
   - Projects camera embeddings to model dimension
   - Coordinates between map, agent, and camera features

3. **CameraAwareAgentDecoder** (`src/smart/modules/camera_aware_agent_decoder.py`)
   - Extends SMARTAgentDecoder with cross-attention layers
   - Implements bipartite attention between agent and camera features
   - Handles temporal alignment of camera data

### Data Flow

```
Camera Embeddings (List[Dict[camera_name, np.ndarray[256, 32]]])
    ↓
Camera Projection (32 → hidden_dim)
    ↓
Cross-Attention with Agent Features
    ↓
Enhanced Agent Representations
    ↓
Final Predictions
```

## Usage

### 1. Data Preparation

First, add camera embeddings to your existing SMART datasets:

```bash
# Process camera embeddings for training data
scripts/add_camera_womd.sh
```

This script:
- Downloads camera+lidar data from Waymo
- Extracts camera embeddings using the codebook
- Adds `camera_embeddings` field to existing pickle files

### 2. Training

Use the camera-aware experiment configuration:

```bash
python -m src.run experiment=camera_aware_finetune
```

Key configuration options:
- `camera_embed_dim: 32` - Dimension of input camera embeddings
- `cross_attn_layers: [2]` - Which layers to add cross-attention (0-indexed)
- `pretrained_path` - Path to pretrained SMART checkpoint
- `finetune.freeze_encoder: true` - Freeze pretrained weights initially

### 3. Model Configuration

Example model config:
```yaml
_target_: src.smart.model.camera_aware_smart.CameraAwareSMART
model_config:
  camera_embed_dim: 32
  cross_attn_layers: [2]  # Add at layer 2
  pretrained_path: "/path/to/pretrained.ckpt"
  finetune:
    freeze_encoder: true
    lr_multiplier: 0.1
```

## Error Handling

The camera-aware model enforces strict validation:

- **Missing camera embeddings**: Raises `ValueError` with clear instructions
- **None frames**: Validates all frames contain valid camera data  
- **Invalid format**: Checks camera embedding structure and dimensions

This ensures you're using the right model for your data:
- Use `CameraAwareSMART` only with camera-processed datasets
- Use original `SMART` for non-camera datasets

## Testing

Run the integration test to verify everything works:

```bash
python test_camera_integration.py
```

This tests:
- Component imports and initialization
- Camera embedding processing
- Error handling for invalid inputs
- Cross-attention computation

## Implementation Details

### Cross-Attention Mechanism

The cross-attention is implemented as bipartite attention between:
- **Query**: Agent features at each timestep
- **Key/Value**: Camera features from multiple views and tokens

```python
# Simplified cross-attention logic
agent_features: [n_agent, n_step, hidden_dim]
camera_features: [n_cameras, 256, hidden_dim]

# Create bipartite edges between all agent-camera pairs
edge_index = create_bipartite_edges(agent_features, camera_features)

# Apply cross-attention
enhanced_features = cross_attention_layer(
    (camera_features.flatten(), agent_features.flatten()),
    edge_index=edge_index
)
```

### Camera Embedding Format

Expected input format:
```python
camera_embeddings = [
    {  # Frame 0
        'FRONT': np.ndarray([256, 32]),
        'FRONT_LEFT': np.ndarray([256, 32]),
        'FRONT_RIGHT': np.ndarray([256, 32]),
        # ... other cameras
    },
    # ... more frames
]
```

### Temporal Alignment

Camera features are aligned with agent trajectory steps:
- Each frame corresponds to a trajectory timestep
- Cross-attention can access different temporal contexts
- Current implementation uses simple frame indexing

## Limitations & Future Work

1. **Inference**: Camera-aware inference for rollouts needs full implementation
2. **Temporal Modeling**: Could benefit from learned temporal alignment
3. **Spatial Attention**: No explicit spatial attention across camera views
4. **Memory Efficiency**: Large camera token sequences may require optimization

## Configuration Files

- `configs/data/camera_aware.yaml` - Data configuration for camera datasets
- `configs/model/camera_aware_smart.yaml` - Model architecture configuration  
- `configs/experiment/camera_aware_finetune.yaml` - Complete training setup

## Troubleshooting

**"Camera embeddings not found in data"**
- Ensure datasets were processed with `add_camera` mode
- Check that pickle files contain `camera_embeddings` field

**"Import could not be resolved"**
- Verify all dependencies are installed
- Check Python path includes the repository root

**Memory issues during training**
- Reduce `train_batch_size` to 1
- Consider reducing `camera_embed_dim` or number of camera tokens
- Use gradient checkpointing if available