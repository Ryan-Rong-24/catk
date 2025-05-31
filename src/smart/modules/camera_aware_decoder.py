import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.modules.camera_aware_agent_decoder import CameraAwareAgentDecoder

class CameraAwareDecoder(SMARTDecoder):
    """SMART decoder with camera cross-attention.
    
    This model extends the SMART decoder by adding cross-attention layers
    to incorporate camera embeddings at selected decoder blocks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        camera_embed_dim: int = 256,  # Dimension of raw camera embeddings
        cross_attn_layers: list = None,  # Which layers to add cross-attention
    ) -> None:
        # Initialize parent class but we'll override the agent encoder
        super().__init__(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
        )
        
        # Replace the agent encoder with camera-aware version
        self.agent_encoder = CameraAwareAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            cross_attn_layers=cross_attn_layers,
        )
        
        # Project camera embeddings to model dimension
        self.camera_proj = nn.Linear(camera_embed_dim, hidden_dim)
                
    def forward(
        self, 
        tokenized_map: Dict[str, torch.Tensor], 
        tokenized_agent: Dict[str, torch.Tensor],
        camera_embeddings: list = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with camera cross-attention.
        Args:
            tokenized_map: Map features
            tokenized_agent: Agent features  
            camera_embeddings: Batch camera embeddings
        Returns:
            Dict containing predictions
        """
        # Get map features (same as SMART)
        map_feature = self.map_encoder(tokenized_map)
        
        # Camera embeddings are required for camera-aware model
        if camera_embeddings is None:
            raise ValueError("Camera embeddings are required for CameraAwareDecoder. "
                           "Use the original SMARTDecoder if you don't have camera data.")
        
        # Process camera embeddings - convert to tensor format
        processed_camera = self._process_camera_embeddings(camera_embeddings)
        
        # Pass to camera-aware agent encoder (same pattern as SMART)
        return self.agent_encoder(tokenized_agent, map_feature, processed_camera)
    
    def _process_camera_embeddings(self, camera_embeddings: list) -> Dict[str, torch.Tensor]:
        """Process camera embeddings into simple tensor format.
        
        Args:
            camera_embeddings: Batch of camera embeddings [batch_size][num_frames][camera_dict]
            
        Returns:
            Dict with camera features for cross-attention
        """
        if camera_embeddings is None:
            raise ValueError("Camera embeddings cannot be None.")
        
        device = self.camera_proj.weight.device
        
        # Collect all camera tokens across batch and frames  
        all_camera_tokens = []
        
        for batch_idx, sample_frames in enumerate(camera_embeddings):
            if not isinstance(sample_frames, list) or len(sample_frames) == 0:
                raise ValueError(f"Sample {batch_idx}: Expected non-empty list of frames")
            
            # For now, just use the first frame's camera data
            # TODO: Handle temporal camera data properly
            frame_dict = sample_frames[0]  # Use first frame
            
            if not isinstance(frame_dict, dict) or len(frame_dict) == 0:
                raise ValueError(f"Sample {batch_idx}, Frame 0: Expected non-empty dict")
            
            # Process each camera in this frame
            for camera_id, camera_emb in frame_dict.items():
                if not isinstance(camera_emb, np.ndarray):
                    raise ValueError(f"Sample {batch_idx}, Camera {camera_id}: expected numpy array")
                if camera_emb.shape != (256, 32):
                    raise ValueError(f"Sample {batch_idx}, Camera {camera_id}: expected shape (256, 32), got {camera_emb.shape}")
                
                # Convert camera embedding to tensor and project to hidden dim
                camera_tensor = torch.from_numpy(camera_emb).float().to(device)  # [256, 32]
                camera_projected = self.camera_proj(camera_tensor)  # [256, hidden_dim]
                
                # Take mean pooling to get one feature per camera
                camera_feature = camera_projected.mean(dim=0)  # [hidden_dim]
                all_camera_tokens.append(camera_feature)
        
        if len(all_camera_tokens) == 0:
            raise ValueError("No camera tokens found in camera embeddings")
        
        # Stack all camera tokens
        camera_tokens = torch.stack(all_camera_tokens)  # [total_cameras, hidden_dim]
        
        camera_features = {
            "camera_token": camera_tokens,  # [total_cameras, hidden_dim]
        }
        
        return camera_features
    
    def inference(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        sampling_scheme,
        camera_embeddings: list = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference with camera embeddings."""
        # Get map features (same as SMART)
        map_feature = self.map_encoder(tokenized_map)
        
        if camera_embeddings is None:
            raise ValueError("Camera embeddings are required for CameraAwareDecoder inference. "
                           "Use the original SMARTDecoder if you don't have camera data.")
        
        # Process camera embeddings
        processed_camera = self._process_camera_embeddings(camera_embeddings)
        
        # Pass to camera-aware agent encoder inference (same pattern as SMART)
        return self.agent_encoder.inference(tokenized_agent, map_feature, sampling_scheme, processed_camera) 