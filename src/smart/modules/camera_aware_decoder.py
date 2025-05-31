import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.layers.attention_layer import AttentionLayer

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
        
        # Project camera embeddings to model dimension
        self.camera_proj = nn.Linear(camera_embed_dim, hidden_dim)
        
        # Add cross-attention layers at specified blocks
        self.cross_attn_layers = nn.ModuleList()
        if cross_attn_layers is None:
            cross_attn_layers = [num_agent_layers // 2]  # Default to middle layer
            
        for i in range(num_agent_layers):
            if i in cross_attn_layers:
                self.cross_attn_layers.append(
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dropout=dropout,
                        bipartite=True,  # Cross-attention is bipartite
                        has_pos_emb=False,  # No positional encoding for camera features
                    )
                )
            else:
                self.cross_attn_layers.append(None)
                
    def forward(
        self, 
        tokenized_map: Dict[str, torch.Tensor], 
        tokenized_agent: Dict[str, torch.Tensor],
        camera_embeddings: list = None,  # List[Dict[camera_name, np.ndarray]] for each frame
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with camera cross-attention.
        Args:
            tokenized_map: Map features
            tokenized_agent: Agent features
            camera_embeddings: List[Dict[camera_name, np.ndarray]] for each frame (optional)
        Returns:
            Dict containing predictions
        """
        # Get map features
        map_feature = self.map_encoder(tokenized_map)
        
        # Camera embeddings are required for camera-aware model
        if camera_embeddings is None:
            raise ValueError("Camera embeddings are required for CameraAwareDecoder. "
                           "Use the original SMARTDecoder if you don't have camera data.")
        
        # Process camera embeddings
        processed_camera = self._process_camera_embeddings(camera_embeddings)
        
        # Pass to agent encoder
        return self.agent_encoder(
            tokenized_agent, 
            map_feature,
            camera_embeddings=processed_camera,
            cross_attn_layers=self.cross_attn_layers,
        )
    
    def _process_camera_embeddings(self, camera_embeddings: list) -> torch.Tensor:
        """Process camera embeddings into tensor format.
        
        Args:
            camera_embeddings: List[Dict[camera_name, np.ndarray]] for each frame
            
        Returns:
            Processed camera tensor [num_frames, num_cameras, 256, hidden_dim]
        """
        if camera_embeddings is None:
            raise ValueError("Camera embeddings cannot be None. Expected list of frame dictionaries.")
        
        if not isinstance(camera_embeddings, list):
            raise ValueError(f"Camera embeddings must be a list, got {type(camera_embeddings)}.")
        
        if len(camera_embeddings) == 0:
            raise ValueError("Camera embeddings list is empty. Need at least one frame.")
        
        camera_emb_list = []
        for frame_idx, frame in enumerate(camera_embeddings):
            if frame is None:
                raise ValueError(f"Frame {frame_idx} contains None camera embeddings. "
                               f"All frames must contain valid camera data.")
            if not isinstance(frame, dict) or len(frame) == 0:
                raise ValueError(f"Frame {frame_idx} contains invalid camera embeddings. "
                               f"Expected non-empty dict, got {type(frame)}.")
            
            # Validate that all values are numpy arrays with correct shape
            for camera_id, emb in frame.items():
                if not isinstance(emb, np.ndarray):
                    raise ValueError(f"Frame {frame_idx}, camera {camera_id}: expected numpy array, got {type(emb)}")
                if emb.shape != (256, 32):
                    raise ValueError(f"Frame {frame_idx}, camera {camera_id}: expected shape (256, 32), got {emb.shape}")
            
            frame_emb = torch.stack([
                self.camera_proj(torch.from_numpy(emb).float())  # emb: [256, 32] -> [256, hidden_dim]
                for camera_id, emb in frame.items()
            ])  # [num_cameras, 256, hidden_dim]
            camera_emb_list.append(frame_emb)
        
        return torch.stack(camera_emb_list)  # [num_frames, num_cameras, 256, hidden_dim]
    
    def inference(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        sampling_scheme,
        camera_embeddings: list = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference with camera embeddings."""
        map_feature = self.map_encoder(tokenized_map)
        
        if camera_embeddings is None:
            raise ValueError("Camera embeddings are required for CameraAwareDecoder inference. "
                           "Use the original SMARTDecoder if you don't have camera data.")
        
        # Process camera embeddings
        processed_camera = self._process_camera_embeddings(camera_embeddings)
        
        # Pass to agent encoder inference
        return self.agent_encoder.inference(
            tokenized_agent, map_feature, sampling_scheme, 
            camera_embeddings=processed_camera,
            cross_attn_layers=self.cross_attn_layers
        ) 