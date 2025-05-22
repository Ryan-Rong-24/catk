import torch
import torch.nn as nn
from typing import Dict, Optional

from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.modules.attention import AttentionLayer

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
        camera_embeddings: list,  # <-- now expects list of dicts of numpy arrays [256, 32]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with camera cross-attention.
        Args:
            tokenized_map: Map features
            tokenized_agent: Agent features
            camera_embeddings: List[Dict[camera_name, np.ndarray]] for each frame
        Returns:
            Dict containing predictions
        """
        # Get map features
        map_feature = self.map_encoder(tokenized_map)
        # Process camera embeddings
        camera_emb_list = []
        for frame in camera_embeddings:
            # TODO: If you want to pool over tokens, do it here:
            # pooled = np.mean(emb, axis=0)  # [32]
            # frame_emb = torch.stack([
            #     self.camera_proj(torch.from_numpy(pooled))
            #     for emb in frame.values()
            # ])
            # For now, keep all 256 embeddings per camera
            frame_emb = torch.stack([
                self.camera_proj(torch.from_numpy(emb))  # emb: [256, 32] -> [256, hidden_dim]
                for emb in frame.values()
            ])  # [num_cameras, 256, hidden_dim]
            camera_emb_list.append(frame_emb)
        camera_emb_tensor = torch.stack(camera_emb_list)  # [num_frames, num_cameras, 256, hidden_dim]
        # Pass to agent encoder (update agent encoder to expect this shape)
        pred_dict = self.agent_encoder(
            tokenized_agent, 
            map_feature,
            camera_embeddings=camera_emb_tensor,
            cross_attn_layers=self.cross_attn_layers,
        )
        
        return pred_dict 