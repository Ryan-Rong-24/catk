import torch
import torch.nn as nn
from typing import Dict, Optional, List

from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.layers.attention_layer import AttentionLayer

class CameraAwareAgentDecoder(SMARTAgentDecoder):
    """SMART agent decoder with camera cross-attention.
    
    This model extends the SMART agent decoder by adding cross-attention layers
    to incorporate camera embeddings at selected decoder blocks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        cross_attn_layers: List[int] = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
        )
        
        # Add cross-attention layers at specified blocks 
        self.cross_attn_layers = nn.ModuleList()
        if cross_attn_layers is None:
            cross_attn_layers = [num_layers // 2]  # Default to middle layer
            
        for i in range(num_layers):
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
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        camera_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with camera cross-attention.
        
        Args:
            tokenized_agent: Agent features
            map_feature: Map features  
            camera_feature: Processed camera embeddings tensor
            
        Returns:
            Dict containing predictions
        """
        # Camera features are required for camera-aware agent decoder
        if camera_feature is None:
            raise ValueError("Camera features are required for CameraAwareAgentDecoder. "
                           "Use the original SMARTAgentDecoder if you don't have camera data.")
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        # Get agent token embeddings
        feat_a = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
        )

        # Build temporal, interaction and map2agent edges
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            mask=mask,
        )

        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )
        batch_pl = torch.cat(
            [
                map_feature["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            batch_s=batch_s,
            mask=mask,
        )

        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            pos_a=pos_a,
            head_a=head_a,
            head_vector_a=head_vector_a,
            mask=mask,
            batch_s=batch_s,
            batch_pl=batch_pl,
        )

        # Camera features don't need edge building - they use global attention
        # Each agent attends to all camera features from its batch

        # Attention layers with camera cross-attention
        feat_map = (
            map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        )
        
        # Camera features - simple tensor, no temporal expansion needed yet
        feat_cam = camera_feature["camera_token"]  # [total_camera_features, hidden_dim]

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            
            # Apply camera cross-attention if this layer has it
            if self.cross_attn_layers[i] is not None:
                # Simple global attention: each agent attends to all camera features from its batch
                # Create global edges: each agent connects to all camera features
                n_agent_flat = feat_a.shape[0]  # n_step * n_agent
                n_cam = feat_cam.shape[0]
                
                # Create edge indices for full bipartite graph (all cameras to all agents)
                edge_index_cam2a = torch.stack([
                    torch.arange(n_cam, device=feat_a.device).repeat(n_agent_flat),  # camera indices
                    torch.arange(n_agent_flat, device=feat_a.device).repeat_interleave(n_cam)  # agent indices
                ])
                
                # No edge features needed for global attention
                r_cam2a = None
                
                # Apply cross-attention
                feat_a = self.cross_attn_layers[i](
                    (feat_cam, feat_a), r_cam2a, edge_index_cam2a
                )
            
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)

        # Final MLP to get outputs
        next_token_logits = self.token_predict_head(feat_a)

        return {
            "next_token_logits": next_token_logits[:, 1:-1],
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],
            "pred_pos": tokenized_agent["sampled_pos"],
            "pred_head": tokenized_agent["sampled_heading"],
            "pred_valid": tokenized_agent["valid_mask"],
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],
            "gt_head_raw": tokenized_agent["gt_head_raw"],
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],
            "gt_pos": tokenized_agent["gt_pos"],
            "gt_head": tokenized_agent["gt_heading"],
            "gt_valid": tokenized_agent["valid_mask"],
        }

    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme,
        camera_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Inference with camera features."""
        if camera_feature is None:
            raise ValueError("Camera features are required for CameraAwareAgentDecoder inference. "
                           "Use the original SMARTAgentDecoder if you don't have camera data.")
        
        # For now, fallback to parent inference 
        # TODO: Implement camera-aware autoregressive generation
        return super().inference(tokenized_agent, map_feature, sampling_scheme) 