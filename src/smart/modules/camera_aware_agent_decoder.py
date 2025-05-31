import torch
import torch.nn as nn
from typing import Dict, Optional, List

from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.modules.attention import AttentionLayer

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
        camera_embeddings: torch.Tensor,  # [num_frames, num_cameras, 256, hidden_dim]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with camera cross-attention.
        
        Args:
            tokenized_agent: Agent features
            map_feature: Map features
            camera_embeddings: [num_frames, num_cameras, 256, hidden_dim]
            
        Returns:
            Dict containing predictions
        """
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

        # Attention layers with camera cross-attention
        feat_map = (
            map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        )

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
                # Reshape camera embeddings to match current frame
                curr_frame = feat_a.view(n_step, n_agent, -1).transpose(0, 1)
                curr_camera = camera_embeddings[i % camera_embeddings.shape[0]]
                # Apply cross-attention
                feat_a = self.cross_attn_layers[i](
                    (curr_frame, curr_camera), None, None
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