import torch
import torch.nn as nn
from typing import Dict, Any
from omegaconf import DictConfig
from pathlib import Path

from src.smart.model.smart import SMART
from src.smart.modules.camera_aware_decoder import CameraAwareDecoder
from src.smart.utils.model import set_model_for_finetuning

class CameraAwareSMART(SMART):
    """SMART model with camera cross-attention.
    
    This model extends SMART by adding camera cross-attention layers
    to incorporate visual information from camera embeddings.
    """
    
    def __init__(self, model_config: DictConfig) -> None:
        super().__init__(model_config)
        
        # Replace decoder with camera-aware version
        self.encoder = CameraAwareDecoder(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
            camera_embed_dim=model_config.camera_embed_dim,
            cross_attn_layers=model_config.cross_attn_layers,
        )
        
        # Load pretrained weights if specified
        if hasattr(model_config, "pretrained_path"):
            self.load_pretrained_weights(model_config.pretrained_path)
            
        # Set up finetuning
        set_model_for_finetuning(self.encoder, model_config.finetune)
        
    def load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained SMART weights and initialize new layers.
        
        Args:
            pretrained_path: Path to pretrained SMART checkpoint
        """
        # Load pretrained state dict
        pretrained = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in pretrained:
            pretrained = pretrained["state_dict"]
            
        # Filter out new layers
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained.items() 
            if k in model_dict and "camera_proj" not in k and "cross_attn_layers" not in k
        }
        
        # Update model weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
    def forward(
        self,
        tokenized_map: Dict[str, torch.Tensor],
        tokenized_agent: Dict[str, torch.Tensor],
        camera_embeddings: list,  # List[Dict[camera_name, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with camera embeddings.
        Args:
            tokenized_map: Map features
            tokenized_agent: Agent features
            camera_embeddings: List[Dict[camera_name, np.ndarray]] for each frame
        Returns:
            Dict containing predictions
        Note:
            If you want to pool over the 256 embeddings per camera, do it before passing to the model.
        """
        return self.encoder(tokenized_map, tokenized_agent, camera_embeddings)
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with camera tokens.
        
        Args:
            batch: Batch of data including camera tokens
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        pred_dict = self(
            batch["tokenized_map"],
            batch["tokenized_agent"],
            batch["camera_tokens"],
        )
        loss = self.training_loss(pred_dict)
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step with camera tokens.
        
        Args:
            batch: Batch of data including camera tokens
            batch_idx: Batch index
        """
        pred_dict = self(
            batch["tokenized_map"],
            batch["tokenized_agent"],
            batch["camera_tokens"],
        )
        
        # Log metrics
        self.minADE(pred_dict)
        self.TokenCls(pred_dict)
        if self.val_closed_loop:
            self.wosac_metrics(pred_dict) 