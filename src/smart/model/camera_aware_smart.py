import torch
from typing import Dict
from omegaconf import DictConfig

from src.smart.model.smart import SMART
from src.smart.modules.camera_aware_decoder import CameraAwareDecoder
from src.smart.utils.finetune import set_model_for_finetuning

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
        camera_embeddings: list,  # List[Dict[camera_name, np.ndarray]] - REQUIRED
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with camera embeddings.
        Args:
            tokenized_map: Map features
            tokenized_agent: Agent features
            camera_embeddings: List[Dict[camera_name, np.ndarray]] for each frame (REQUIRED)
        Returns:
            Dict containing predictions
        """
        if camera_embeddings is None:
            raise ValueError("Camera embeddings are required for CameraAwareSMART. "
                           "Use the original SMART model if you don't have camera data.")
        return self.encoder(tokenized_map, tokenized_agent, camera_embeddings)
        
    def training_step(self, data, batch_idx):
        """Override SMART training step to include camera embeddings."""
        tokenized_map, tokenized_agent = self.token_processor(data)
        
        # Extract camera embeddings from data - required for camera-aware model
        camera_embeddings = data.get("camera_embeddings", None)
        if camera_embeddings is None:
            raise ValueError("Camera embeddings not found in data. Ensure you are using camera-aware datasets "
                           "processed with 'add_camera' mode. Use the original SMART model for non-camera data.")
        
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent, camera_embeddings)
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
                camera_embeddings=camera_embeddings,
            )

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],
            token_traj=tokenized_agent["token_traj"],
            train_mask=data["agent"]["train_mask"],
            current_epoch=self.current_epoch,
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)
        return loss
        
    def validation_step(self, data, batch_idx):
        """Override SMART validation step to include camera embeddings."""
        # Use parent validation logic but with camera-aware encoder
        tokenized_map, tokenized_agent = self.token_processor(data)
        camera_embeddings = data.get("camera_embeddings", None)
        if camera_embeddings is None:
            raise ValueError("Camera embeddings not found in data. Ensure you are using camera-aware datasets "
                           "processed with 'add_camera' mode. Use the original SMART model for non-camera data.")

        # Open-loop validation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent, camera_embeddings)
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],
                token_traj=tokenized_agent["token_traj"],
            )

            self.TokenCls.update(
                pred=pred["next_token_logits"],
                pred_valid=pred["next_token_valid"],
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log("val_open/acc", self.TokenCls, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # Closed-loop validation  
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for i in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling,
                    camera_embeddings=camera_embeddings
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)
            pred_z = torch.stack(pred_z, dim=1)
            pred_head = torch.stack(pred_head, dim=1)

            # Continue with existing WOSAC evaluation logic
            scenario_rollouts = None
            if self.wosac_submission.is_active:
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    from src.utils.wosac_utils import get_scenario_rollouts
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()
            else:
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][:, self.num_historical_steps :, : pred_traj.shape[-1]],
                    target_valid=data["agent"]["valid_mask"][:, self.num_historical_steps :],
                )
                # Add WOSAC metrics computation if needed
                # ... (rest of parent validation logic) 