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
    
    def _combine_batch_predictions(self, pred_list):
        """Combine predictions from individual samples back into batch format."""
        if not pred_list:
            return {}
        
        # Get all keys from first prediction
        batch_pred = {}
        for key in pred_list[0].keys():
            # Stack tensors along batch dimension
            if isinstance(pred_list[0][key], torch.Tensor):
                batch_pred[key] = torch.cat([pred[key] for pred in pred_list], dim=0)
            else:
                # For non-tensor values, just take from first sample
                batch_pred[key] = pred_list[0][key]
        
        return batch_pred
        
    def training_step(self, data, batch_idx):
        """Override SMART training step to include camera embeddings."""
        tokenized_map, tokenized_agent = self.token_processor(data)
        
        # Extract and process camera embeddings from batch data
        batch_camera_embeddings = data.get("camera_embeddings", None)
        if batch_camera_embeddings is None:
            raise ValueError("Camera embeddings not found in data. Ensure you are using camera-aware datasets "
                           "processed with 'add_camera' mode. Use the original SMART model for non-camera data.")
        
        # Process camera embeddings for the batch
        # batch_camera_embeddings: List[List[Dict]] - [batch_size, num_frames, {camera_id: embedding}]
        # We need to process each sample individually since the encoder processes one sample at a time
        processed_camera_batch = []
        for sample_camera_embeddings in batch_camera_embeddings:
            processed_camera_batch.append(sample_camera_embeddings)
        
        if self.training_rollout_sampling.num_k <= 0:
            # For now, process each sample individually
            # TODO: This should be vectorized for efficiency
            batch_size = len(processed_camera_batch)
            pred_list = []
            for i in range(batch_size):
                sample_tokenized_map = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_map.items()}
                sample_tokenized_agent = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_agent.items()}
                sample_pred = self.encoder(sample_tokenized_map, sample_tokenized_agent, processed_camera_batch[i])
                pred_list.append(sample_pred)
            
            # Combine predictions back into batch format
            pred = self._combine_batch_predictions(pred_list)
        else:
            # Handle inference with camera embeddings per sample
            batch_size = len(processed_camera_batch)
            pred_list = []
            for i in range(batch_size):
                sample_tokenized_map = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_map.items()}
                sample_tokenized_agent = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_agent.items()}
                sample_pred = self.encoder.inference(
                    sample_tokenized_map,
                    sample_tokenized_agent,
                    sampling_scheme=self.training_rollout_sampling,
                    camera_embeddings=processed_camera_batch[i],
                )
                pred_list.append(sample_pred)
            
            pred = self._combine_batch_predictions(pred_list)

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
        batch_camera_embeddings = data.get("camera_embeddings", None)
        if batch_camera_embeddings is None:
            raise ValueError("Camera embeddings not found in data. Ensure you are using camera-aware datasets "
                           "processed with 'add_camera' mode. Use the original SMART model for non-camera data.")

        # Process camera embeddings for the batch
        processed_camera_batch = []
        for sample_camera_embeddings in batch_camera_embeddings:
            processed_camera_batch.append(sample_camera_embeddings)

        # Open-loop validation
        if self.val_open_loop:
            # Process each sample individually
            batch_size = len(processed_camera_batch)
            pred_list = []
            for i in range(batch_size):
                sample_tokenized_map = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_map.items()}
                sample_tokenized_agent = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_agent.items()}
                sample_pred = self.encoder(sample_tokenized_map, sample_tokenized_agent, processed_camera_batch[i])
                pred_list.append(sample_pred)
            
            pred = self._combine_batch_predictions(pred_list)
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
            for rollout_idx in range(self.n_rollout_closed_val):
                # Process each sample individually for closed-loop validation
                batch_size = len(processed_camera_batch)
                rollout_pred_list = []
                for i in range(batch_size):
                    sample_tokenized_map = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_map.items()}
                    sample_tokenized_agent = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v for k, v in tokenized_agent.items()}
                    sample_pred = self.encoder.inference(
                        sample_tokenized_map, sample_tokenized_agent, self.validation_rollout_sampling,
                        camera_embeddings=processed_camera_batch[i]
                    )
                    rollout_pred_list.append(sample_pred)
                
                pred = self._combine_batch_predictions(rollout_pred_list)
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