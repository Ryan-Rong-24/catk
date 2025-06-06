# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from google.protobuf import text_format
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
import logging as log


class WOSACMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(self, prefix: str, ego_only: bool = False) -> None:
        super().__init__()
        self.is_mp_init = False
        self.prefix = prefix
        self.ego_only = ego_only
        self.wosac_config = self.load_metrics_config()

        self.field_names = [
            "metametric",
            "average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            "distance_to_nearest_object_likelihood",
            "collision_indication_likelihood",
            "time_to_collision_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
            "min_average_displacement_error",
            "simulated_collision_rate",
            "simulated_offroad_rate",
        ]
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    def _compute_scenario_metrics(
        config, scenario_file, scenario_rollout, ego_only
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        log.info(f"Loading scenario from {scenario_file}")
        scenario = scenario_pb2.Scenario()
        dataset = tf.data.TFRecordDataset([scenario_file], compression_type="")
        for data in dataset:
            scenario.ParseFromString(bytes(data.numpy()))
            break
        log.info("Scenario loaded successfully")
        
        if ego_only:
            log.info("Processing ego-only scenario")
            for i in range(len(scenario.tracks)):
                if i != scenario.sdc_track_index:
                    for t in range(91):
                        scenario.tracks[i].states[t].valid = False
            while len(scenario.tracks_to_predict) > 1:
                scenario.tracks_to_predict.pop()
            scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index
            log.info("Ego-only processing complete")

        log.info("Computing scenario metrics...")
        try:
            metrics = wosac_metrics.compute_scenario_metrics_for_bundle(
                config, scenario, scenario_rollout
            )
            log.info("Metrics computation complete")
            return metrics
        except Exception as e:
            log.error(f"Error computing metrics: {str(e)}")
            raise

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:
        log.info(f"Starting WOSAC metrics update for {len(scenario_files)} scenarios")
        
        # Always use sequential processing for validation
        log.info("Running metrics computation sequentially...")
        pool_scenario_metrics = []
        
        for i, (_scenario, _scenario_rollout) in enumerate(zip(scenario_files, scenario_rollouts)):
            log.info(f"Processing scenario {i+1}/{len(scenario_files)}")
            try:
                # Log memory usage before processing scenario
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                log.info(f"Memory usage before scenario: {mem_info.rss / 1024 / 1024:.2f} MB")
                
                metrics = self._compute_scenario_metrics(
                    self.wosac_config, _scenario, _scenario_rollout, self.ego_only
                )
                pool_scenario_metrics.append(metrics)
                
                # Log memory usage after processing scenario
                mem_info = process.memory_info()
                log.info(f"Memory usage after scenario: {mem_info.rss / 1024 / 1024:.2f} MB")
                
            except Exception as e:
                log.error(f"Error processing scenario {i+1}: {str(e)}")
                continue

        if not pool_scenario_metrics:
            log.warning("No metrics were computed successfully")
            return

        log.info("Updating metric states...")
        for scenario_metrics in pool_scenario_metrics:
            self.scenario_counter += 1
            self.metametric += scenario_metrics.metametric
            self.average_displacement_error += (
                scenario_metrics.average_displacement_error
            )
            self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += (
                scenario_metrics.linear_acceleration_likelihood
            )
            self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += (
                scenario_metrics.angular_acceleration_likelihood
            )
            self.distance_to_nearest_object_likelihood += (
                scenario_metrics.distance_to_nearest_object_likelihood
            )
            self.collision_indication_likelihood += (
                scenario_metrics.collision_indication_likelihood
            )
            self.time_to_collision_likelihood += (
                scenario_metrics.time_to_collision_likelihood
            )
            self.distance_to_road_edge_likelihood += (
                scenario_metrics.distance_to_road_edge_likelihood
            )
            self.offroad_indication_likelihood += (
                scenario_metrics.offroad_indication_likelihood
            )
            self.min_average_displacement_error += (
                scenario_metrics.min_average_displacement_error
            )
            self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
            self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate
        
        log.info("WOSAC metrics update complete")

    def compute(self) -> Dict[str, Tensor]:
        log.info("="*50)
        log.info("Starting WOSAC metrics computation...")
        log.info(f"Total scenarios processed: {self.scenario_counter}")
        
        metrics_dict = {}
        for k in self.field_names:
            raw_value = getattr(self, k)
            metrics_dict[k] = raw_value / self.scenario_counter
            log.info(f"Computed {k}: raw={raw_value}, normalized={metrics_dict[k]}")

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="", **metrics_dict
        )
        log.info("Computing final metrics from mean metrics...")
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            self.wosac_config, mean_metrics
        )
        log.info(f"Final metrics computed: {final_metrics}")

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric": final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics": final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics": final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics": final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade": final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter": self.scenario_counter,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]
        
        # Print each metric value separately for clarity
        log.info("Final metrics values:")
        for k, v in out_dict.items():
            if isinstance(v, Tensor):
                log.info(f"{k}: {v.item()}")
            else:
                log.info(f"{k}: {v}")
        
        log.info("="*50)
        return out_dict

    @staticmethod
    def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
        config_path = (
            Path(wosac_metrics.__file__).parent / "challenge_2024_config.textproto"
        )
        with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
        return config
