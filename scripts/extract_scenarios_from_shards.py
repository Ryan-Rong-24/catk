#!/usr/bin/env python3
"""
Extract individual scenario protos from sharded TFRecord files, given a list of scenario IDs.

Usage:
  python extract_scenarios_from_shards.py \
    --shard_dir /path/to/scenario/training \
    --scenario_id_file downloaded_scenario_ids_training.txt \
    --output_dir /path/to/extracted_scenarios/training
"""
import os
from pathlib import Path
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from tqdm import tqdm
import argparse

def extract_scenarios(shard_dir, scenario_id_file, output_dir):
    # Load scenario IDs to extract
    with open(scenario_id_file, 'r') as f:
        scenario_ids = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(scenario_ids)} scenario IDs to extract.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # List all sharded files
    shard_files = sorted(Path(shard_dir).glob("*.tfrecord*"))
    found = 0

    for shard_file in tqdm(shard_files, desc="Scanning shards"):
        dataset = tf.data.TFRecordDataset(str(shard_file), compression_type="")
        for raw_record in dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw_record.numpy())
            sid = scenario.scenario_id
            if sid in scenario_ids:
                # Save as individual file
                out_path = output_dir / f"{sid}.tfrecord"
                with tf.io.TFRecordWriter(str(out_path)) as writer:
                    writer.write(raw_record.numpy())
                found += 1
                scenario_ids.remove(sid)
                if len(scenario_ids) == 0:
                    print("All scenarios found and extracted.")
                    return
    print(f"Extraction complete. {found} scenarios extracted. {len(scenario_ids)} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract scenarios from sharded TFRecords by scenario ID.")
    parser.add_argument("--shard_dir", type=str, required=True, help="Directory with sharded scenario TFRecords")
    parser.add_argument("--scenario_id_file", type=str, required=True, help="Text file with scenario IDs to extract (one per line)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted scenario files")
    args = parser.parse_args()
    extract_scenarios(args.shard_dir, args.scenario_id_file, args.output_dir) 