import tensorflow as tf
import os
from pathlib import Path

def read_tfrecord_file(file_path):
    """Read a TFRecord file and print its contents."""
    try:
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        
        # Create a dataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        
        # Get the first record
        for raw_record in dataset.take(1):
            # Parse the raw record
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            # Print all available features
            print("Available features:")
            for key, feature in example.features.feature.items():
                kind = feature.WhichOneof('kind')
                if kind == 'bytes_list':
                    values = [v.decode('utf-8') for v in feature.bytes_list.value]
                elif kind == 'float_list':
                    values = feature.float_list.value
                elif kind == 'int64_list':
                    values = feature.int64_list.value
                print(f"  {key}: {values}")
            
            break  # Only process the first record
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Path to the directory containing TFRecord files
    tfrecord_dir = os.path.join("scratch", "data", "womd", "uncompressed", "scenario", "testing")
    
    # Get all TFRecord files in the directory
    tfrecord_files = list(Path(tfrecord_dir).glob("*.tfrecord*"))
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    # Process each file
    for file_path in tfrecord_files:
        read_tfrecord_file(str(file_path))

if __name__ == "__main__":
    main() 