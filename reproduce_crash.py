import torch
from torch.utils.data import DataLoader
from lib.data.compressed_dataset import CompressedStreamDataset
import logging
import sys
import faulthandler

# Enable faulthandler to catch segfaults
faulthandler.enable()

# Configure logging
log_filename = "reproduce_crash.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def test_data_loading():
    data_root = 'datasets/videos'
    seq_len = 30
    batch_size = 1
    
    logging.info(f"Initializing CompressedStreamDataset with root: {data_root}")
    try:
        dataset = CompressedStreamDataset(data_root, sequence_length=seq_len)
    except Exception as e:
        logging.error(f"Failed to initialize dataset: {e}")
        return

    logging.info(f"Dataset length: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logging.info("Starting iteration...")
    try:
        for batch_idx, data in enumerate(loader):
            logging.info(f"Successfully loaded Batch {batch_idx}")
            if batch_idx >= 2: # Just try a few batches
                break
    except Exception as e:
        logging.error(f"Error during iteration: {e}", exc_info=True)

if __name__ == "__main__":
    test_data_loading()
