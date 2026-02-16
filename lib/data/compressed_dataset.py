import torch
from torch.utils.data import Dataset
import numpy as np
import os
from .video_loader import VideoLoader

class CompressedStreamDataset(Dataset):
    def __init__(self, data_root, sequence_length=30, target_resolution=(640, 360)):
        self.data_root = data_root
        self.seq_len = sequence_length
        self.target_resolution = target_resolution
        # Assume data_root is directory, check for mp4
        video_files = [f for f in os.listdir(data_root) if f.endswith('.mp4')]
        if not video_files:
             raise FileNotFoundError(f"No .mp4 files found in {data_root}")
             
        self.video_path = os.path.join(data_root, video_files[0]) # Use first video
        self.loader = VideoLoader(self.video_path, sequence_length, target_resolution=self.target_resolution)
        
        # Calculate GOP indices
        self.gop_starts = [i * sequence_length for i in range(len(self.loader))]

    def __len__(self):
        return len(self.gop_starts)

    def __getitem__(self, idx):
        start_frame = self.gop_starts[idx]
        data = self.loader.get_gop(start_frame)
        
        if data is None:
             # Handle edge case or cycle
             start_frame = 0
             data = self.loader.get_gop(0)
             
        print(f"DEBUG: Dataset __getitem__ returning for idx {idx}")
        return data
