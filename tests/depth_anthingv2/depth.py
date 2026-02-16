import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from depth_anything_v2 import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--full-path', type=str, default='lib/models/depth_anthingv2/input.png')
    parser.add_argument('--output-path', type=str, default='lib/models/depth_anthingv2/output_depth.png')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/depth_anything_v2_vits.pth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    encoder = args.encoder
    
    model = DepthAnythingV2(**model_configs[encoder])
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # Check if state_dict is inside a key like 'model' or 'state_dict'
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    else:
        print(f"Checkpoint {args.checkpoint} not found!") 
        # For testing purposes, if checkpoint is missing, we might proceed with random weights 
        # but user specifically asked to use the checkpoint.
        print("Exiting...")
        exit(1)
    
    model = model.to(DEVICE).eval()
    
    if os.path.isfile(args.full_path):
        raw_image = cv2.imread(args.full_path)
        
        depth = model.infer_image(raw_image) # H, W
        
        # Colorize depth map (simple heatmap)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        cv2.imwrite(args.output_path, depth_color)
        print(f"Depth map saved to {args.output_path}")
    else:
        print(f"Input file {args.full_path} not found.")
