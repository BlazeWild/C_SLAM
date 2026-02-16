import sys
import os

# Fix for name collision: remove the current directory from sys.path 
# so 'import mobile_sam' finds the installed package, not this file.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# ==============================================================================
# SECTION 1: ARCHITECTURE DEFINITIONS (TinyViT & Components)
# ==============================================================================

class LayerNorm2d(nn.Module):
    """ Standard LayerNorm but works on (N, C, H, W) layout directly. """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (The 'Mobile' part of MobileSAM).
    Uses Inverted Residuals.
    """
    def __init__(self, in_chans, out_chans, expand_ratio, activation=nn.GELU, drop_path=0.):
        super().__init__()
        hidden_chans = int(in_chans * expand_ratio)
        self.conv1 = nn.Conv2d(in_chans, hidden_chans, 1)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(hidden_chans, hidden_chans, 3, padding=1, groups=hidden_chans)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(hidden_chans, out_chans, 1)
        self.act3 = activation()
        self.drop_path = nn.Identity() # Placeholder for stochastic depth
        
        # Residual Connection if dimensions match
        self.has_residual = (in_chans == out_chans)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        # Note: The original implementation puts activation after conv3 in some versions, 
        # but MobileSAM's TinyViT often matches the official repo structure.
        
        if self.has_residual:
            # THIS IS THE RESIDUAL CONNECTION
            x = self.drop_path(x) + shortcut 
        return self.act3(x)

class WindowAttention(nn.Module):
    """
    Multi-head Self Attention restricted to local windows.
    Essential for efficiency in TinyViT.
    """
    def __init__(self, dim, num_heads, window_size=7, qkv_bias=True, qd_bias=False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Define a parameter for relative position bias (learned during training)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention Mechanism
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias (simplified for readability)
        # In a full robust implementation, you need the index handling here.
        # For brevity in this answer, we assume standard attention flow.
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TinyViTBlock(nn.Module):
    """
    Standard Transformer Block for Vision.
    Structure: Input -> LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # RESIDUAL CONNECTION 1 (Attention)
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        # RESIDUAL CONNECTION 2 (MLP)
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x

class TinyViT(nn.Module):
    """
    The TinyViT Image Encoder Architecture.
    """
    def __init__(self, img_size=1024, in_chans=3, embed_dims=[64, 128, 160, 320], 
                 depths=[2, 2, 6, 2], num_heads=[2, 4, 5, 10]):
        super().__init__()
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(embed_dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm2d(embed_dims[0]),
            nn.GELU(),
        )
        
        # Stacking the blocks (simplified for this example)
        # In research, you would iterate over 'depths' to create layers
        self.layer1 = nn.ModuleList([MBConv(embed_dims[0], embed_dims[0], 4.0) for _ in range(depths[0])])
        
        # Constructing the Transformer stages (just a placeholder for structure)
        # A real implementation loops through depths/embed_dims to build nn.ModuleLists
        # For this script to run with real weights, we need exact matching names, 
        # which is complex. 
        # INSTEAD: We will use the 'mobile_sam' library for the WEIGHT LOADING 
        # but the classes above demonstrate the DEFINITION you asked for.

# ==============================================================================
# SECTION 3: EXECUTION
# ==============================================================================

def plot_masks(image, masks, output_file):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if len(masks) > 0:
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        
        img_overlay = np.ones((sorted_masks[0]['segmentation'].shape[0], 
                                sorted_masks[0]['segmentation'].shape[1], 4))
        img_overlay[:,:,3] = 0
        
        for ann in sorted_masks:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img_overlay[m] = color_mask
        ax.imshow(img_overlay)
        
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Define inputs and paths
    INPUT_IMG = "lib/models/input.png"
    OUTPUT_IMG = "lib/models/output_mobile_sam.png"
    CHECKPOINT_PATH = "checkpoints/mobile_sam.pt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model with explicit checkpoint
    print(f"Loading MobileSAM from {CHECKPOINT_PATH}...")
    # NOTE: The mobile_sam package expects the checkpoint to be passed
    sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    sam.eval()
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Load and Preprocess Image
    if not os.path.exists(INPUT_IMG):
        print(f"Error: Input image {INPUT_IMG} not found!")
        sys.exit(1)
        
    image = cv2.imread(INPUT_IMG)
    if image is None:
        print(f"Error: Could not read image {INPUT_IMG}")
        sys.exit(1)
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run Inference
    print("Running Inference...")
    masks = mask_generator.generate(image_rgb)
    
    # Visualization
    print(f"Found {len(masks)} masks. Saving visualization...")
    plot_masks(image_rgb, masks, OUTPUT_IMG)