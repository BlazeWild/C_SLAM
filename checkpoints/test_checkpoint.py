import sys
import os
import torch
import numpy as np
from torchinfo import summary

# Add project root to sys.path
# Script is in checkpoints/, so root is ../
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from lib.models.teacher.mobile_sam_wrapper import MobileSAMWrapper
    from lib.models.teacher.depth_anything_v2 import DepthAnythingV2
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("Initializing models...")
    
    # Paths to checkpoints (relative to where script is run)
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
    sam_checkpoint = os.path.join(checkpoint_dir, "mobile_sam.pt")
    depth_checkpoint = os.path.join(checkpoint_dir, "depth_anything_v2_vits.pth")
    
    device = "cpu" # Use CPU for param counting to avoid OOM
    if torch.cuda.is_available():
        # User might want to check on GPU, but param count is device agnostic.
        # However, loading might require CUDA if checkpoints were saved that way (usually handled by map_location)
        device = "cuda"
        print(f"Using device: {device}")
    else:
        print("Using CPU")

    # ==========================================
    # 1. MobileSAM
    # ==========================================
    print(f"\n[{'='*20}]\nLoading MobileSAM...\n[{'='*20}]")
    try:
        # Check if checkpoint exists, if not, wrapper downloads it usually.
        sam_wrapper = MobileSAMWrapper(checkpoint_path=sam_checkpoint, device=device)
        
        # MobileSAMWrapper wraps the model in self.model
        # The underlying model is a TinyViT based SAM
        # self.model.image_encoder is the heavy part (ViT)
        # self.model.prompt_encoder and self.model.mask_decoder are lighter.
        
        print("\n--- MobileSAM Image Encoder Summary ---")
        # Standard SAM input size is 1024x1024
        try:
            summary(sam_wrapper.model.image_encoder, input_size=(1, 3, 1024, 1024), device=device)
        except Exception as e:
            print(f"Could not run torchinfo on image_encoder: {e}")

        # Total parameters manually
        total_params = sum(p.numel() for p in sam_wrapper.model.parameters())
        trainable_params = sum(p.numel() for p in sam_wrapper.model.parameters() if p.requires_grad)
        print(f"\nMobileSAM Total Parameters: {total_params:,}")
        print(f"MobileSAM Trainable Parameters: {trainable_params:,}")

    except Exception as e:
        print(f"Error initializing/summarizing MobileSAM: {e}")
        import traceback
        traceback.print_exc()


    # ==========================================
    # 2. DepthAnythingV2
    # ==========================================
    print(f"\n[{'='*20}]\nLoading DepthAnythingV2...\n[{'='*20}]")
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    try:
        depth_model = DepthAnythingV2(**model_configs['vits'])
        
        # Load weights
        if os.path.exists(depth_checkpoint):
             print(f"Loading weights from {depth_checkpoint}")
             state_dict = torch.load(depth_checkpoint, map_location='cpu')
             # Handle different checkpoint keys
             if 'model' in state_dict:
                 state_dict = state_dict['model']
             elif 'state_dict' in state_dict:
                 state_dict = state_dict['state_dict']
             
             # DepthAnythingV2 weights might have prefixes if saved from DDP or similar
             # But here we assume standard loading
             try:
                depth_model.load_state_dict(state_dict)
             except Exception as e:
                 print(f"Error loading state dict strictly: {e}. Trying strict=False")
                 depth_model.load_state_dict(state_dict, strict=False)
        else:
             print(f"Warning: Depth checkpoint not found at {depth_checkpoint}, checking param count of initialized model.")
        
        depth_model.to(device)
        
        print("\n--- DepthAnythingV2 Summary ---")
        # DepthAnythingV2 typically uses 518x518
        try:
            summary(depth_model, input_size=(1, 3, 518, 518), device=device)
        except Exception as e:
             print(f"Could not run torchinfo on DepthAnythingV2: {e}")
             
        # Total parameters manually
        total_params = sum(p.numel() for p in depth_model.parameters())
        trainable_params = sum(p.numel() for p in depth_model.parameters() if p.requires_grad)
        print(f"\nDepthAnythingV2 Total Parameters: {total_params:,}")
        print(f"DepthAnythingV2 Trainable Parameters: {trainable_params:,}")

    except Exception as e:
        print(f"Error initializing/summarizing DepthAnythingV2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
