import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
import gc

# Import sub-modules
# Note: Using relative imports for package structure
from .mobile_sam_wrapper import MobileSAMWrapper
from .depth_anything_v2 import DepthAnythingV2

class TeacherModule(nn.Module):
    def __init__(self, sam_checkpoint="checkpoints/mobile_sam.pt", depth_checkpoint="checkpoints/depth_anything_v2_vits.pth", device=None):
        super().__init__()
        print("Initializing Teacher Module (Frozen)...")
        
        # MobileSAM
        # Increased points_per_side to 6 for better object coverage (CPU cost increase)
        self.sam = MobileSAMWrapper(checkpoint_path=sam_checkpoint, device=device, points_per_side=6, crop_n_layers=0)
        
        # Depth Anything V2
        # Config for vits
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        
        # Load Depth Weights
        print(f"Loading Depth weights from {depth_checkpoint}...")
        state_dict = torch.load(depth_checkpoint, map_location='cpu')
        # Check keys just in case (DepthAnythingV2 saves typically as 'model' or direct)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        self.depth_model.load_state_dict(state_dict)
        self.depth_model.eval()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def forward(self, i_frame_rgb):
        """
        Process I-Frame to get priors.
        Args:
            i_frame_rgb (torch.Tensor): [B, 3, H, W] normalized or [B, H, W, 3] uint8?
            Ideally, we handle formats here. Assuming numpy/cv2 readable for now or converting.
        
        Returns:
            dict: {
                "masks": list of masks (per batch items),
                "depth": torch.Tensor [B, 1, H, W]
            }
        """
        # Data conversion for MobileSAM/Depth (they expect different things usually)
        # MobileSAM expects numpy uint8 RGB
        # DepthAnything expects tensor or numpy
        
        # Placeholder for batch processing loop (since SAM is often single image)
        
        # Assuming batch size 1 for simplicity in this demo or loop
        # i_frame_rgb is likely Tensor [1, 3, H, W]
        
        # Convert to numpy for SAM: [H, W, 3] 0-255
        img_np = i_frame_rgb[0].permute(1, 2, 0).cpu().numpy()
        # Denormalize if it was normalized? Assuming input is roughly 0-1 or standard.
        # Let's assume input is 0-1 float tensors.
        img_np = (img_np * 255).astype(np.uint8)
        
        # 1. Segmentation
        logging.info("Teacher: Running SAM...")
        
        # Resize for SAM to avoid CPU OOM / massive slowdown
        # Target max dimension 480 (increased for better segment quality)
        h, w = img_np.shape[:2]
        scale = 480.0 / max(h, w)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            logging.info(f"Resizing for SAM: {h}x{w} -> {new_h}x{new_w}")
            img_sam = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_sam = img_np
            
        masks = self.sam(img_sam)
        logging.info("Teacher: SAM done.")
        
        # Explicit GC to free up SAM intermediate tensors
        gc.collect()
        
        # If we resized, we need to verify masks are usable. 
        # SAM returns binary masks. If we need them at original res, we should resize back?
        # Current pipeline logic for 'masks' usage needs to be checked. 
        # Assuming simple usage for now, or that downstream handles it?
        # Actually, let's resize masks back to original resolution to be safe.
        if scale < 1.0:
             logging.info("Resizing masks back to original resolution...")
             # masks is list of dicts. 'segmentation' key is [H, W] bool
             for i in range(len(masks)):
                 m = masks[i]['segmentation'].astype(np.uint8)
                 m_orig = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                 masks[i]['segmentation'] = m_orig.astype(bool)
        
        # 2. Depth
        # Depth model handles raw image internally or pre-processed tensor
        # Our modified depth_anything_v2 can take raw image via infer_image 
        # OR we call forward() with proper resize/norm.
        # Let's use infer_image for robust manual handling
        logging.info("Teacher: Running DepthAnything...")
        depth_map_np = self.depth_model.infer_image(img_np)
        logging.info("Teacher: Depth done.")
        depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        
        return {
            "masks": masks,
            "depth": depth_tensor.to(i_frame_rgb.device)
        }
