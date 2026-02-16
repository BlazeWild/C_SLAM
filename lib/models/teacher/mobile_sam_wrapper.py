import torch
import torch.nn as nn
import os
import urllib.request
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

class MobileSAMWrapper(nn.Module):
    def __init__(self, checkpoint_path="checkpoints/mobile_sam.pt", device=None, points_per_side=1, pred_iou_thresh=0.88, crop_n_layers=0, min_mask_region_area=100):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Ensure weights exist
        if not os.path.exists(self.checkpoint_path):
            self._download_weights()
            
        # Load Model
        self.model = sam_model_registry["vit_t"](checkpoint=self.checkpoint_path)
        self.model.to(device=self.device)
        self.model.eval()

        # Reverted Quantization: causing potential instability or just not helping enough
        # if str(self.device) == 'cpu':
        #     print("Applying dynamic quantization to MobileSAM image encoder...")
        #     self.model.image_encoder = torch.quantization.quantize_dynamic(
        #         self.model.image_encoder, {nn.Linear}, dtype=torch.qint8
        #     )
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=min_mask_region_area
        )

    def _download_weights(self):
        print("Downloading MobileSAM weights...")
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
        urllib.request.urlretrieve(url, self.checkpoint_path)
        print("Download done.")

    @torch.no_grad()
    def forward(self, image):
        """
        Generate masks for an image.
        Args:
            image (np.array): RGB image [H, W, 3]
        Returns:
            list[dict]: List of masks
        """
        masks = self.mask_generator.generate(image)
        return masks
