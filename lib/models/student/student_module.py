import torch
import torch.nn as nn
from .mv_parser import MVParser

class StudentModule(nn.Module):
    def __init__(self, residual_threshold=0.1):
        super().__init__()
        self.parser = MVParser()
        self.residual_threshold = residual_threshold
        
    def forward(self, p_frames_data):
        """
        Process P-frames to get Trusted Motion Vectors.
        
        Args:
            p_frames_data (dict/Tensor): Input data. 
                                         If list of dicts or batched dict from loader.
        Returns:
            clean_mvs (Tensor): [B, T, 2, H, W]
            confidence (Tensor): [B, T, 1, H, W] Binary mask of trusted pixels
        """
        # Parse
        # mvs: [B, T, 2, H, W]
        # residuals: [B, T, 1, H, W]
        mvs, residuals = self.parser(p_frames_data)
        
        # Filter Logic (Residual Filter)
        # If residual energy > threshold, confidence = 0
        confidence = (residuals < self.residual_threshold).float()
        
        # Return Trusted MVs (Zero out untrusted ones or just return mask)
        clean_mvs = mvs * confidence
        
        return clean_mvs, confidence
