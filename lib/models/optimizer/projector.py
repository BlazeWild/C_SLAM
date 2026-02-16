import torch
import torch.nn as nn

class DifferentiableProjector(nn.Module):
    def __init__(self, intrinsic_matrix):
        """
        Args:
            intrinsic_matrix (Tensor): [3, 3] Camera intrinsics (fx, fy, cx, cy)
        """
        super().__init__()
        self.register_buffer('K', intrinsic_matrix)

    def forward(self, points_3d):
        """
        Project 3D points to 2D image plane.
        
        Args:
            points_3d (Tensor): [B, N, 3] or [B, 3] in Camera Coordinate System.
            
        Returns:
            points_2d (Tensor): [B, N, 2] or [B, 2] Normalized or Pixel coordinates
            depth (Tensor): [B, N, 1] or [B, 1] Z-depth
        """
        # Extract X, Y, Z
        X = points_3d[..., 0]
        Y = points_3d[..., 1]
        Z = points_3d[..., 2]
        
        # Avoid division by zero
        Z = torch.clamp(Z, min=1e-5)
        
        # Perspective Projection
        # u = fx * (X/Z) + cx
        # v = fy * (Y/Z) + cy
        
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        
        points_2d = torch.stack([u, v], dim=-1)
        depth = Z.unsqueeze(-1)
        
        return points_2d, depth
