import torch
import torch.nn as nn
from .projector import DifferentiableProjector

class CSLAMOptimizer(nn.Module):
    def __init__(self, sequence_length, intrinsic_matrix, init_traj=None):
        """
        Self-Supervised Optimizer Core.
        
        Args:
            sequence_length (int): Number of frames in the sequence (GOP).
            intrinsic_matrix (Tensor): Camera intrinsics.
            init_traj (Tensor, optional): Initial guess for trajectory [T, 3].
        """
        super().__init__()
        self.seq_len = sequence_length
        
        # Learnable Variables: 3D Trajectory (x, y, z) for each frame
        if init_traj is not None:
             self.traj_vars = nn.Parameter(init_traj.clone())
        else:
             # Initialize around origin or some default
             self.traj_vars = nn.Parameter(torch.zeros(sequence_length, 3))
             
        # Projector
        self.projector = DifferentiableProjector(intrinsic_matrix)

    def forward(self, frame_indices=None):
        """
        Get predicted 2D and Depth for specific frames or all.
        
        Args:
            frame_indices (Tensor, optional): Indices of frames to query.
            
        Returns:
            pred_2d (Tensor): [B, 2]
            pred_depth (Tensor): [B, 1]
            current_traj (Tensor): [B, 3]
        """
        if frame_indices is None:
            current_traj = self.traj_vars
        else:
            current_traj = self.traj_vars[frame_indices]
            
        # Project current 3D estimates to 2D
        pred_2d, pred_depth = self.projector(current_traj)
        
        return pred_2d, pred_depth, current_traj
