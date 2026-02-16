import torch
import numpy as np

def compute_ate(pred_traj, gt_traj):
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        pred_traj (np.array): Predicted trajectory [T, 3]
        gt_traj (np.array): Ground truth trajectory [T, 3]
    
    Returns:
        float: RMSE of ATE
    """
    # Align trajectories (Horn's method or similar) if needed.
    # For now assuming aligned start/scale or simple RMSE.
    diff = pred_traj - gt_traj
    return np.sqrt((diff ** 2).mean())

def compute_rpe(pred_traj, gt_traj):
    """
    Compute Relative Pose Error (RPE).
    """
    # Placeholder for RPE logic
    return 0.0
