import torch
import torch.nn as nn
import torch.nn.functional as F

class CSLoss(nn.Module):
    def __init__(self, w_flow=1.0, w_geo=1.0, w_smooth=0.1):
        super(CSLoss, self).__init__()
        self.w_flow = w_flow
        self.w_geo = w_geo
        self.w_smooth = w_smooth

    def forward(self, pred_flow, target_flow, pred_depth, target_depth, traj_vars):
        """
        Compute total loss.
        
        Args:
            pred_flow (Tensor): Predicted 2D flow from projector [B, 2, H, W]
            target_flow (Tensor): Clean Motion Vectors from Student [B, 2, H, W]
            pred_depth (Tensor): Predicted depth from projector [B, 1, H, W]
            target_depth (Tensor): Depth Prior from Teacher [B, 1, H, W]
            traj_vars (Tensor): 3D trajectory variables [B, T, 3]
            
        Returns:
            loss (Tensor): Weighted sum of losses
            metrics (dict): Individual loss components
        """
        l_flow = self.flow_loss(pred_flow, target_flow)
        l_geo = self.geo_loss(pred_depth, target_depth)
        l_smooth = self.smoothness_loss(traj_vars)
        
        total_loss = (self.w_flow * l_flow) + (self.w_geo * l_geo) + (self.w_smooth * l_smooth)
        
        return total_loss, {
            "loss_flow": l_flow.item(),
            "loss_geo": l_geo.item(),
            "loss_smooth": l_smooth.item()
        }

    def flow_loss(self, pred, target):
        # L1 Loss for flow consistency
        # Ignore regions where target flow is 0 (if padded or invalid)
        mask = (target.abs().sum(dim=1, keepdim=True) > 1e-6).float()
        return F.l1_loss(pred * mask, target * mask, reduction='mean')

    def geo_loss(self, pred, target):
        # Scale-invariant depth loss or simple L1/L2
        # Here using simple L1 for demonstration as per 'Geometry Prior'
        return F.l1_loss(pred, target, reduction='mean')

    def smoothness_loss(self, traj):
        # Encourage smooth trajectory (minimize 2nd derivative / acceleration)
        # traj: [B, T, 3]
        if traj.shape[1] < 3:
            return torch.tensor(0.0, device=traj.device)
            
        # acceleration = p(t+1) - 2p(t) + p(t-1)
        acc = traj[:, 2:] - 2 * traj[:, 1:-1] + traj[:, :-2]
        return acc.norm(dim=-1).mean()
