import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import logging
import datetime
import gc

# Configure logging
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", datetime.datetime.now().strftime("train_cslam_%Y%m%d_%H%M%S.log"))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

from lib.data.compressed_dataset import CompressedStreamDataset
from lib.models.teacher.teacher_module import TeacherModule
from lib.models.student.student_module import StudentModule
from lib.models.optimizer.cslam_optimizer import CSLAMOptimizer
from lib.utils.losses import CSLoss

from lib.utils.viz_3d import plot_trajectory_3d

def main(args):
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Running on {device}")

    # 1. Dataset & Loader
    # Ensure data_root points to directory containing mp4s
    if args.data_root.endswith('.mp4'):
        # If user passed direct file path, handle it (hacky but useful)
        import os
        data_dir = os.path.dirname(args.data_root)
        # But if we want specific file, we might need to adjust dataset class or just pass dir.
        logging.info(f"Initializing CompressedStreamDataset with dir: {data_dir}")
        dataset = CompressedStreamDataset(data_dir, sequence_length=args.seq_len)
    else:
        logging.info(f"Initializing CompressedStreamDataset with root: {args.data_root}")
        # Reduce resolution to avoid OOM (480p: 854x480)
        dataset = CompressedStreamDataset(args.data_root, sequence_length=args.seq_len, target_resolution=(854, 480))
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False) # Shuffle False for seq

    # 2. Modules
    # We will initialize them lazily to save RAM
    logging.info("Modules will be initialized on-the-fly.")
    # teacher = TeacherModule(device=device).to(device) # Frozen
    # student = StudentModule().to(device) # Lightweight
    
    # Intrinsics (Placeholder: should strictly come from dataset or calib)
    # Approx for 480p (854x480)
    # 1080p K: fx=1000, cx=960, cy=540
    # 480p (scale ~ 480/1080 = 0.444 or just 480p standard)
    # Let's assume standard 480p K scaled from 1080p
    # fx_new = fx * (480/1080) ~ 1000 * 0.444 = 444
    # cx_new = 854/2 = 427
    # cy_new = 480/2 = 240
    K = torch.tensor([[444.0, 0.0, 427.0], 
                      [0.0, 444.0, 240.0], 
                      [0.0, 0.0, 1.0]]).to(device)
                      
    optimizer_module = CSLAMOptimizer(args.seq_len, K).to(device)
    
    # 3. Optimization Setup
    # Moved inside loop to be per-object
    criterion = CSLoss()
    
    # Store optimizers and trajectories per object ID
    # For simplicity in this demo with batch_size=1, we re-init per batch or just use batch 0
    object_optimizers = {} 
    
    logging.info("Starting C-SLAM Optimization...")
    
    final_traj_dict = {}

    for epoch in range(args.epochs):
        logging.info(f"Starting Epoch {epoch}")
        for batch_idx, data in enumerate(loader):
            logging.info(f"Processing Batch {batch_idx}") # Changed to INFO
            print(f"DEBUG: Processing Batch {batch_idx}", flush=True)
            if data is None: 
                logging.warning(f"Batch {batch_idx} is None, skipping.")
                continue
            
            # Unpack Data
            print("DEBUG: moving data to device...", flush=True)
            i_frame = data['i_frame'].to(device) # [B, 3, H, W]
            p_frames_mv = data['p_frames_mv'].to(device) # [B, T-1, 2, H, W]
            p_frames_res = data['p_frames_res'].to(device) # [B, T-1, 1, H, W]
            
            # --- TEACHER ACQUISITION (I-Frame) ---
            # Get priors (Masks, Depth) for Frame 0
            print("DEBUG: Initializing Teacher...", flush=True)
            teacher = TeacherModule(device=device).to(device)
            
            print("DEBUG: Calling teacher...", flush=True)
            with torch.no_grad():
                teacher_out = teacher(i_frame)
                # teacher_out['depth'] is [B, 1, 1, H, W] or similar depending on batch
                # Check shape correctness
                target_depth = teacher_out['depth'] 
                if target_depth.dim() == 5: target_depth = target_depth.squeeze(1) # [B, 1, H, W]
            
            # Extract masks before deleting teacher (masks are list of dicts, cpu/numpy usually)
            masks = teacher_out['masks']
            
            print("DEBUG: Teacher done. Deleting...", flush=True)
            del teacher
            gc.collect()
            torch.cuda.empty_cache() # If using GPU, but harmless on CPU
                
            # --- STUDENT ACQUISITION (P-Frames) ---
            # Parse MVs and Filter
            print("DEBUG: Initializing Student...", flush=True)
            student = StudentModule().to(device)
            
            p_frames_data = {'mvs': p_frames_mv, 'residuals': p_frames_res}
            print("DEBUG: Calling student...", flush=True)
            clean_mvs, confidence = student(p_frames_data)
            
            print("DEBUG: Student done. Deleting...", flush=True)
            del student
            gc.collect()
            torch.cuda.empty_cache()
            
            # --- OPTIMIZATION STEP ---
            # We now loop over masks found in Frame 0
            # masks is a list of dicts (MobileSAM format)
            # masks = teacher_out['masks'] # Already extracted above
            
            # If it's the first batch/epoch, init optimizers for these masks
            # Note: This simple logic assumes 1 scene/video per run or re-inits.
            # Real online SLAM would treat new masks as new tracklets.
            if len(object_optimizers) == 0:
                print(f"DEBUG: Found {len(masks)} masks. Initializing optimizers...", flush=True)
                for i, mask_data in enumerate(masks):
                     # mask_data['segmentation'] is boolean mask
                     logging.info(f"Initializing optimizer for Object {i}")
                     opt_mod = CSLAMOptimizer(args.seq_len, K).to(device)
                     optimizer = optim.Adam(opt_mod.parameters(), lr=args.lr)
                     object_optimizers[i] = {'model': opt_mod, 'opt': optimizer}

            # Loop over objects
            total_loss = 0
            for obj_id, obj_data in object_optimizers.items():
                opt = obj_data['opt']
                model = obj_data['model']
                
                # Get Mask for this object
                # For basic demo, we assume the mask from frame 0 propagates simply or we use flow to track it?
                # Actually, the student/clean_mvs gives us flow. We need to know which pixels belong to this object.
                # STRICTLY SPEAKING: We need mask propagation (e.g. XMem) or just use Frame 0 mask 
                # and assume short sequence/small motion for the *mask area* check, 
                # OR, more robustly, we use the mask to filter the INITIAL flow, 
                # and then maybe assume the object stays roughly there?
                #
                # Let's do: Use Frame 0 mask to select relevant flow vectors from p_frames_mv[:, 0]
                # And assume that's the object motion.
                
                mask_def = masks[obj_id]['segmentation'] # [H, W] bool
                # Resize mask to match flow if needed (flow usually full res, but we might have resized internal SAM?)
                # We resized masks back to original in teacher, so it should be [H, W]
                
                mask_tensor = torch.from_numpy(mask_def).to(device).float() # [H, W]
                # Broadcast to [B, T-1, 1, H, W] for simplistic weighting?
                # Actually just need it for the loss or flow averaging.
                
                opt.zero_grad()
                
                # Forward Projector
                pred_2d, pred_depth, current_traj = model() 
                
                # Add batch dimension if missing [T, 2] -> [1, T, 2]
                if pred_2d.dim() == 2:
                    pred_2d = pred_2d.unsqueeze(0)
                
                # Get Target Flow for this object
                # [B, T-1, 2, H, W]
                # We want the MEAN flow of pixels inside the mask.
                # Simplification: Look at Flow(0->1), Flow(0->2)... 
                # Student returns flow for pairs? 
                # CompressedDataset/Student returns flow from 0->1, 1->2 usually?
                # Let's check `p_frames_mv` structure. compressed_dataset usually gives motion vectors.
                # Assuming p_frames_mv is flow relative to previous frame or frame 0? 
                # Standard P-frame is Ref->Cur. 
                # Let's assume we just want to minimize Reprojection Error vs Mean Flow.
                
                # Mask the flow
                # p_frames_mv: [B, T-1, 2, H, W]
                # Select spatial region
                # We aggregate flow over the mask area
                
                # Reshape mask to broadcast
                # mask_tensor: [H, W] -> [H, W]
                # clean_mvs: [B, T, 2, H, W]
                
                # Check for mask mismatch if we resized resolution
                if mask_tensor.shape[-2:] != clean_mvs.shape[-2:]:
                    # Resize mask to flow resolution
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor[None, None, ...], 
                        size=clean_mvs.shape[-2:], 
                        mode='nearest'
                    )[0, 0]
                
                # Use Boolean Indexing to save memory
                # mask_bool: [H, W]
                mask_bool = mask_tensor > 0.5
                
                if mask_bool.sum() == 0:
                     # Fallback if mask is empty
                     target_vel = torch.zeros_like(pred_2d[:, 1:, :])
                else:
                    # Select flows inside mask: [B, T, 2, N_points]
                    # clean_mvs is [1, 29, 2, 360, 640]
                    # We want to select spatials.
                    # Flatten spatial: [1, 29, 2, H*W]
                    B, T, C, H, W = clean_mvs.shape
                    
                    # Method 1: Masked Select
                    # clean_mvs[..., mask_bool] -> [B, T, C, N_points]
                    selected_flow = clean_mvs[..., mask_bool]
                    
                    # Mean Flow over N_points [B, T, C]
                    mean_flow_obj = selected_flow.mean(dim=-1) 
                    
                # --- Loss Calculation ---
                # 1. Velocity Loss (MSE between Predicted Velocity and Mean Optical Flow)
                pred_vel = pred_2d[:, 1:, :] - pred_2d[:, :-1, :] # [B, T-1, 2]
                loss_flow = torch.nn.functional.mse_loss(pred_vel, mean_flow_obj)
                
                # 2. Depth Loss (MSE between Predicted Depth and Mean Depth Prior)
                # target_depth is [B, 1, H, W]
                # We need mean depth in the mask.
                # Resize mask to depth resolution if needed (usually same H, W)
                # target_depth [B, 1, 480, 854]
                if mask_bool.sum() > 0:
                    # target_depth[0, 0, mask_bool] -> [N_points]
                    mean_depth_target = target_depth[0, 0][mask_bool].mean()
                else:
                    mean_depth_target = torch.tensor(0.0).to(device)
                    
                # pred_depth is [B, T, 1] ? Estimator output
                # Let's check projector output. Usually [B, T, 1].
                loss_depth = torch.nn.functional.mse_loss(pred_depth.mean(), mean_depth_target) # Simple mean matching
                
                # 3. Smoothness Loss
                # Acceleration minimization
                if current_traj.shape[1] > 2:
                    acc = current_traj[:, 2:] - 2 * current_traj[:, 1:-1] + current_traj[:, :-2]
                    loss_smooth = acc.norm(dim=-1).mean()
                else:
                    loss_smooth = torch.tensor(0.0).to(device)
                
                # Total Loss
                w_flow, w_depth, w_smooth = 1.0, 0.5, 0.1
                loss_obj = (w_flow * loss_flow) + (w_depth * loss_depth) + (w_smooth * loss_smooth)
                
                loss_obj.backward()
                opt.step()
                
                total_loss += loss_obj.item()
                
                # Cache trajectory
                final_traj_dict[obj_id] = current_traj.detach().cpu().numpy() # [T, 3]

            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch} | Batch {batch_idx} | Total Loss: {total_loss:.4f}")
                
            # Cleanup Batch Memory
            del i_frame, p_frames_mv, p_frames_res, clean_mvs, masks, teacher_out
            torch.cuda.empty_cache()
            gc.collect()
            
    print("Optimization Complete.")
    
    if len(final_traj_dict) > 0:
        plot_trajectory_3d(final_traj_dict, "final_trajectory.png")
        logging.info("Trajectory plotted to final_trajectory.png")
    else:
        logging.warning("No trajectory created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to directory containing the video
    parser.add_argument('--data_root', type=str, default='datasets/videos') 
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logging.error("An error occurred during execution:", exc_info=True)
        raise e
