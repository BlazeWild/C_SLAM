import cv2
import torch
import numpy as np
import os
import logging

class VideoLoader:
    def __init__(self, video_path, sequence_length=30, target_resolution=(1920, 1080)):
        self.video_path = video_path
        self.seq_len = sequence_length
        self.target_resolution = target_resolution # (Width, Height)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Loaded video {video_path} with {self.num_frames} frames. Target resolution: {self.target_resolution}")

    def get_gop(self, start_frame_idx):
        """
        Extract a Group of Pictures (GOP) starting from start_frame_idx.
        Returns:
            i_frame (Tensor): [3, H, W] Normalized RGB
            mvs (Tensor): [T-1, 2, H, W] Motion Vectors (Optical Flow approximation)
            residuals (Tensor): [T-1, 1, H, W] Residual Error
        """
        logging.info(f"DEBUG: seeking to frame {start_frame_idx}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        frames = []
        for _ in range(self.seq_len):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize if needed
            if self.target_resolution is not None:
                if (frame.shape[1], frame.shape[0]) != self.target_resolution:
                    frame = cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_LINEAR)
            
            frames.append(frame)
            
        if len(frames) < self.seq_len:
            return None # End of video
            
        # I-Frame (First frame of GOP)
        i_frame_bgr = frames[0]
        i_frame_rgb = cv2.cvtColor(i_frame_bgr, cv2.COLOR_BGR2RGB)
        i_frame_norm = torch.from_numpy(i_frame_rgb).permute(2, 0, 1).float() / 255.0
        
        # P-Frames Logic (Simulated with Optical Flow)
        mvs_list = []
        res_list = []
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        logging.info(f"DEBUG: starting flow loop for {len(frames)} frames")
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Optical Flow (Farneback) -> [H, W, 2]
            logging.info(f"DEBUG: calculating flow for frame {i} (shape: {prev_gray.shape} -> {curr_gray.shape})")
            if prev_gray is None or curr_gray is None:
                logging.error(f"ERROR: Frame {i} is None")
                continue
            if prev_gray.shape != curr_gray.shape:
                 logging.error(f"ERROR: Frame shape mismatch: {prev_gray.shape} vs {curr_gray.shape}")
                 continue
                 
            # Optimization: Downscale for Optical Flow to speed up CPU processing
            flow_scale_width = 512
            h, w = prev_gray.shape
            scale = flow_scale_width / w
            
            prev_small = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            curr_small = cv2.resize(curr_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
            try:
                flow_small = cv2.calcOpticalFlowFarneback(prev_small, curr_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            except Exception as e:
                logging.error(f"CRASH: Error in calcOpticalFlowFarneback at frame {i}: {e}")
                raise e
                
            # Upscale flow back to original resolution
            # flow values also need to be scaled by (1/scale) because they represent displacement in pixels
            flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
            flow = flow * (1.0 / scale)
            
            # Residuals (Difference after warping - simplified as abs diff for now)
            # In a real codec, residual is what's left after motion compensation.
            # Here we approximate with pixel difference.
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Convert to Tensor
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float() # [2, H, W]
            res_tensor = torch.from_numpy(diff).float().unsqueeze(0) / 255.0 # [1, H, W]
            
            mvs_list.append(flow_tensor)
            res_list.append(res_tensor)
            
            prev_gray = curr_gray
            
        logging.info("DEBUG: Flow loop done. Stacking MVs...")
        try:
            mvs = torch.stack(mvs_list) # [T-1, 2, H, W]
            logging.info(f"DEBUG: MVs stacked. Shape: {mvs.shape}")
        except Exception as e:
             logging.error(f"CRASH: Error stacking MVs: {e}")
             raise e

        logging.info("DEBUG: Stacking Residuals...")
        try:
            residuals = torch.stack(res_list) # [T-1, 1, H, W]
            logging.info(f"DEBUG: Residuals stacked. Shape: {residuals.shape}")
        except Exception as e:
             logging.error(f"CRASH: Error stacking Residuals: {e}")
             raise e
        
        logging.info("DEBUG: Returning from get_gop")
        
        return {
            "i_frame": i_frame_norm,
            "p_frames_mv": mvs,
            "p_frames_res": residuals,
            "seq_id": f"gop_{start_frame_idx}"
        }

    def __len__(self):
        return self.num_frames // self.seq_len
