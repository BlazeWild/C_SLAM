# C-SLAM (Compressed-SLAM)

A novel self-supervised 3D object tracking system that operates directly on compressed video streams (H.264/HEVC) without full frame decoding.

## Overview

C-SLAM introduces a physics-guided optimization approach that leverages motion vectors from compressed video streams to reconstruct 3D object trajectories. By avoiding expensive frame decoding and using a teacher-student architecture, the system achieves significant computational savings while maintaining tracking accuracy.

## Key Features

- **Compressed Domain Processing**: Extracts motion vectors directly from P-frames without full decoding
- **Teacher-Student Architecture**: 
  - Teacher module processes keyframes with heavy AI models (MobileSAM, Depth-Anything-V2)
  - Student module processes P-frames using lightweight motion vector parsing
- **Self-Supervised Learning**: Uses differentiable projection and multiple loss functions for optimization
- **Physics-Guided Optimization**: Replaces black-box neural networks with interpretable physics constraints

## Architecture

The system consists of three main components:

### 1. Teacher Module (Keyframes Only)
- RGB Decoder for I-frames
- MobileSAM for object segmentation
- Depth-Anything-V2 for monocular depth estimation
- Provides ground truth anchors for each GOP (Group of Pictures)

### 2. Student Module (P-Frames)
- Motion Vector Extractor (no decoding required)
- Residual Energy Filter for confidence checking
- Produces trusted motion vectors for flow supervision

### 3. Optimization Core
- Learnable 3D trajectory variables
- Differentiable perspective projector
- Self-supervised losses:
  - **L_flow**: Flow consistency loss
  - **L_geo**: Geometry prior loss
  - **L_smooth**: Smoothness constraint

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BlazeWild/C_SLAM.git
cd C_SLAM
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pretrained models:
   - Place `depth_anything_v2_vits.pth` in `checkpoints/`
   - Place `mobile_sam.pt` in `checkpoints/`

## Usage

### Training

```bash
python train_cslam.py
```

The training script will:
- Load compressed video data
- Process keyframes with teacher models
- Extract motion vectors from P-frames
- Optimize 3D trajectories using self-supervised losses

### Testing

```bash
python tests/test_samonnx.py
```

## Project Structure

```
CSLAM/
├── lib/                    # Core library modules
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations
│   │   ├── teacher/       # Teacher models (SAM, Depth)
│   │   └── student/       # Student models (MV parser)
│   └── utils/             # Utility functions
├── checkpoints/           # Model weights (excluded from git)
├── datasets/              # Dataset storage
├── logs/                  # Training logs (excluded from git)
├── tests/                 # Test scripts
├── train_cslam.py        # Main training script
└── system_framework.md   # Detailed architecture documentation

```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- OpenCV
- FFmpeg (for video processing)
- See `requirements.txt` for complete list

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cslam2026,
  title={C-SLAM: Self-Supervised 3D Object Tracking from Compressed Video Streams},
  author={Your Name},
  year={2026}
}
```

## License

[Add your license here]

## Acknowledgments

- MobileSAM for efficient segmentation
- Depth-Anything-V2 for monocular depth estimation
- FFmpeg for video codec support

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
