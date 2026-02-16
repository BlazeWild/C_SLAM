import os
import urllib.request
import sys

def download_file(url, destination):
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return

    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Successfully downloaded to {destination}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    # Define the checkpoints directory relative to this script
    # Script location: lib/tools/download_checkpoints.py
    # Checkpoints location: checkpoints/
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from lib/tools to reach the root
    root_dir = os.path.dirname(os.path.dirname(script_dir)) 
    checkpoints_dir = os.path.join(root_dir, 'checkpoints')

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        print(f"Created directory: {checkpoints_dir}")

    # Dictionary of Name -> (URL, Filename)
    models = {
        "Mobile SAM": (
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            "mobile_sam.pt"
        ),
        "Depth-Anything-V2-Small": (
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
            "depth_anything_v2_vits.pth"
        )
    }

    for name, (url, filename) in models.items():
        destination = os.path.join(checkpoints_dir, filename)
        print(f"Processing {name}...")
        download_file(url, destination)
        print("-" * 50)

if __name__ == "__main__":
    main()
