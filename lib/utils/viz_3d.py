import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def plot_trajectory_3d(traj_dict, output_path="trajectory_3d.png"):
    """
    Plot 3D Trajectories for multiple objects.
    
    Args:
        traj_dict (dict): {obj_id: np.array [T, 3]} 
        output_path (str): Path to save plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(traj_dict)))
    
    for i, (obj_id, traj_coords) in enumerate(traj_dict.items()):
        xs = traj_coords[:, 0]
        ys = traj_coords[:, 1]
        zs = traj_coords[:, 2]
        
        color = colors[i]
        
        ax.plot(xs, ys, zs, label=f'Obj {obj_id}', color=color, marker='o', markersize=3)
        
        # Start/End
        ax.scatter(xs[0], ys[0], zs[0], color=color, marker='^', s=50) # Start
        ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker='x', s=50) # End
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title('3D Multi-Object Trajectory Visualization')
    ax.legend()
    
    # Invert Z axis usually for Camera coordinates (Z forward/down)
    ax.invert_zaxis()
    
    plt.savefig(output_path)
    plt.close()
    print(f"3D Trajectory plot saved to {output_path}")
