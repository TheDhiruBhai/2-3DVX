"""
Visualizer for 2D Sketches and Reconstructed 3D Voxel Models
Generates comparison plots showing:
1. Input 2D Sketch (64x64)
2. Ground Truth 3D Voxels (32x32x32)
3. Model Predicted 3D Voxels (32x32x32)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(dataset_path, model_path=None, num_samples=3, out_png="Tools/DataGen/samples_preview.png"):
    data = np.load(dataset_path)
    inputs = data['inputs']
    targets = data['targets']
    
    total = len(inputs)
    indices = np.random.choice(total, min(num_samples, total), replace=False)

    model = None
    if model_path and os.path.exists(model_path):
        import torch
        from train_model import Draw2VoxelNet
        model = Draw2VoxelNet()
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt)
        model.eval()

    cols = 3 if model else 2
    fig = plt.figure(figsize=(cols * 4, num_samples * 4))

    for i, idx in enumerate(indices):
        sketch = inputs[idx, 0] # (64, 64)
        gt_vox = targets[idx, 0] # (32, 32, 32)

        # 1. Plot 2D Sketch
        ax1 = fig.add_subplot(num_samples, cols, i * cols + 1)
        ax1.imshow(sketch, cmap='gray_r', origin='lower')
        ax1.set_title(f"Sample #{idx}: 2D Sketch (64x64)")
        ax1.axis('off')

        # 2. Plot Ground Truth 3D Voxels
        ax2 = fig.add_subplot(num_samples, cols, i * cols + 2, projection='3d')
        ax2.voxels(gt_vox > 0.5, edgecolor='k', alpha=0.8, facecolors='cyan')
        ax2.set_title("Ground Truth 3D Voxels (32x32x32)")
        ax2.view_init(elev=20, azim=45)

        # 3. Plot Predicted Voxels (if model available)
        if model:
            import torch
            with torch.no_grad():
                tensor_in = torch.from_numpy(sketch[None, None, ...]).float()
                pred_vox = model(tensor_in)[0, 0].numpy()
            
            ax3 = fig.add_subplot(num_samples, cols, i * cols + 3, projection='3d')
            ax3.voxels(pred_vox > 0.5, edgecolor='k', alpha=0.8, facecolors='crimson')
            ax3.set_title(f"Predicted Voxels (>0.5)")
            ax3.view_init(elev=20, azim=45)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[Visualizer] Saved preview image to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Tools/DataGen/dataset_voxel.npz")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out", type=str, default="Tools/DataGen/samples_preview.png")
    args = parser.parse_args()

    visualize_samples(args.data, args.model, args.samples, args.out)
