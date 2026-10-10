import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

from generate_dataset_triview import (
    TARGET_ASSET_GENERATORS,
    project_canonical_view,
    elastic_transform,
    IMAGE_DIM,
    VOXEL_DIM
)

def render_multimodal_sketch_demo(clean_proj, mode="outline"):
    binary_mask = (clean_proj > 0.4).astype(np.uint8)

    if mode == "outline":
        stroke_px = random.choice([1, 2, 3])
        k = 2 if stroke_px == 1 else (3 if stroke_px == 2 else 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        stroke = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel).astype(np.float32)
    elif mode == "silhouette":
        stroke = binary_mask.astype(np.float32)
    elif mode == "partial":
        stroke_px = random.choice([1, 2, 3])
        k = 2 if stroke_px == 1 else (3 if stroke_px == 2 else 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        stroke = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel).astype(np.float32)
        # Drop out 1-2 chunks
        for _ in range(random.randint(1, 2)):
            rx = random.randint(12, 52)
            ry = random.randint(12, 52)
            rw = random.randint(6, 12)
            stroke[max(0, rx-rw//2):min(IMAGE_DIM, rx+rw//2), max(0, ry-rw//2):min(IMAGE_DIM, ry+rw//2)] = 0.0

    # Tremor
    stroke = elastic_transform(stroke, alpha=random.uniform(4.0, 6.5), sigma=random.uniform(2.0, 2.8))
    # Affine
    angle = random.uniform(-6.0, 6.0)
    stroke = ndimage.rotate(stroke, angle, reshape=False, order=1, mode='constant', cval=0.0)
    shift_x = random.uniform(-1.5, 1.5)
    shift_y = random.uniform(-1.5, 1.5)
    stroke = ndimage.shift(stroke, (shift_x, shift_y), order=1, mode='constant', cval=0.0)

    stroke = np.clip(stroke, 0.0, 1.0)
    return (stroke > 0.25).astype(np.float32)

def generate_preview():
    class_names = [
        "Sword", "Shield", "Blaster", "Axe", "TeddyBear",
        "RubberDuck", "ToyRobot", "ToyBunny", "Gamepad", "RetroTV", "GameBoy"
    ]
    num_classes = len(TARGET_ASSET_GENERATORS)

    fig, axes = plt.subplots(num_classes, 4, figsize=(12, 2.2 * num_classes))
    plt.suptitle("AntiGravity Draw2Voxel: Multi-Modal Training Data (Outlines vs Silhouettes vs Partial)", fontsize=14, y=0.995)

    col_titles = [
        "Mode 1: Hollow Outline (70%)\n[Morph Gradient + Tremor]",
        "Mode 2: Solid Silhouette (20%)\n[Binary Preset + Jitter]",
        "Mode 3: Partial Outline (10%)\n[Unclosed Loops / Dropouts]",
        "Target 3D Voxel Ground Truth\n[100% Solid Occupancy + RGB]"
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, pad=10)

    for i in range(num_classes):
        gen_fn, cat_id = TARGET_ASSET_GENERATORS[i]
        voxels_4ch = gen_fn()
        view_id = 0 # Front View
        clean_proj = project_canonical_view(voxels_4ch, view_id)

        outline_sketch = render_multimodal_sketch_demo(clean_proj, mode="outline")
        silhouette_sketch = render_multimodal_sketch_demo(clean_proj, mode="silhouette")
        partial_sketch = render_multimodal_sketch_demo(clean_proj, mode="partial")

        # Visualizing 3D voxels: front color projection
        occ = voxels_4ch[0]
        rgb = voxels_4ch[1:4] # (3, 64, 64, 64)
        # Depth argmax for front projection
        z_indices = np.argmax(occ, axis=2) # (64, 64)
        has_occ = np.max(occ, axis=2) > 0.5 # (64, 64)

        rgb_proj = np.zeros((64, 64, 3), dtype=np.float32)
        for x in range(64):
            for y in range(64):
                if has_occ[x, y]:
                    z = z_indices[x, y]
                    rgb_proj[x, y] = rgb[:, x, y, z]
                else:
                    rgb_proj[x, y] = [0.12, 0.12, 0.15] # Dark background

        # Plot Outline
        axes[i, 0].imshow(outline_sketch, cmap='gray_r', interpolation='nearest')
        axes[i, 0].set_ylabel(f"{class_names[i]}\n(ID: {cat_id})", fontsize=10, rotation=0, labelpad=40, va='center')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Plot Silhouette
        axes[i, 1].imshow(silhouette_sketch, cmap='gray_r', interpolation='nearest')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Plot Partial
        axes[i, 2].imshow(partial_sketch, cmap='gray_r', interpolation='nearest')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Plot Ground Truth 3D
        axes[i, 3].imshow(np.transpose(rgb_proj, (1, 0, 2)), origin='lower', interpolation='nearest')
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])

    plt.tight_layout()
    out_artifact = "/home/tuf/.gemini/antigravity-cli/brain/ee7eaca9-153b-4a22-a7d0-06103c846a27/outline_vs_silhouette_preview.png"
    plt.savefig(out_artifact, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[Preview] Successfully generated preview image at: {out_artifact}")

if __name__ == "__main__":
    generate_preview()
