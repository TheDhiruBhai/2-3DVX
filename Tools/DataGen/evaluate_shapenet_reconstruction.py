#!/usr/bin/env python3
"""
==============================================================================
ShapeNetCore 3D Voxel Benchmark & Validation Metric Evaluator
Part of AntiGravity 2-3DVX Milestone 5
==============================================================================
Metrics Computed:
1. Volumetric Intersection over Union (IoU)
2. 3D Surface Chamfer Distance (using cKDTree)
3. Precision, Recall, and F1-score
4. Per-category and Macro-averaged benchmarks for Academic Report
==============================================================================
"""

import os
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

def compute_volumetric_iou(voxel_pred, voxel_gt):
    """
    Computes Volumetric IoU between two binary 3D grids:
    IoU = |Pred ∩ GT| / |Pred ∪ GT|
    """
    p_bin = voxel_pred > 0.5
    g_bin = voxel_gt > 0.5
    intersection = np.logical_and(p_bin, g_bin).sum()
    union = np.logical_or(p_bin, g_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)

def extract_surface_voxels(voxel_grid):
    """Extracts boundary surface voxels (where voxel is 1 and has at least one 0 neighbor)."""
    grid = voxel_grid > 0.5
    if grid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    eroded = ndimage.binary_erosion(grid, iterations=1)
    boundary = grid ^ eroded
    pts = np.argwhere(boundary).astype(np.float32)
    return pts

def compute_chamfer_distance(voxel_pred, voxel_gt):
    """
    Computes bidirectional 3D Chamfer Distance between surface voxel point clouds:
    CD = (1/|S_p|) * sum(min ||p - g||) + (1/|S_g|) * sum(min ||g - p||)
    """
    pts_pred = extract_surface_voxels(voxel_pred)
    pts_gt = extract_surface_voxels(voxel_gt)

    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return float('inf')

    tree_gt = cKDTree(pts_gt)
    dist_p2g, _ = tree_gt.query(pts_pred)

    tree_pred = cKDTree(pts_pred)
    dist_g2p, _ = tree_pred.query(pts_gt)

    chamfer = float(np.mean(dist_p2g) + np.mean(dist_g2p)) / 2.0
    return chamfer

def compute_classification_metrics(voxel_pred, voxel_gt):
    """Computes Precision, Recall, and F1-score."""
    p_bin = voxel_pred > 0.5
    g_bin = voxel_gt > 0.5

    tp = float(np.logical_and(p_bin, g_bin).sum())
    fp = float(np.logical_and(p_bin, ~g_bin).sum())
    fn = float(np.logical_and(~p_bin, g_bin).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return prec, rec, f1

def evaluate_shapenet_benchmark(dataset_path="Tools/DataGen/dataset_shapenet_4class.npz"):
    """Evaluates validation metrics across all 4 ShapeNet categories."""
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset file not found at: {dataset_path}")
        print("Please run 'python Tools/DataGen/build_shapenet_voxels.py' first.")
        return

    data = np.load(dataset_path)
    val_voxels = data['val_voxels'][:, 0] # (N, 32, 32, 32)
    val_labels = data['val_labels']
    classes = data['classes']

    num_samples = len(val_labels)
    print("\n" + "=" * 78)
    print("ShapeNetCore 3D Voxel Ground-Truth Benchmark Evaluation")
    print(f"Validation Samples: {num_samples} | Categories: {list(classes)}")
    print("=" * 78)

    per_class_ious = {c: [] for c in classes}
    per_class_chamfers = {c: [] for c in classes}
    per_class_f1s = {c: [] for c in classes}

    # Evaluate 3D Volumetric Infilling reconstruction (curvaturized inflation from Front XY view)
    # matching the real-time pipeline in UniversalSandboxEngine / SentisMultiTaskEngine
    for i in range(num_samples):
        gt = val_voxels[i]
        c_name = str(classes[val_labels[i]])

        # Simulate 2.5D visual hull / rounded inflation from front orthographic silhouette
        front_view = (np.sum(gt, axis=2) > 0).astype(np.float32)

        # Morphological inflation & depth profile matching UniversalSandboxEngine
        dist = ndimage.distance_transform_edt(front_view > 0.5)
        max_d = max(np.max(dist), 1.0)
        norm_d = dist / max_d

        pred_vox = np.zeros_like(gt)
        z_center = gt.shape[2] // 2

        for x in range(gt.shape[0]):
            for y in range(gt.shape[1]):
                if front_view[x, y] > 0.5:
                    half_w = max(1, int(round(norm_d[x, y] * 6)))
                    z0 = max(0, z_center - half_w)
                    z1 = min(gt.shape[2], z_center + half_w + 1)
                    pred_vox[x, y, z0:z1] = 1

        iou = compute_volumetric_iou(pred_vox, gt)
        cd = compute_chamfer_distance(pred_vox, gt)
        _, _, f1 = compute_classification_metrics(pred_vox, gt)

        per_class_ious[c_name].append(iou)
        if not np.isinf(cd):
            per_class_chamfers[c_name].append(cd)
        per_class_f1s[c_name].append(f1)

    print(f"{'Category':<16} | {'Volumetric IoU':<16} | {'Chamfer Dist (px)':<18} | {'Voxel F1-Score':<14}")
    print("-" * 78)

    macro_iou = []
    macro_cd = []
    macro_f1 = []

    for c in classes:
        mean_iou = np.mean(per_class_ious[c])
        mean_cd = np.mean(per_class_chamfers[c])
        mean_f1 = np.mean(per_class_f1s[c])

        macro_iou.append(mean_iou)
        macro_cd.append(mean_cd)
        macro_f1.append(mean_f1)

        print(f"{c.capitalize():<16} | {mean_iou*100:6.2f}%          | {mean_cd:6.3f} voxels      | {mean_f1*100:6.2f}%")

    print("-" * 78)
    print(f"{'MACRO AVERAGE':<16} | {np.mean(macro_iou)*100:6.2f}%          | {np.mean(macro_cd):6.3f} voxels      | {np.mean(macro_f1)*100:6.2f}%")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Tools/DataGen/dataset_shapenet_4class.npz")
    args = parser.parse_args()
    evaluate_shapenet_benchmark(args.dataset)
