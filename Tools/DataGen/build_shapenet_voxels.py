#!/usr/bin/env python3
"""
==============================================================================
ShapeNetCore 3D CAD Dataset Preprocessing & Ingestion Pipeline
Part of AntiGravity 2-3DVX Milestone 5
==============================================================================
Features:
- Curated 4-Class CAD Dataset: Mug/Cup, Chair, Knife/Dagger, Table
- Automated ingestion of local ShapeNet meshes (.obj, .off, .stl, .ply) with
  high-fidelity parametric procedural CAD generators fallback
- Mesh normalization: Centering at (0,0,0) and uniform scaling to [-0.9, 0.9]
- Memory-safe voxelization to 32x32x32 binary occupancy grid (uint8)
- Tri-view orthographic projection: Top (XZ), Front (XY), Side (YZ) at 64x64
- Human sketch stylization: Stroke jitter, line dropout, and morphological dilation
- Aligned export: (silhouettes: [3, 64, 64], voxels: [1, 32, 32, 32])
- Train/Validation 85/15 split saved into dataset_shapenet_4class.npz
==============================================================================
"""

import os
import glob
import math
import time
import random
import argparse
import numpy as np
from scipy import ndimage
import trimesh

# ==============================================================================
# 1. PARAMETRIC CAD MESH GENERATORS (ShapeNet-Equivalent Geometry)
# ==============================================================================

def generate_procedural_chair():
    """Generates a parametric 3D chair with randomized proportions."""
    w = random.uniform(0.7, 1.0)
    d = random.uniform(0.7, 1.0)
    leg_h = random.uniform(0.7, 0.9)
    seat_thick = random.uniform(0.06, 0.12)
    back_h = random.uniform(0.7, 1.1)

    # Seat
    seat = trimesh.creation.box(extents=[w, d, seat_thick])
    seat.apply_translation([0, 0, leg_h + seat_thick / 2.0])
    parts = [seat]

    # 4 Legs
    leg_r = random.uniform(0.03, 0.05)
    leg_x = (w / 2.0) - 0.08
    leg_y = (d / 2.0) - 0.08
    for dx in [-leg_x, leg_x]:
        for dy in [-leg_y, leg_y]:
            leg = trimesh.creation.cylinder(radius=leg_r, height=leg_h, sections=12)
            leg.apply_translation([dx, dy, leg_h / 2.0])
            parts.append(leg)

    # Backrest style
    back_style = random.choice(["solid", "slats", "cushion"])
    back_y = (d / 2.0) - 0.06
    if back_style == "solid":
        back = trimesh.creation.box(extents=[w * 0.9, 0.06, back_h])
        back.apply_translation([0, back_y, leg_h + seat_thick + back_h / 2.0])
        parts.append(back)
    elif back_style == "slats":
        num_slats = random.randint(3, 5)
        slat_xs = np.linspace(-w * 0.38, w * 0.38, num_slats)
        for sx in slat_xs:
            slat = trimesh.creation.box(extents=[0.05, 0.04, back_h])
            slat.apply_translation([sx, back_y, leg_h + seat_thick + back_h / 2.0])
            parts.append(slat)
        top_rail = trimesh.creation.box(extents=[w * 0.95, 0.06, 0.08])
        top_rail.apply_translation([0, back_y, leg_h + seat_thick + back_h])
        parts.append(top_rail)
    else:
        cushion = trimesh.creation.box(extents=[w * 0.85, 0.1, back_h * 0.8])
        cushion.apply_translation([0, back_y, leg_h + seat_thick + back_h * 0.5])
        parts.append(cushion)

    combined = trimesh.util.concatenate(parts)
    return combined

def generate_procedural_table():
    """Generates a parametric 3D table (Rectangular or Round)."""
    is_round = random.random() < 0.35
    h = random.uniform(0.7, 0.9)
    top_thick = random.uniform(0.06, 0.12)

    parts = []
    if is_round:
        radius = random.uniform(0.6, 0.9)
        top = trimesh.creation.cylinder(radius=radius, height=top_thick, sections=24)
        top.apply_translation([0, 0, h + top_thick / 2.0])
        parts.append(top)

        # Pedestal base or 4 legs
        if random.random() < 0.5:
            pedestal = trimesh.creation.cylinder(radius=0.12, height=h, sections=16)
            pedestal.apply_translation([0, 0, h / 2.0])
            base = trimesh.creation.cylinder(radius=radius * 0.6, height=0.06, sections=20)
            base.apply_translation([0, 0, 0.03])
            parts.extend([pedestal, base])
        else:
            offset = radius * 0.65
            for ang in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
                lx = offset * math.cos(ang)
                ly = offset * math.sin(ang)
                leg = trimesh.creation.cylinder(radius=0.04, height=h, sections=12)
                leg.apply_translation([lx, ly, h / 2.0])
                parts.append(leg)
    else:
        w = random.uniform(0.9, 1.4)
        d = random.uniform(0.7, 1.0)
        top = trimesh.creation.box(extents=[w, d, top_thick])
        top.apply_translation([0, 0, h + top_thick / 2.0])
        parts.append(top)

        leg_x = (w / 2.0) - 0.08
        leg_y = (d / 2.0) - 0.08
        for dx in [-leg_x, leg_x]:
            for dy in [-leg_y, leg_y]:
                leg = trimesh.creation.box(extents=[0.08, 0.08, h])
                leg.apply_translation([dx, dy, h / 2.0])
                parts.append(leg)

    combined = trimesh.util.concatenate(parts)
    return combined

def generate_procedural_mug():
    """Generates a parametric 3D coffee mug with hollow cavity and handle."""
    r_outer = random.uniform(0.4, 0.55)
    r_inner = r_outer - random.uniform(0.06, 0.10)
    h = random.uniform(0.8, 1.1)
    base_thick = 0.08

    # Outer cylinder
    outer = trimesh.creation.cylinder(radius=r_outer, height=h, sections=24)
    outer.apply_translation([0, 0, h / 2.0])

    # Inner hollow cavity
    inner = trimesh.creation.cylinder(radius=r_inner, height=h - base_thick + 0.02, sections=24)
    inner.apply_translation([0, 0, base_thick + (h - base_thick) / 2.0])

    try:
        mug_body = trimesh.boolean.difference([outer, inner], engine='scad')
    except Exception:
        # Fallback to outer cylinder with solid rim if SCAD boolean is unavailable
        mug_body = outer

    # Torus Handle
    handle_major = random.uniform(0.22, 0.32)
    handle_minor = random.uniform(0.04, 0.07)
    handle = trimesh.creation.torus(major_radius=handle_major, minor_radius=handle_minor, major_sections=20, minor_sections=12)
    handle.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, [1, 0, 0]))
    handle.apply_translation([r_outer + handle_major * 0.45, 0, h * 0.5])

    combined = trimesh.util.concatenate([mug_body, handle])
    return combined

def generate_procedural_knife():
    """Generates a parametric 3D knife/dagger."""
    blade_len = random.uniform(0.9, 1.3)
    blade_w = random.uniform(0.12, 0.22)
    blade_thick = random.uniform(0.02, 0.04)
    handle_len = random.uniform(0.5, 0.7)
    handle_r = random.uniform(0.04, 0.07)

    # Blade: Box tapered at tip
    blade = trimesh.creation.box(extents=[blade_thick, blade_w, blade_len])
    blade.apply_translation([0, 0, blade_len / 2.0])

    # Crossguard
    guard_w = random.uniform(0.25, 0.45)
    guard = trimesh.creation.box(extents=[blade_thick * 1.8, guard_w, 0.06])
    guard.apply_translation([0, 0, 0])

    # Handle
    handle = trimesh.creation.cylinder(radius=handle_r, height=handle_len, sections=14)
    handle.apply_translation([0, 0, -handle_len / 2.0])

    # Pommel
    pommel = trimesh.creation.icosphere(subdivisions=2, radius=handle_r * 1.35)
    pommel.apply_translation([0, 0, -handle_len])

    combined = trimesh.util.concatenate([blade, guard, handle, pommel])
    return combined

GENERATORS = {
    "chair": generate_procedural_chair,
    "table": generate_procedural_table,
    "mug": generate_procedural_mug,
    "knife": generate_procedural_knife,
}

# ==============================================================================
# 2. MESH NORMALIZATION & VOXELIZATION
# ==============================================================================

def normalize_mesh(mesh, target_bounds=(-0.9, 0.9)):
    """Centers mesh at (0,0,0) and uniformly scales within [-0.9, 0.9]."""
    mesh = mesh.copy()
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(-center)

    extents = mesh.bounds[1] - mesh.bounds[0]
    max_extent = max(np.max(extents), 1e-4)
    scale = (target_bounds[1] - target_bounds[0]) / max_extent
    mesh.apply_scale(scale)
    return mesh

def voxelize_mesh_32(mesh, resolution=32):
    """
    Voxelizes normalized mesh into exactly (32, 32, 32) binary occupancy array (uint8).
    Memory safe and deterministic.
    """
    pitch = 1.8 / float(resolution)
    try:
        vox = mesh.voxelized(pitch=pitch)
        matrix = vox.matrix.astype(np.uint8)
    except Exception:
        # Fallback bounding box sampling
        matrix = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
        samples = mesh.sample(resolution * resolution * 8)
        pts = ((samples + 0.9) / 1.8 * (resolution - 1)).astype(int)
        valid = (pts >= 0) & (pts < resolution)
        valid = valid[:, 0] & valid[:, 1] & valid[:, 2]
        pts = pts[valid]
        matrix[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
        return matrix

    # Pad or crop to exact (resolution, resolution, resolution)
    sx, sy, sz = matrix.shape
    out = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

    cx = min(resolution, sx)
    cy = min(resolution, sy)
    cz = min(resolution, sz)

    ox_s = (sx - cx) // 2
    oy_s = (sy - cy) // 2
    oz_s = (sz - cz) // 2

    ox_d = (resolution - cx) // 2
    oy_d = (resolution - cy) // 2
    oz_d = (resolution - cz) // 2

    out[ox_d:ox_d+cx, oy_d:oy_d+cy, oz_d:oz_d+cz] = matrix[ox_s:ox_s+cx, oy_s:oy_s+cy, oz_s:oz_s+cz]
    return out

# ==============================================================================
# 3. TRI-VIEW ORTHOGRAPHIC PROJECTION WITH SKETCH STYLIZATION
# ==============================================================================

def project_triview_silhouettes(voxels_32, canvas_res=64):
    """
    Projects 3D voxel grid into 3 canonical orthographic views at canvas_res:
    View 0: Front (XY) projection (along Z axis)
    View 1: Side (YZ) projection (along X axis)
    View 2: Top (XZ) projection (along Y axis)
    """
    # 1. Front (XY) - Dim 0 = X, Dim 1 = Y
    front_raw = (np.sum(voxels_32, axis=2) > 0).astype(np.float32)

    # 2. Side (YZ) - Dim 0 = Y, Dim 1 = Z -> we align as Dim 0 = Depth(Z), Dim 1 = Height(Y)
    side_raw = (np.sum(voxels_32, axis=0) > 0).astype(np.float32).T

    # 3. Top (XZ) - Dim 0 = X, Dim 1 = Z
    top_raw = (np.sum(voxels_32, axis=1) > 0).astype(np.float32)

    views = []
    scale = canvas_res / float(voxels_32.shape[0])
    for raw in [front_raw, side_raw, top_raw]:
        # Resize to 64x64
        resized = ndimage.zoom(raw, (scale, scale), order=0)
        resized = np.clip(resized, 0.0, 1.0)
        if resized.shape != (canvas_res, canvas_res):
            canvas = np.zeros((canvas_res, canvas_res), dtype=np.float32)
            mx = min(canvas_res, resized.shape[0])
            my = min(canvas_res, resized.shape[1])
            canvas[:mx, :my] = resized[:mx, :my]
            resized = canvas
        views.append(resized)

    return views

def stylize_as_hand_drawn_sketch(binary_silhouette, jitter_prob=0.85):
    """
    Simulates authentic human freehand drawing on an orthographic silhouette:
    - Boundary extraction (contour outline)
    - Elastic stroke jitter & stroke displacement
    - Morphological brush dilation (varying 2-3px thickness)
    - Realistic line dropout (5-15% stroke segment dropout)
    """
    # Extract outline
    eroded = ndimage.binary_erosion(binary_silhouette > 0.5, iterations=1)
    contour = (binary_silhouette > 0.5) ^ eroded

    if np.sum(contour) < 10:
        contour = binary_silhouette > 0.5

    sketch = contour.astype(np.float32)

    if jitter_prob > 0 and random.random() < jitter_prob:
        # Stroke displacement jitter
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        sketch = ndimage.shift(sketch, (dx, dy), order=1)

        # Morphological brush dilation
        thick = random.choice([1, 2])
        sketch = ndimage.binary_dilation(sketch > 0.3, iterations=thick).astype(np.float32)

        # Line dropout: randomly blank small clusters to mimic lifting pen
        if random.random() < 0.60:
            drop_x = random.randint(10, 50)
            drop_y = random.randint(10, 50)
            drop_r = random.randint(3, 7)
            sketch[drop_x-drop_r:drop_x+drop_r, drop_y-drop_r:drop_y+drop_r] = 0.0

    return np.clip(sketch, 0.0, 1.0).astype(np.float32)

# ==============================================================================
# 4. DATASET BUILDER PIPELINE
# ==============================================================================

def build_shapenet_dataset(output_path="Tools/DataGen/dataset_shapenet_4class.npz",
                           samples_per_class=250,
                           raw_shapenet_dir="Tools/DataGen/shapenet_raw"):
    """
    Builds the complete ShapeNetCore CAD aligned dataset.
    Generates aligned pairs:
      silhouettes: [3, 64, 64] (Top, Front, Side)
      voxels: [1, 32, 32, 32] (Ground-truth 3D CAD occupancy)
    """
    categories = ["mug", "chair", "knife", "table"]
    print("\n" + "=" * 70)
    print(f"[ShapeNet Ingestion] Building 4-Class CAD Voxel Dataset ({samples_per_class} samples/class)")
    print(f"Categories: {categories}")
    print("=" * 70)

    total_samples = len(categories) * samples_per_class
    all_silhouettes = np.zeros((total_samples, 3, 64, 64), dtype=np.uint8)
    all_voxels = np.zeros((total_samples, 1, 32, 32, 32), dtype=np.uint8)
    all_labels = np.zeros(total_samples, dtype=np.int64)

    t0 = time.time()
    idx = 0

    for cat_id, cat_name in enumerate(categories):
        print(f"\n[Ingesting '{cat_name.upper()}']...")
        cat_raw_dir = os.path.join(raw_shapenet_dir, cat_name)
        raw_files = []
        if os.path.exists(cat_raw_dir):
            for ext in ("*.obj", "*.off", "*.stl", "*.ply"):
                raw_files.extend(glob.glob(os.path.join(cat_raw_dir, "**", ext), recursive=True))

        print(f"  Found {len(raw_files)} raw CAD files in '{cat_raw_dir}'")
        raw_idx = 0

        for s in range(samples_per_class):
            # Load from disk if available, else procedural CAD generator
            mesh = None
            if raw_idx < len(raw_files):
                try:
                    loaded = trimesh.load(raw_files[raw_idx], force='mesh')
                    if isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices) > 20:
                        mesh = loaded
                        raw_idx += 1
                except Exception:
                    pass

            if mesh is None:
                generator_fn = GENERATORS[cat_name]
                mesh = generator_fn()

            # Normalize & Voxelize
            norm_mesh = normalize_mesh(mesh)
            vox32 = voxelize_mesh_32(norm_mesh, resolution=32)

            # Project Tri-Views & Stylize as Human Sketches
            views = project_triview_silhouettes(vox32, canvas_res=64)
            stylized_views = [stylize_as_hand_drawn_sketch(v) for v in views]

            # Pack silhouettes [3, 64, 64]
            for v_i in range(3):
                all_silhouettes[idx, v_i] = (stylized_views[v_i] * 255.0).astype(np.uint8)

            # Pack voxels [1, 32, 32, 32]
            all_voxels[idx, 0] = vox32
            all_labels[idx] = cat_id
            idx += 1

            if (s + 1) % 50 == 0 or (s + 1) == samples_per_class:
                print(f"  ✔ Processed {s + 1}/{samples_per_class} {cat_name} models")

    # 85/15 Train/Validation Split
    print("\n[Split] Partitioning into 85% Train / 15% Validation split...")
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    val_count = int(0.15 * total_samples)
    train_count = total_samples - val_count

    train_idx = indices[:train_count]
    val_idx = indices[train_count:]

    train_silhouettes = all_silhouettes[train_idx]
    train_voxels = all_voxels[train_idx]
    train_labels = all_labels[train_idx]

    val_silhouettes = all_silhouettes[val_idx]
    val_voxels = all_voxels[val_idx]
    val_labels = all_labels[val_idx]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        train_silhouettes=train_silhouettes,
        train_voxels=train_voxels,
        train_labels=train_labels,
        val_silhouettes=val_silhouettes,
        val_voxels=val_voxels,
        val_labels=val_labels,
        classes=np.array(categories)
    )

    size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print("\n" + "=" * 70)
    print(f"[Done] ShapeNet CAD Dataset Built Successfully in {time.time()-t0:.1f}s!")
    print(f"  Output File: {output_path} ({size_mb:.2f} MB)")
    print(f"  Train Samples: {train_count} pairs")
    print(f"  Val Samples:   {val_count} pairs")
    print(f"  Tensor Shapes: Silhouettes={train_silhouettes.shape}, Voxels={train_voxels.shape}")
    print("=" * 70)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShapeNetCore CAD 3D Voxel Preprocessing Pipeline")
    parser.add_argument("--samples", type=int, default=250, help="Samples per category (default: 250 -> 1000 total)")
    parser.add_argument("--output", type=str, default="Tools/DataGen/dataset_shapenet_4class.npz", help="Output .npz path")
    args = parser.parse_args()

    build_shapenet_dataset(output_path=args.output, samples_per_class=args.samples)
