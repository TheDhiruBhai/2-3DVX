"""
Blender Headless Synthetic Data Generator for 2D-to-3D Voxel Reconstruction
Run with:
blender -b -P Tools/DataGen/generate_dataset_blender.py -- --samples 1000 --out Tools/DataGen/blender_voxel_dataset.npz
"""

import sys
import os
import math
import random
import argparse
import numpy as np

try:
    import bpy
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    IS_BLENDER = True
except ImportError:
    IS_BLENDER = False

VOXEL_DIM = 32
IMAGE_DIM = 64

def clean_scene():
    if not IS_BLENDER:
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)

def create_procedural_sword():
    """Builds a 3D sword mesh in Blender."""
    clean_scene()
    
    # 1. Handle (Cylinder)
    handle_len = random.uniform(1.2, 1.8)
    handle_r = random.uniform(0.12, 0.18)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=handle_r,
        depth=handle_len,
        location=(0, 0, handle_len / 2.0)
    )
    handle_obj = bpy.context.active_object
    
    # 2. Pommel (Sphere)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=handle_r * 1.5,
        location=(0, 0, 0)
    )
    pommel_obj = bpy.context.active_object
    
    # 3. Crossguard (Cube)
    guard_y = handle_len
    guard_w = random.uniform(1.0, 2.2)
    guard_h = 0.2
    guard_d = 0.25
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0, 0, guard_y + guard_h / 2.0)
    )
    guard_obj = bpy.context.active_object
    guard_obj.scale = (guard_w, guard_d, guard_h)
    bpy.ops.object.transform_apply(scale=True)
    
    # 4. Blade
    blade_len = random.uniform(3.5, 5.0)
    blade_w = random.uniform(0.4, 0.7)
    blade_thick = 0.1
    blade_z_start = guard_y + guard_h
    
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(0, 0, blade_z_start + blade_len / 2.0)
    )
    blade_obj = bpy.context.active_object
    blade_obj.scale = (blade_w, blade_thick, blade_len)
    bpy.ops.object.transform_apply(scale=True)
    
    # Join all into one object
    for obj in [handle_obj, pommel_obj, guard_obj, blade_obj]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = blade_obj
    bpy.ops.object.join()
    
    joined = bpy.context.active_object
    joined.name = "WeaponMesh"
    return joined

def voxelize_blender_object(obj, dim=VOXEL_DIM):
    """Computes a 32x32x32 binary voxel grid for the active mesh using ray casting."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    bm.from_object(obj, depsgraph)
    bmesh.ops.transform(bm, matrix=obj.matrix_world, verts=bm.verts)
    
    # Compute bounds
    verts = [v.co for v in bm.verts]
    min_b = Vector([min(v[i] for v in verts) for i in range(3)])
    max_b = Vector([max(v[i] for v in verts) for i in range(3)])
    size = max_b - min_b
    max_side = max(size.x, size.y, size.z)
    
    if max_side < 1e-5:
        bm.free()
        return np.zeros((dim, dim, dim), dtype=np.float32)

    center = (min_b + max_b) * 0.5
    scale_factor = (dim - 4) / max_side
    
    # Center and normalize vertices into [2, dim-2]
    for v in bm.verts:
        v.co = (v.co - center) * scale_factor + Vector([dim/2, dim/2, dim/2])
        
    tree = BVHTree.FromBMesh(bm)
    grid = np.zeros((dim, dim, dim), dtype=np.float32)

    # Cast rays along X axis to identify inside/outside voxels
    for y in range(dim):
        for z in range(dim):
            origin = Vector((-1.0, float(y), float(z)))
            direction = Vector((1.0, 0.0, 0.0))
            
            curr_x = -1.0
            hit_count = 0
            hits = []
            while True:
                hit, normal, index, dist = tree.ray_cast(origin + direction * (curr_x + 1.01), direction)
                if hit is None:
                    break
                curr_x = hit.x
                hits.append(curr_x)
                
            # Fill between odd/even intersections
            for i in range(0, len(hits) - 1, 2):
                x_start = int(max(0, math.floor(hits[i])))
                x_end = int(min(dim - 1, math.ceil(hits[i+1])))
                grid[x_start:x_end+1, y, z] = 1.0
                
    bm.free()
    return grid

def main():
    if not IS_BLENDER:
        print("[BlenderDataGen] Error: This script must be executed via Blender: blender -b -P generate_dataset_blender.py")
        sys.exit(1)

    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        args_list = argv[argv.index("--") + 1:]
    else:
        args_list = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", type=str, default="Tools/DataGen/blender_voxel_dataset.npz")
    args = parser.parse_args(args_list)

    print(f"[BlenderDataGen] Generating {args.samples} samples in headless Blender...")
    inputs = np.zeros((args.samples, 1, IMAGE_DIM, IMAGE_DIM), dtype=np.float32)
    targets = np.zeros((args.samples, 1, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), dtype=np.float32)

    for i in range(args.samples):
        obj = create_procedural_sword()
        grid = voxelize_blender_object(obj, dim=VOXEL_DIM)
        
        # 2D projection
        proj = np.max(grid, axis=1) # Shape (32, 32)
        # Resample to 64x64 via 2x nearest-neighbor upsampling (pure numpy)
        sketch = np.repeat(np.repeat(proj, 2, axis=0), 2, axis=1)
        sketch = (sketch > 0.3).astype(np.float32)
        
        inputs[i, 0] = sketch
        targets[i, 0] = grid
        
        if (i + 1) % 50 == 0:
            print(f"  Blender processed {i + 1}/{args.samples} samples...")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, inputs=inputs, targets=targets)
    print(f"[BlenderDataGen] Saved Blender dataset to {args.out}")

if __name__ == "__main__":
    main()
