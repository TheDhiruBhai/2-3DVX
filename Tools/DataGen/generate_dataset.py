"""
Procedural 2D-to-3D Voxel Multi-Category Dataset Generator
Generates paired samples:
- Inputs:  (N, 1, 64, 64) float32 2D raster sketch images (with human drawing jitter, dilation, gaps)
- Targets: (N, 1, 32, 32, 32) float32 3D voxel occupancy grids (0.0 or 1.0)
- Categories: (N,) int32 category IDs (0: Weapons, 1: Toys, 2: Electronics)

Categories & Generators:
1. WEAPONS (Cat 0):
   - Melee: Swords, Daggers, Battleaxes, Warhammers, Bats/Clubs, Shields
   - Firearms: Handguns/Pistols, Rifles, Sci-Fi Blasters
2. TOYS (Cat 1):
   - Living beings stylized as toys: Teddy Bears, Rubber Ducks, Toy Robots, Toy Bunnies, Toy Karts/Cars
3. ELECTRONICS & GADGETS (Cat 2):
   - Game Controllers (Gamepads), Retro Handhelds (Game Boy style), Vintage CRT TVs, Retro Cameras
"""

import os
import sys
import math
import random
import argparse
import numpy as np
from scipy import ndimage

VOXEL_DIM = 32
IMAGE_DIM = 64

def create_empty_voxel():
    return np.zeros((VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), dtype=np.float32)

# ==============================================================================
# GEOMETRIC CSG PRIMITIVES
# ==============================================================================

def add_cylinder(grid, p0, p1, radius):
    """Draws a solid cylinder in voxel space from p0 to p1 with given radius."""
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    v = p1 - p0
    length_sq = np.dot(v, v)
    if length_sq < 1e-6:
        return grid

    min_pt = np.clip(np.floor(np.minimum(p0, p1) - radius - 1).astype(int), 0, VOXEL_DIM - 1)
    max_pt = np.clip(np.ceil(np.maximum(p0, p1) + radius + 1).astype(int), 0, VOXEL_DIM - 1)

    xs = np.arange(min_pt[0], max_pt[0] + 1)
    ys = np.arange(min_pt[1], max_pt[1] + 1)
    zs = np.arange(min_pt[2], max_pt[2] + 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)

    w = pts - p0
    t = np.clip(np.dot(w, v) / length_sq, 0.0, 1.0)
    proj = p0 + t[..., np.newaxis] * v
    dist_sq = np.sum((pts - proj) ** 2, axis=-1)

    inside = dist_sq <= (radius ** 2)
    grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1] = np.maximum(
        grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1],
        inside.astype(np.float32)
    )
    return grid

def add_box(grid, center, half_extents):
    """Draws an axis-aligned solid box in voxel space."""
    c = np.array(center, dtype=np.float32)
    h = np.array(half_extents, dtype=np.float32)

    min_pt = np.clip(np.floor(c - h).astype(int), 0, VOXEL_DIM - 1)
    max_pt = np.clip(np.ceil(c + h).astype(int), 0, VOXEL_DIM - 1)

    grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1] = 1.0
    return grid

def add_sphere(grid, center, radius):
    """Draws a solid sphere in voxel space."""
    c = np.array(center, dtype=np.float32)
    min_pt = np.clip(np.floor(c - radius).astype(int), 0, VOXEL_DIM - 1)
    max_pt = np.clip(np.ceil(c + radius).astype(int), 0, VOXEL_DIM - 1)

    xs = np.arange(min_pt[0], max_pt[0] + 1)
    ys = np.arange(min_pt[1], max_pt[1] + 1)
    zs = np.arange(min_pt[2], max_pt[2] + 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    dist_sq = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2

    inside = dist_sq <= (radius ** 2)
    grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1] = np.maximum(
        grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1],
        inside.astype(np.float32)
    )
    return grid

def add_ellipsoid(grid, center, rx, ry, rz):
    """Draws a solid 3D ellipsoid in voxel space."""
    c = np.array(center, dtype=np.float32)
    radii = np.array([max(rx, 0.5), max(ry, 0.5), max(rz, 0.5)], dtype=np.float32)

    min_pt = np.clip(np.floor(c - radii).astype(int), 0, VOXEL_DIM - 1)
    max_pt = np.clip(np.ceil(c + radii).astype(int), 0, VOXEL_DIM - 1)

    xs = np.arange(min_pt[0], max_pt[0] + 1)
    ys = np.arange(min_pt[1], max_pt[1] + 1)
    zs = np.arange(min_pt[2], max_pt[2] + 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    dist = ((X - c[0]) / radii[0]) ** 2 + ((Y - c[1]) / radii[1]) ** 2 + ((Z - c[2]) / radii[2]) ** 2

    inside = dist <= 1.0
    grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1] = np.maximum(
        grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1],
        inside.astype(np.float32)
    )
    return grid

def add_cone(grid, p0, p1, r0, r1):
    """Draws a truncated cone tapering from r0 at p0 to r1 at p1."""
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    v = p1 - p0
    length_sq = np.dot(v, v)
    if length_sq < 1e-6:
        return grid

    max_r = max(r0, r1)
    min_pt = np.clip(np.floor(np.minimum(p0, p1) - max_r - 1).astype(int), 0, VOXEL_DIM - 1)
    max_pt = np.clip(np.ceil(np.maximum(p0, p1) + max_r + 1).astype(int), 0, VOXEL_DIM - 1)

    xs = np.arange(min_pt[0], max_pt[0] + 1)
    ys = np.arange(min_pt[1], max_pt[1] + 1)
    zs = np.arange(min_pt[2], max_pt[2] + 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)

    w = pts - p0
    t = np.clip(np.dot(w, v) / length_sq, 0.0, 1.0)
    proj = p0 + t[..., np.newaxis] * v
    dist_sq = np.sum((pts - proj) ** 2, axis=-1)

    cur_r = (1.0 - t) * r0 + t * r1
    inside = dist_sq <= (cur_r ** 2)
    grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1] = np.maximum(
        grid[min_pt[0]:max_pt[0] + 1, min_pt[1]:max_pt[1] + 1, min_pt[2]:max_pt[2] + 1],
        inside.astype(np.float32)
    )
    return grid

def add_tapered_blade(grid, base_center, tip_center, base_w, tip_w, thickness):
    """Draws a tapered blade segment aligned primarily along the Y axis."""
    y_start = int(max(0, math.floor(base_center[1])))
    y_end = int(min(VOXEL_DIM - 1, math.ceil(tip_center[1])))
    if y_end <= y_start:
        return grid

    total_h = tip_center[1] - base_center[1]
    if total_h <= 0:
        return grid

    for y in range(y_start, y_end + 1):
        alpha = (y - base_center[1]) / total_h
        cur_w = (1.0 - alpha) * base_w + alpha * tip_w
        cx = (1.0 - alpha) * base_center[0] + alpha * tip_center[0]
        cz = (1.0 - alpha) * base_center[2] + alpha * tip_center[2]

        x_min = int(max(0, math.floor(cx - cur_w)))
        x_max = int(min(VOXEL_DIM - 1, math.ceil(cx + cur_w)))
        z_min = int(max(0, math.floor(cz - thickness)))
        z_max = int(min(VOXEL_DIM - 1, math.ceil(cz + thickness)))

        grid[x_min:x_max + 1, y, z_min:z_max + 1] = 1.0
    return grid

# ==============================================================================
# CATEGORY 1: WEAPONS (Melee & Firearms)
# ==============================================================================

def generate_sword():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    handle_len = random.uniform(5.0, 7.5)
    handle_rad = random.uniform(0.9, 1.4)
    add_cylinder(grid, (cx, 2.0, cz), (cx, 2.0 + handle_len, cz), handle_rad)
    add_sphere(grid, (cx, 1.8, cz), handle_rad * 1.4) # Pommel

    guard_y = 2.0 + handle_len
    guard_width = random.uniform(5.0, 9.0)
    add_box(grid, (cx, guard_y, cz), (guard_width, 0.9, 1.1))

    blade_start_y = guard_y + 0.8
    blade_end_y = random.uniform(26.0, 30.0)
    blade_base_w = random.uniform(1.8, 3.2)
    blade_tip_w = random.uniform(0.4, 1.2)
    blade_thick = random.uniform(0.7, 1.1)
    add_tapered_blade(grid, (cx, blade_start_y, cz), (cx, blade_end_y, cz), blade_base_w, blade_tip_w, blade_thick)
    return grid

def generate_dagger():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    handle_len = random.uniform(4.0, 6.0)
    add_cylinder(grid, (cx, 4.0, cz), (cx, 4.0 + handle_len, cz), 1.0)
    guard_y = 4.0 + handle_len
    add_box(grid, (cx, guard_y, cz), (random.uniform(3.0, 5.0), 0.8, 1.0))
    add_tapered_blade(grid, (cx, guard_y + 0.7, cz), (cx, random.uniform(19.0, 24.0), cz), 2.0, 0.4, 0.9)
    return grid

def generate_axe():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    shaft_len = random.uniform(23.0, 28.0)
    add_cylinder(grid, (cx, 2.0, cz), (cx, 2.0 + shaft_len, cz), random.uniform(0.9, 1.3))
    head_y = random.uniform(19.0, 24.0)
    # Axe blade
    blade_span = random.uniform(6.0, 10.0)
    blade_cx = cx + blade_span * 0.5
    add_box(grid, (blade_cx, head_y, cz), (blade_span * 0.5, random.uniform(3.0, 5.0), 0.8))
    # Flare
    add_box(grid, (blade_cx + 1.0, head_y, cz), (blade_span * 0.35, random.uniform(4.5, 6.5), 0.7))
    return grid

def generate_hammer():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    shaft_len = random.uniform(22.0, 26.0)
    add_cylinder(grid, (cx, 2.0, cz), (cx, 2.0 + shaft_len, cz), random.uniform(1.0, 1.4))
    head_y = 2.0 + shaft_len - 3.0
    add_box(grid, (cx, head_y, cz), (random.uniform(5.5, 8.5), random.uniform(2.5, 4.0), random.uniform(2.5, 4.0)))
    return grid

def generate_bat():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    add_cone(grid, (cx, 2.5, cz), (cx, 28.0, cz), 0.9, 2.8)
    add_sphere(grid, (cx, 2.3, cz), 1.2)
    return grid

def generate_shield():
    grid = create_empty_voxel()
    cx, cy, cz = 16.0, 16.0, 16.0
    rx = random.uniform(6.0, 9.5)
    ry = random.uniform(8.0, 12.0)
    add_ellipsoid(grid, (cx, cy, cz), rx, ry, 1.8)
    add_sphere(grid, (cx, cy, cz + 1.2), 2.2) # Center boss
    return grid

def generate_pistol():
    grid = create_empty_voxel()
    cz = 16.0
    # Grip handle
    grip_angle = random.uniform(10.0, 18.0)
    grip_rad = math.radians(grip_angle)
    for y in np.linspace(4.0, 14.0, 15):
        alpha = (y - 4.0) / 10.0
        gx = 18.0 - alpha * math.tan(grip_rad) * 10.0
        add_box(grid, (gx, y, cz), (1.8, 0.6, 1.4))

    # Trigger guard & trigger
    add_box(grid, (15.5, 12.0, cz), (1.5, 0.4, 0.9))

    # Slide & Barrel
    barrel_len = random.uniform(14.0, 18.0)
    slide_w = barrel_len * 0.5
    slide_cx = 25.0 - slide_w
    add_box(grid, (slide_cx, 15.5, cz), (slide_w, 1.8, 1.6))
    # Front barrel tip
    add_cylinder(grid, (slide_cx - slide_w - 2.0, 15.5, cz), (slide_cx - slide_w, 15.5, cz), 1.1)
    return grid

def generate_rifle():
    grid = create_empty_voxel()
    cz = 16.0
    # Shoulder stock (rear)
    add_box(grid, (26.0, 14.5, cz), (3.5, 2.2, 1.2))
    # Grip
    add_box(grid, (21.5, 10.5, cz), (1.2, 2.8, 1.2))
    # Receiver box
    add_box(grid, (18.0, 15.0, cz), (4.5, 1.8, 1.5))
    # Magazine
    add_box(grid, (17.5, 10.0, cz), (1.4, 2.5, 1.1))
    # Long barrel extending forward
    add_cylinder(grid, (3.0, 15.2, cz), (14.0, 15.2, cz), 0.9)
    # Handguard
    add_box(grid, (9.0, 14.8, cz), (3.5, 1.3, 1.3))
    # Top scope
    if random.random() < 0.7:
        add_cylinder(grid, (15.0, 17.5, cz), (21.0, 17.5, cz), 0.8)
    return grid

def generate_blaster():
    grid = create_empty_voxel()
    cz = 16.0
    # Sci-fi grip
    add_cylinder(grid, (20.0, 5.0, cz), (18.0, 14.0, cz), 1.5)
    # Main power body
    add_box(grid, (15.0, 15.0, cz), (5.5, 2.5, 2.2))
    # Power cell cartridge underneath
    add_cylinder(grid, (14.0, 10.5, cz - 1.0), (14.0, 10.5, cz + 1.0), 1.8)
    # Segmented barrel rings
    add_cylinder(grid, (4.0, 15.0, cz), (10.0, 15.0, cz), 1.2)
    for rx in [5.0, 7.5, 9.5]:
        add_cylinder(grid, (rx, 15.0, cz - 0.5), (rx, 15.0, cz + 0.5), 2.0)
    # Muzzle tip
    add_cone(grid, (4.0, 15.0, cz), (2.0, 15.0, cz), 1.2, 2.2)
    return grid

# ==============================================================================
# CATEGORY 2: TOYS (Living beings & animals stylized into toys)
# ==============================================================================

def generate_teddy_bear():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    # Chubby body ellipsoid
    add_ellipsoid(grid, (cx, 11.0, cz), 4.5, 5.0, 4.0)
    # Spherical head
    add_sphere(grid, (cx, 20.0, cz), 3.8)
    # Snout
    add_sphere(grid, (cx, 19.5, cz + 2.8), 1.6)
    # Ears
    add_sphere(grid, (cx - 3.2, 23.2, cz), 1.4)
    add_sphere(grid, (cx + 3.2, 23.2, cz), 1.4)
    # Arms
    add_cylinder(grid, (cx - 3.5, 14.0, cz), (cx - 5.5, 9.5, cz + 1.5), 1.5)
    add_cylinder(grid, (cx + 3.5, 14.0, cz), (cx + 5.5, 9.5, cz + 1.5), 1.5)
    # Stubby seated legs
    add_ellipsoid(grid, (cx - 3.2, 6.5, cz + 2.0), 1.8, 1.8, 2.8)
    add_ellipsoid(grid, (cx + 3.2, 6.5, cz + 2.0), 1.8, 1.8, 2.8)
    return grid

def generate_toy_duck():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    # Round duck body
    add_ellipsoid(grid, (cx - 1.5, 12.0, cz), 5.5, 4.2, 4.5)
    # Upturned tail cone
    add_cone(grid, (cx - 6.0, 13.0, cz), (cx - 9.0, 16.0, cz), 2.5, 0.5)
    # Head sphere
    add_sphere(grid, (cx + 3.5, 18.5, cz), 3.4)
    # Flat duck bill (beak)
    add_box(grid, (cx + 8.0, 17.8, cz), (2.0, 0.7, 2.0))
    return grid

def generate_toy_robot():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    # Cube head
    add_box(grid, (cx, 22.0, cz), (2.8, 2.5, 2.8))
    # Head antenna
    add_cylinder(grid, (cx, 24.5, cz), (cx, 28.0, cz), 0.5)
    add_sphere(grid, (cx, 28.2, cz), 1.0)
    # Neck
    add_cylinder(grid, (cx, 18.5, cz), (cx, 19.5, cz), 1.2)
    # Box torso
    add_box(grid, (cx, 14.0, cz), (4.5, 4.2, 3.2))
    # Dial gauge on chest
    add_cylinder(grid, (cx, 14.5, cz + 2.5), (cx, 14.5, cz + 3.8), 1.8)
    # Arms with claw clamps
    add_cylinder(grid, (cx - 4.5, 16.0, cz), (cx - 6.5, 10.0, cz), 1.1)
    add_cylinder(grid, (cx + 4.5, 16.0, cz), (cx + 6.5, 10.0, cz), 1.1)
    # Legs / Tank treads
    add_box(grid, (cx - 2.8, 6.0, cz), (1.5, 3.5, 3.0))
    add_box(grid, (cx + 2.8, 6.0, cz), (1.5, 3.5, 3.0))
    return grid

def generate_toy_bunny():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    # Body
    add_ellipsoid(grid, (cx, 10.5, cz), 4.2, 4.8, 3.8)
    # Head
    add_sphere(grid, (cx, 18.5, cz), 3.4)
    # Long upright ears
    add_ellipsoid(grid, (cx - 2.0, 25.5, cz), 1.1, 4.5, 0.9)
    add_ellipsoid(grid, (cx + 2.0, 25.5, cz), 1.1, 4.5, 0.9)
    # Puffy tail
    add_sphere(grid, (cx, 8.5, cz - 3.8), 1.5)
    # Front paws
    add_sphere(grid, (cx - 2.2, 5.5, cz + 2.5), 1.4)
    add_sphere(grid, (cx + 2.2, 5.5, cz + 2.5), 1.4)
    return grid

def generate_toy_car():
    grid = create_empty_voxel()
    cx, cz = 16.0, 16.0
    # Chassis
    add_box(grid, (cx, 11.0, cz), (8.5, 2.2, 4.8))
    # Cockpit cabin dome
    add_box(grid, (cx - 1.0, 15.0, cz), (4.5, 2.0, 3.8))
    # 4 Wheels
    wheel_r = 2.4
    for wx in [cx - 5.5, cx + 5.5]:
        for wz in [cz - 4.8, cz + 4.8]:
            add_cylinder(grid, (wx, 9.0, wz - 1.0), (wx, 9.0, wz + 1.0), wheel_r)
    # Rear spoiler
    add_box(grid, (cx - 7.5, 15.5, cz), (0.8, 1.8, 4.5))
    return grid

# ==============================================================================
# CATEGORY 3: ELECTRONICS & GADGETS
# ==============================================================================

def generate_gamepad():
    grid = create_empty_voxel()
    cx, cy, cz = 16.0, 16.0, 16.0
    # Central bridge body
    add_box(grid, (cx, cy, cz), (5.5, 3.2, 2.2))
    # Left & Right ergonomic grips
    add_cylinder(grid, (cx - 5.0, cy + 1.0, cz), (cx - 8.5, cy - 6.5, cz), 2.4)
    add_cylinder(grid, (cx + 5.0, cy + 1.0, cz), (cx + 8.5, cy - 6.5, cz), 2.4)
    # D-pad cross on left
    add_box(grid, (cx - 3.5, cy + 1.0, cz + 2.2), (1.6, 0.6, 0.6))
    add_box(grid, (cx - 3.5, cy + 1.0, cz + 2.2), (0.6, 1.6, 0.6))
    # Action buttons on right
    add_sphere(grid, (cx + 3.5, cy + 2.0, cz + 2.2), 0.8)
    add_sphere(grid, (cx + 3.5, cy - 0.2, cz + 2.2), 0.8)
    # Dual analog stick pods
    add_cylinder(grid, (cx - 2.0, cy - 2.0, cz + 1.5), (cx - 2.0, cy - 2.0, cz + 3.0), 1.2)
    add_cylinder(grid, (cx + 2.0, cy - 2.0, cz + 1.5), (cx + 2.0, cy - 2.0, cz + 3.0), 1.2)
    return grid

def generate_handheld_console():
    grid = create_empty_voxel()
    cx, cy, cz = 16.0, 16.0, 16.0
    # Main vertical rectangular body (Game Boy style)
    add_box(grid, (cx, cy, cz), (5.5, 9.5, 2.0))
    # Recessed screen frame
    add_box(grid, (cx, cy + 3.5, cz + 1.8), (4.2, 3.8, 0.6))
    # D-pad cross
    add_box(grid, (cx - 2.5, cy - 3.5, cz + 2.0), (1.5, 0.6, 0.5))
    add_box(grid, (cx - 2.5, cy - 3.5, cz + 2.0), (0.6, 1.5, 0.5))
    # A & B buttons
    add_cylinder(grid, (cx + 2.2, cy - 4.2, cz + 1.8), (cx + 2.2, cy - 4.2, cz + 2.6), 0.7)
    add_cylinder(grid, (cx + 3.8, cy - 2.8, cz + 1.8), (cx + 3.8, cy - 2.8, cz + 2.6), 0.7)
    return grid

def generate_retro_tv():
    grid = create_empty_voxel()
    cx, cy, cz = 16.0, 14.0, 16.0
    # Bulky cabinet
    add_box(grid, (cx, cy, cz), (7.0, 6.0, 5.5))
    # Convex CRT screen on front
    add_box(grid, (cx - 1.5, cy, cz + 5.2), (4.8, 4.5, 1.0))
    # Knobs / dials on right panel
    add_cylinder(grid, (cx + 4.8, cy + 2.5, cz + 5.0), (cx + 4.8, cy + 2.5, cz + 6.4), 1.1)
    add_cylinder(grid, (cx + 4.8, cy - 1.5, cz + 5.0), (cx + 4.8, cy - 1.5, cz + 6.4), 1.1)
    # V-shaped rabbit ear antennae on top
    add_cylinder(grid, (cx, cy + 6.0, cz), (cx - 6.0, cy + 14.0, cz), 0.5)
    add_cylinder(grid, (cx, cy + 6.0, cz), (cx + 6.0, cy + 14.0, cz), 0.5)
    # 4 small feet pegs
    for fx in [cx - 5.0, cx + 5.0]:
        for fz in [cz - 4.0, cz + 4.0]:
            add_cylinder(grid, (fx, cy - 6.0, fz), (fx, cy - 7.5, fz), 0.8)
    return grid

def generate_camera():
    grid = create_empty_voxel()
    cx, cy, cz = 16.0, 14.0, 16.0
    # Main camera body box
    add_box(grid, (cx, cy, cz), (8.0, 5.0, 3.2))
    # Protruding concentric cylindrical lens barrel
    add_cylinder(grid, (cx - 0.5, cy, cz + 2.5), (cx - 0.5, cy, cz + 7.5), 3.8)
    add_cylinder(grid, (cx - 0.5, cy, cz + 6.5), (cx - 0.5, cy, cz + 8.5), 3.2)
    # Top prism / flash block
    add_box(grid, (cx - 0.5, cy + 5.8, cz), (2.8, 1.2, 2.5))
    # Shutter button on top right
    add_cylinder(grid, (cx + 5.5, cy + 5.0, cz), (cx + 5.5, cy + 6.5, cz), 1.0)
    return grid

# ==============================================================================
# CATEGORY REGISTRIES & DISPATCHER
# ==============================================================================

CATEGORY_GENERATORS = {
    0: [ # WEAPONS (Melee + Firearms)
        (generate_sword, 0.20),
        (generate_dagger, 0.10),
        (generate_axe, 0.10),
        (generate_hammer, 0.10),
        (generate_bat, 0.08),
        (generate_shield, 0.08),
        (generate_pistol, 0.12),
        (generate_rifle, 0.12),
        (generate_blaster, 0.10),
    ],
    1: [ # TOYS (Animals & Figures)
        (generate_teddy_bear, 0.25),
        (generate_toy_duck, 0.20),
        (generate_toy_robot, 0.20),
        (generate_toy_bunny, 0.20),
        (generate_toy_car, 0.15),
    ],
    2: [ # ELECTRONICS & GADGETS
        (generate_gamepad, 0.30),
        (generate_handheld_console, 0.25),
        (generate_retro_tv, 0.25),
        (generate_camera, 0.20),
    ]
}

CATEGORY_NAMES = {
    0: "Weapons",
    1: "Toys",
    2: "Electronics"
}

def sample_voxel_by_category(category_id):
    funcs, weights = zip(*CATEGORY_GENERATORS[category_id])
    chosen_gen = random.choices(funcs, weights=weights, k=1)[0]
    return chosen_gen()

# ==============================================================================
# 2D STROKE RASTERIZATION & HUMAN SKETCH SIMULATION
# ==============================================================================

def render_sketch_from_voxel(voxel_grid, augment=True):
    """
    Simulates a human VR/mouse gesture sketch of the 3D voxel object:
    1. Frontal orthographic projection (collapse Z depth).
    2. Upsample to 64x64.
    3. Edge extraction & human stroke dilation.
    4. Real-world hand drawing imperfections (rotation, scale, gaps).
    """
    proj_2d = np.max(voxel_grid, axis=2) # Shape: (32, 32)
    zoom_factor = IMAGE_DIM / VOXEL_DIM
    sketch = ndimage.zoom(proj_2d, zoom_factor, order=1)
    sketch = (sketch > 0.3).astype(np.float32)

    struct = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(sketch, structure=struct)
    edges = (sketch - eroded).astype(np.float32)

    stroke_mode = random.random()
    if stroke_mode < 0.65:
        # Outline sketch
        stroke = edges
    elif stroke_mode < 0.85:
        # Thick outline
        stroke = ndimage.binary_dilation(edges, structure=struct, iterations=1).astype(np.float32)
    else:
        # Filled silhouette
        stroke = sketch

    if augment:
        angle = random.uniform(-12.0, 12.0)
        stroke = ndimage.rotate(stroke, angle, reshape=False, order=1, mode='constant', cval=0.0)

        shift_x = random.uniform(-2.0, 2.0)
        shift_y = random.uniform(-2.0, 2.0)
        stroke = ndimage.shift(stroke, (shift_x, shift_y), order=1, mode='constant', cval=0.0)

        if random.random() < 0.4:
            stroke = ndimage.binary_dilation(stroke > 0.4, iterations=1).astype(np.float32)
        elif random.random() < 0.2:
            stroke = ndimage.binary_erosion(stroke > 0.5, iterations=1).astype(np.float32)

        # Hand lifting gap simulation
        if random.random() < 0.25:
            rx = random.randint(10, 50)
            ry = random.randint(10, 50)
            rw = random.randint(4, 8)
            stroke[rx:rx+rw, ry:ry+rw] *= 0.0

    stroke = np.clip(stroke, 0.0, 1.0)
    stroke = (stroke > 0.25).astype(np.float32)
    return stroke

# ==============================================================================
# DATASET GENERATION PIPELINE
# ==============================================================================

def generate_dataset(num_samples=36000, output_path="Tools/DataGen/dataset_multicategory.npz"):
    print(f"[DatasetGen] Generating {num_samples} balanced multi-category samples...")
    print("  Categories: 0: Weapons (Melee+Guns) | 1: Toys (Animals/Figures) | 2: Electronics (Gadgets)")

    inputs = np.zeros((num_samples, 1, IMAGE_DIM, IMAGE_DIM), dtype=np.float32)
    targets = np.zeros((num_samples, 1, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), dtype=np.float32)
    categories = np.zeros(num_samples, dtype=np.int32)

    samples_per_cat = num_samples // 3

    for i in range(num_samples):
        cat_id = i // samples_per_cat
        if cat_id > 2:
            cat_id = 2

        vox = sample_voxel_by_category(cat_id)
        sketch = render_sketch_from_voxel(vox, augment=True)

        inputs[i, 0] = sketch
        targets[i, 0] = vox
        categories[i] = cat_id

        if (i + 1) % 3000 == 0 or (i + 1) == num_samples:
            print(f"  Processed {i + 1}/{num_samples} samples ({CATEGORY_NAMES[cat_id]} completed)...")

    print(f"[DatasetGen] Saving compressed dataset to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, inputs=inputs, targets=targets, categories=categories)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[DatasetGen] Successfully generated {num_samples} samples! File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Category 2D-to-3D Voxel Dataset Generator")
    parser.add_argument("--samples", type=int, default=36000, help="Total number of samples")
    parser.add_argument("--out", type=str, default="Tools/DataGen/dataset_multicategory.npz", help="Output file path")
    args = parser.parse_args()

    generate_dataset(num_samples=args.samples, output_path=args.out)
