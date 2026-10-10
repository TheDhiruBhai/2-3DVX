"""
Procedural Tri-View 3D Colored Voxel Dataset Generator (AntiGravity Architecture Spike)
Resolution: 64x64x64 3D Voxel Grid (4 Channels: Occupancy, R, G, B)
Input: 1x64x64 Simulated 2D Sketch from 3 Orthographic Views (Front, Side, Top)
Multi-Task Targets:
- Category ID: 0 (Sword), 1 (Shield), 2 (Projectile/Blaster)
- View ID: 0 (Front XY), 1 (Side ZY), 2 (Top XZ)
- 3D Voxel Grid: (4, 64, 64, 64) [Channel 0: Occupancy, Channels 1..3: RGB]
"""

import os
import sys
import math
import random
import argparse
import numpy as np
from scipy import ndimage

VOXEL_DIM = 64
IMAGE_DIM = 64

def create_empty_colored_voxel():
    # 4 channels: 0: Occupancy (0 or 1), 1: R (0..1), 2: G (0..1), 3: B (0..1)
    return np.zeros((4, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), dtype=np.float32)

# ==============================================================================
# 3D CSG PRIMITIVES WITH PALETTE COLOR
# ==============================================================================

def add_box_colored(grid, center, half_extents, rgb):
    c = np.array(center, dtype=np.float32)
    h = np.array(half_extents, dtype=np.float32)
    min_pt = np.clip(np.floor(c - h).astype(int), 0, VOXEL_DIM - 1)
    max_pt = np.clip(np.ceil(c + h).astype(int), 0, VOXEL_DIM - 1)

    sl_x = slice(min_pt[0], max_pt[0] + 1)
    sl_y = slice(min_pt[1], max_pt[1] + 1)
    sl_z = slice(min_pt[2], max_pt[2] + 1)

    grid[0, sl_x, sl_y, sl_z] = 1.0
    grid[1, sl_x, sl_y, sl_z] = rgb[0]
    grid[2, sl_x, sl_y, sl_z] = rgb[1]
    grid[3, sl_x, sl_y, sl_z] = rgb[2]
    return grid

def add_cylinder_colored(grid, p0, p1, radius, rgb):
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
    sl_x = slice(min_pt[0], max_pt[0] + 1)
    sl_y = slice(min_pt[1], max_pt[1] + 1)
    sl_z = slice(min_pt[2], max_pt[2] + 1)

    mask = inside.astype(bool)
    grid[0, sl_x, sl_y, sl_z][mask] = 1.0
    grid[1, sl_x, sl_y, sl_z][mask] = rgb[0]
    grid[2, sl_x, sl_y, sl_z][mask] = rgb[1]
    grid[3, sl_x, sl_y, sl_z][mask] = rgb[2]
    return grid

def add_sphere_colored(grid, center, radius, rgb):
    c = np.array(center, dtype=np.float32)
    min_pt = np.clip(np.floor(c - radius).astype(int), 0, VOXEL_DIM - 1)
    max_pt = np.clip(np.ceil(c + radius).astype(int), 0, VOXEL_DIM - 1)

    xs = np.arange(min_pt[0], max_pt[0] + 1)
    ys = np.arange(min_pt[1], max_pt[1] + 1)
    zs = np.arange(min_pt[2], max_pt[2] + 1)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    dist_sq = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2

    inside = dist_sq <= (radius ** 2)
    sl_x = slice(min_pt[0], max_pt[0] + 1)
    sl_y = slice(min_pt[1], max_pt[1] + 1)
    sl_z = slice(min_pt[2], max_pt[2] + 1)

    mask = inside.astype(bool)
    grid[0, sl_x, sl_y, sl_z][mask] = 1.0
    grid[1, sl_x, sl_y, sl_z][mask] = rgb[0]
    grid[2, sl_x, sl_y, sl_z][mask] = rgb[1]
    grid[3, sl_x, sl_y, sl_z][mask] = rgb[2]
    return grid

def add_ellipsoid_colored(grid, center, rx, ry, rz, rgb):
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
    sl_x = slice(min_pt[0], max_pt[0] + 1)
    sl_y = slice(min_pt[1], max_pt[1] + 1)
    sl_z = slice(min_pt[2], max_pt[2] + 1)

    mask = inside.astype(bool)
    grid[0, sl_x, sl_y, sl_z][mask] = 1.0
    grid[1, sl_x, sl_y, sl_z][mask] = rgb[0]
    grid[2, sl_x, sl_y, sl_z][mask] = rgb[1]
    grid[3, sl_x, sl_y, sl_z][mask] = rgb[2]
    return grid

def add_cone_colored(grid, p0, p1, r0, r1, rgb):
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

    r_at_t = r0 + t * (r1 - r0)
    inside = dist_sq <= (r_at_t ** 2)

    sl_x = slice(min_pt[0], max_pt[0] + 1)
    sl_y = slice(min_pt[1], max_pt[1] + 1)
    sl_z = slice(min_pt[2], max_pt[2] + 1)

    mask = inside.astype(bool)
    grid[0, sl_x, sl_y, sl_z][mask] = 1.0
    grid[1, sl_x, sl_y, sl_z][mask] = rgb[0]
    grid[2, sl_x, sl_y, sl_z][mask] = rgb[1]
    grid[3, sl_x, sl_y, sl_z][mask] = rgb[2]
    return grid

# ==============================================================================
# PARAMETRIC ASSET GENERATORS (64^3 Resolution with Color)
# ==============================================================================

# Palette colors
COLOR_STEEL = [0.85, 0.90, 0.95]
COLOR_GOLD = [0.95, 0.80, 0.25]
COLOR_LEATHER = [0.45, 0.28, 0.15]
COLOR_WOOD = [0.48, 0.30, 0.18]
COLOR_CRIMSON = [0.85, 0.20, 0.25]
COLOR_COBALT = [0.20, 0.40, 0.85]
COLOR_DARK_METAL = [0.22, 0.24, 0.28]
COLOR_PLASMA_CYAN = [0.15, 0.95, 0.95]
COLOR_AMBER = [0.95, 0.55, 0.12]
COLOR_YELLOW = [0.98, 0.88, 0.15]
COLOR_ORANGE = [0.98, 0.50, 0.10]
COLOR_TEDDY_BROWN = [0.55, 0.35, 0.22]
COLOR_TEDDY_LIGHT = [0.72, 0.52, 0.36]
COLOR_ROBOT_SILVER = [0.75, 0.80, 0.85]
COLOR_ROBOT_RED = [0.90, 0.25, 0.25]
COLOR_BUNNY_WHITE = [0.94, 0.94, 0.96]
COLOR_BUNNY_PINK = [0.96, 0.65, 0.75]
COLOR_CONSOLE_GREY = [0.70, 0.72, 0.75]
COLOR_CONSOLE_SCREEN = [0.40, 0.55, 0.35]
COLOR_TV_WOOD = [0.38, 0.24, 0.14]
COLOR_TV_SCREEN = [0.65, 0.72, 0.80]
COLOR_GAMEPAD_DARK = [0.25, 0.27, 0.30]

# --- 1. WEAPONS ---
def generate_sword_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    handle_len = random.uniform(10.0, 14.0)
    add_cylinder_colored(grid, (cx, 4.0, cz), (cx, 4.0 + handle_len, cz), random.uniform(2.0, 2.6), COLOR_LEATHER)
    add_sphere_colored(grid, (cx, 4.0, cz), random.uniform(3.0, 3.8), COLOR_GOLD)
    guard_y = 4.0 + handle_len
    guard_w = random.uniform(12.0, 18.0)
    add_box_colored(grid, (cx, guard_y, cz), (guard_w, 2.0, 2.5), COLOR_GOLD)
    blade_start = guard_y + 1.8
    blade_end = random.uniform(52.0, 60.0)
    blade_len = blade_end - blade_start
    blade_base_w = random.uniform(4.5, 6.5)
    blade_tip_w = random.uniform(1.0, 2.5)
    for y_idx in range(int(blade_start), int(blade_end) + 1):
        alpha = (y_idx - blade_start) / max(blade_len, 1e-4)
        cur_w = (1.0 - alpha) * blade_base_w + alpha * blade_tip_w
        add_box_colored(grid, (cx, y_idx, cz), (cur_w, 0.6, 1.8), COLOR_STEEL)
    return grid

def generate_shield_64():
    grid = create_empty_colored_voxel()
    cx, cy, cz = 32.0, 32.0, 32.0
    rx = random.uniform(14.0, 19.0)
    ry = random.uniform(18.0, 24.0)
    rz = random.uniform(3.5, 4.8)
    rim_color = random.choice([COLOR_STEEL, COLOR_GOLD])
    body_color = random.choice([COLOR_CRIMSON, COLOR_COBALT])
    add_ellipsoid_colored(grid, (cx, cy, cz), rx, ry, rz, rim_color)
    add_ellipsoid_colored(grid, (cx, cy, cz + 0.8), rx - 2.5, ry - 2.5, rz, body_color)
    add_sphere_colored(grid, (cx, cy, cz + 2.5), random.uniform(4.5, 6.0), COLOR_GOLD)
    return grid

def generate_blaster_64():
    grid = create_empty_colored_voxel()
    cz = 32.0
    add_cylinder_colored(grid, (42.0, 10.0, cz), (38.0, 28.0, cz), 3.2, COLOR_DARK_METAL)
    add_box_colored(grid, (30.0, 31.0, cz), (12.0, 5.0, 4.5), COLOR_DARK_METAL)
    add_cylinder_colored(grid, (30.0, 21.0, cz - 1.5), (30.0, 21.0, cz + 1.5), 3.8, COLOR_AMBER)
    add_cylinder_colored(grid, (8.0, 31.0, cz), (20.0, 31.0, cz), 2.6, COLOR_DARK_METAL)
    for rx in [11.0, 16.0, 21.0]:
        add_cylinder_colored(grid, (rx, 31.0, cz - 1.0), (rx, 31.0, cz + 1.0), 4.2, COLOR_PLASMA_CYAN)
    add_cylinder_colored(grid, (4.0, 31.0, cz), (8.0, 31.0, cz), 3.6, COLOR_GOLD)
    return grid

def generate_axe_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    add_cylinder_colored(grid, (cx, 4.0, cz), (cx, 58.0, cz), 2.2, COLOR_WOOD)
    add_cylinder_colored(grid, (cx, 4.0, cz), (cx, 8.0, cz), 2.8, COLOR_STEEL)
    add_box_colored(grid, (cx + 10.0, 46.0, cz), (9.0, 7.5, 2.0), COLOR_STEEL)
    add_box_colored(grid, (cx + 18.0, 46.0, cz), (3.0, 11.0, 1.2), COLOR_STEEL)
    add_box_colored(grid, (cx, 46.0, cz), (3.5, 4.0, 3.5), COLOR_DARK_METAL)
    return grid

# --- 2. TOYS ---
def generate_teddy_bear_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    add_ellipsoid_colored(grid, (cx, 22.0, cz), 11.0, 11.0, 10.0, COLOR_TEDDY_BROWN)
    add_ellipsoid_colored(grid, (cx, 21.0, cz + 3.0), 7.0, 7.0, 6.0, COLOR_TEDDY_LIGHT)
    add_sphere_colored(grid, (cx, 40.0, cz), 9.0, COLOR_TEDDY_BROWN)
    add_sphere_colored(grid, (cx, 38.0, cz + 7.5), 3.8, COLOR_TEDDY_LIGHT)
    add_sphere_colored(grid, (cx, 39.0, cz + 10.5), 1.4, COLOR_DARK_METAL)
    add_sphere_colored(grid, (cx - 8.5, 47.0, cz), 3.2, COLOR_TEDDY_BROWN)
    add_sphere_colored(grid, (cx + 8.5, 47.0, cz), 3.2, COLOR_TEDDY_BROWN)
    add_cylinder_colored(grid, (cx - 10.0, 28.0, cz), (cx - 15.0, 20.0, cz + 3.0), 3.2, COLOR_TEDDY_BROWN)
    add_cylinder_colored(grid, (cx + 10.0, 28.0, cz), (cx + 15.0, 20.0, cz + 3.0), 3.2, COLOR_TEDDY_BROWN)
    add_ellipsoid_colored(grid, (cx - 8.5, 12.0, cz + 6.0), 4.5, 4.5, 6.5, COLOR_TEDDY_BROWN)
    add_ellipsoid_colored(grid, (cx + 8.5, 12.0, cz + 6.0), 4.5, 4.5, 6.5, COLOR_TEDDY_BROWN)
    return grid

def generate_rubber_duck_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    add_ellipsoid_colored(grid, (cx - 4.0, 24.0, cz), 13.0, 10.0, 11.0, COLOR_YELLOW)
    add_cone_colored(grid, (cx - 16.0, 26.0, cz), (cx - 23.0, 33.0, cz), 6.0, 1.5, COLOR_YELLOW)
    add_sphere_colored(grid, (cx + 7.0, 38.0, cz), 8.0, COLOR_YELLOW)
    add_box_colored(grid, (cx + 17.0, 36.0, cz), (5.0, 2.0, 4.5), COLOR_ORANGE)
    add_sphere_colored(grid, (cx + 10.0, 41.0, cz + 6.0), 1.2, COLOR_DARK_METAL)
    add_sphere_colored(grid, (cx + 10.0, 41.0, cz - 6.0), 1.2, COLOR_DARK_METAL)
    return grid

def generate_toy_robot_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    add_box_colored(grid, (cx, 44.0, cz), (6.5, 6.0, 6.5), COLOR_ROBOT_SILVER)
    add_cylinder_colored(grid, (cx, 50.0, cz), (cx, 58.0, cz), 1.2, COLOR_STEEL)
    add_sphere_colored(grid, (cx, 58.0, cz), 2.2, COLOR_YELLOW)
    add_cylinder_colored(grid, (cx, 37.0, cz), (cx, 39.0, cz), 2.8, COLOR_DARK_METAL)
    add_box_colored(grid, (cx, 28.0, cz), (10.0, 9.0, 7.0), COLOR_ROBOT_SILVER)
    add_cylinder_colored(grid, (cx, 29.0, cz + 5.5), (cx, 29.0, cz + 7.5), 4.2, COLOR_ROBOT_RED)
    add_cylinder_colored(grid, (cx - 12.0, 33.0, cz), (cx - 16.0, 20.0, cz), 2.5, COLOR_ROBOT_SILVER)
    add_cylinder_colored(grid, (cx + 12.0, 33.0, cz), (cx + 16.0, 20.0, cz), 2.5, COLOR_ROBOT_SILVER)
    add_box_colored(grid, (cx - 6.0, 12.0, cz), (4.0, 7.0, 7.5), COLOR_DARK_METAL)
    add_box_colored(grid, (cx + 6.0, 12.0, cz), (4.0, 7.0, 7.5), COLOR_DARK_METAL)
    return grid

def generate_toy_bunny_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    add_ellipsoid_colored(grid, (cx, 21.0, cz), 9.5, 10.5, 8.5, COLOR_BUNNY_WHITE)
    add_sphere_colored(grid, (cx, 37.0, cz), 7.5, COLOR_BUNNY_WHITE)
    add_ellipsoid_colored(grid, (cx - 4.5, 52.0, cz), 2.2, 9.5, 2.0, COLOR_BUNNY_WHITE)
    add_ellipsoid_colored(grid, (cx + 4.5, 52.0, cz), 2.2, 9.5, 2.0, COLOR_BUNNY_WHITE)
    add_ellipsoid_colored(grid, (cx - 4.5, 52.0, cz + 0.8), 1.2, 7.5, 1.0, COLOR_BUNNY_PINK)
    add_ellipsoid_colored(grid, (cx + 4.5, 52.0, cz + 0.8), 1.2, 7.5, 1.0, COLOR_BUNNY_PINK)
    add_sphere_colored(grid, (cx, 17.0, cz - 8.5), 3.5, COLOR_BUNNY_WHITE)
    add_sphere_colored(grid, (cx - 5.5, 11.0, cz + 5.5), 3.0, COLOR_BUNNY_WHITE)
    add_sphere_colored(grid, (cx + 5.5, 11.0, cz + 5.5), 3.0, COLOR_BUNNY_WHITE)
    return grid

# --- 3. ELECTRONICS ---
def generate_gamepad_64():
    grid = create_empty_colored_voxel()
    cx, cy, cz = 32.0, 32.0, 32.0
    add_box_colored(grid, (cx, cy, cz), (12.0, 7.0, 4.5), COLOR_GAMEPAD_DARK)
    add_cylinder_colored(grid, (cx - 11.0, cy + 2.0, cz), (cx - 18.0, cy - 13.0, cz), 5.2, COLOR_GAMEPAD_DARK)
    add_cylinder_colored(grid, (cx + 11.0, cy + 2.0, cz), (cx + 18.0, cy - 13.0, cz), 5.2, COLOR_GAMEPAD_DARK)
    add_box_colored(grid, (cx - 8.0, cy + 2.0, cz + 4.8), (3.5, 1.2, 1.2), COLOR_DARK_METAL)
    add_box_colored(grid, (cx - 8.0, cy + 2.0, cz + 4.8), (1.2, 3.5, 1.2), COLOR_DARK_METAL)
    add_sphere_colored(grid, (cx + 8.0, cy + 4.0, cz + 4.8), 1.6, COLOR_CRIMSON)
    add_sphere_colored(grid, (cx + 8.0, cy - 0.5, cz + 4.8), 1.6, COLOR_COBALT)
    add_cylinder_colored(grid, (cx - 4.5, cy - 4.0, cz + 3.0), (cx - 4.5, cy - 4.0, cz + 5.5), 2.6, COLOR_DARK_METAL)
    add_cylinder_colored(grid, (cx + 4.5, cy - 4.0, cz + 3.0), (cx + 4.5, cy - 4.0, cz + 5.5), 2.6, COLOR_DARK_METAL)
    return grid

def generate_retro_tv_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    add_box_colored(grid, (cx, 29.0, cz), (15.0, 13.0, 12.0), COLOR_TV_WOOD)
    add_box_colored(grid, (cx - 3.5, 29.0, cz + 10.5), (10.0, 9.5, 2.5), COLOR_TV_SCREEN)
    add_cylinder_colored(grid, (cx + 10.0, 34.0, cz + 10.0), (cx + 10.0, 34.0, cz + 13.0), 2.2, COLOR_GOLD)
    add_cylinder_colored(grid, (cx + 10.0, 24.0, cz + 10.0), (cx + 10.0, 24.0, cz + 13.0), 2.2, COLOR_GOLD)
    add_cylinder_colored(grid, (cx - 5.0, 42.0, cz), (cx - 15.0, 58.0, cz), 1.0, COLOR_STEEL)
    add_cylinder_colored(grid, (cx + 5.0, 42.0, cz), (cx + 15.0, 58.0, cz), 1.0, COLOR_STEEL)
    add_box_colored(grid, (cx - 10.0, 14.0, cz), (2.5, 2.5, 9.0), COLOR_TV_WOOD)
    add_box_colored(grid, (cx + 10.0, 14.0, cz), (2.5, 2.5, 9.0), COLOR_TV_WOOD)
    return grid

def generate_gameboy_64():
    grid = create_empty_colored_voxel()
    cx, cz = 32.0, 32.0
    add_box_colored(grid, (cx, 32.0, cz), (12.0, 20.0, 4.2), COLOR_CONSOLE_GREY)
    add_box_colored(grid, (cx, 40.0, cz + 3.8), (9.5, 8.0, 1.0), COLOR_DARK_METAL)
    add_box_colored(grid, (cx, 40.0, cz + 4.5), (7.5, 6.0, 0.6), COLOR_CONSOLE_SCREEN)
    add_box_colored(grid, (cx - 5.5, 24.0, cz + 4.2), (3.2, 1.2, 1.0), COLOR_DARK_METAL)
    add_box_colored(grid, (cx - 5.5, 24.0, cz + 4.2), (1.2, 3.2, 1.0), COLOR_DARK_METAL)
    add_cylinder_colored(grid, (cx + 4.5, 22.0, cz + 3.5), (cx + 4.5, 22.0, cz + 5.2), 1.6, COLOR_CRIMSON)
    add_cylinder_colored(grid, (cx + 8.5, 25.0, cz + 3.5), (cx + 8.5, 25.0, cz + 5.2), 1.6, COLOR_CRIMSON)
    return grid

TARGET_ASSET_GENERATORS = [
    (generate_sword_64, 0),        # 0: Sword
    (generate_shield_64, 1),       # 1: Shield
    (generate_blaster_64, 2),      # 2: Blaster / Pistol
    (generate_axe_64, 3),          # 3: Axe
    (generate_teddy_bear_64, 4),   # 4: Teddy Bear
    (generate_rubber_duck_64, 5),  # 5: Rubber Duck
    (generate_toy_robot_64, 6),    # 6: Toy Robot
    (generate_toy_bunny_64, 7),    # 7: Toy Bunny
    (generate_gamepad_64, 8),      # 8: Gamepad
    (generate_retro_tv_64, 9),     # 9: Retro TV
    (generate_gameboy_64, 10),     # 10: Game Boy
]

# ==============================================================================
# TRI-VIEW ORTHOGRAPHIC PROJECTION PIPELINE
# ==============================================================================

def project_canonical_view(voxel_grid_4ch, view_id):
    """
    Computes 2D orthographic projection from 3D voxel grid:
    - View 0: Front View (XY plane): max across Z
    - View 1: Side View (ZY plane):  max across X
    - View 2: Top View (XZ plane):   max across Y
    Returns binary projection (64, 64) in [0.0, 1.0]
    """
    occ = voxel_grid_4ch[0] # Occupancy channel (64, 64, 64)

    if view_id == 0:
        # Front View: Collapse Z depth
        proj = np.max(occ, axis=2) # (X, Y)
    elif view_id == 1:
        # Side View: Collapse X width
        proj = np.max(occ, axis=0) # (Y, Z) -> transpose to (Z, Y)
        proj = proj.T
    elif view_id == 2:
        # Top View: Collapse Y height
        proj = np.max(occ, axis=1) # (X, Z)
    else:
        raise ValueError(f"Invalid view_id: {view_id}")

    return proj.astype(np.float32)

# ==============================================================================
# HUMAN SKETCH SIMULATION (Elastic Deform, Stroke Jitter, Pressure, Dropout)
# ==============================================================================

def elastic_transform(image, alpha=6.0, sigma=2.2):
    """Simulates natural player hand tremor / unsteady stroke curves."""
    shape = image.shape
    dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def render_imperfect_sketch(clean_proj):
    """Transforms a clean 2D projection into a loose, imperfect human drawing."""
    # 1. Edge extraction to convert solid silhouette into outline stroke
    struct = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(clean_proj > 0.4, structure=struct)
    edges = ((clean_proj > 0.4).astype(np.float32) - eroded.astype(np.float32))

    stroke_style = random.random()
    if stroke_style < 0.65:
        # Outline sketch with variable brush thickness
        stroke = ndimage.binary_dilation(edges, structure=struct, iterations=random.choice([1, 2])).astype(np.float32)
    else:
        # Loose filled silhouette
        stroke = clean_proj.copy()

    # 2. Elastic deformation (hand tremor)
    if random.random() < 0.85:
        stroke = elastic_transform(stroke, alpha=random.uniform(4.0, 7.5), sigma=random.uniform(2.0, 3.0))

    # 3. Affine transformations (rotation, translation, scale)
    angle = random.uniform(-10.0, 10.0)
    stroke = ndimage.rotate(stroke, angle, reshape=False, order=1, mode='constant', cval=0.0)

    shift_x = random.uniform(-2.5, 2.5)
    shift_y = random.uniform(-2.5, 2.5)
    stroke = ndimage.shift(stroke, (shift_x, shift_y), order=1, mode='constant', cval=0.0)

    # 4. Stroke thickness jitter (pressure changes)
    if random.random() < 0.35:
        stroke = ndimage.binary_dilation(stroke > 0.3, iterations=1).astype(np.float32)
    elif random.random() < 0.20:
        stroke = ndimage.binary_erosion(stroke > 0.4, iterations=1).astype(np.float32)

    # 5. Hand lifting gaps / stroke dropout
    if random.random() < 0.30:
        rx = random.randint(12, 50)
        ry = random.randint(12, 50)
        rw = random.randint(4, 9)
        stroke[rx:rx+rw, ry:ry+rw] *= 0.0

    stroke = np.clip(stroke, 0.0, 1.0)
    stroke = (stroke > 0.25).astype(np.float32)
    return stroke

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

def generate_triview_dataset(num_samples=3300, output_path="Tools/DataGen/dataset_triview_64.npz"):
    print(f"[DataGen Tri-View] Generating {num_samples} multi-task samples across {len(TARGET_ASSET_GENERATORS)} classes (uint8 packed)...")
    print(f"  Target Shape: (4, 64, 64, 64) | Input: (1, 64, 64)")
    print(f"  Categories: 11 Classes (Weapons, Toys, Electronics)")
    print(f"  Views: 0 (Front XY), 1 (Side ZY), 2 (Top XZ)")

    inputs = np.zeros((num_samples, 1, IMAGE_DIM, IMAGE_DIM), dtype=np.uint8)
    targets_voxels = np.zeros((num_samples, 4, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM), dtype=np.uint8)
    categories = np.zeros(num_samples, dtype=np.int64)
    views = np.zeros(num_samples, dtype=np.int64)

    num_classes = len(TARGET_ASSET_GENERATORS)

    for i in range(num_samples):
        # Balanced category and view selection
        gen_fn, cat_id = TARGET_ASSET_GENERATORS[i % num_classes]
        voxel_4ch = gen_fn()

        # Cycle through Front, Side, Top views
        view_id = (i // num_classes) % 3

        clean_proj = project_canonical_view(voxel_4ch, view_id)
        noisy_sketch = render_imperfect_sketch(clean_proj)

        inputs[i, 0] = (noisy_sketch * 255.0).astype(np.uint8)
        
        # Channel 0: Occupancy (0 or 1)
        targets_voxels[i, 0] = (voxel_4ch[0] > 0.5).astype(np.uint8)
        # Channels 1..3: RGB (0..255)
        targets_voxels[i, 1:4] = np.clip(voxel_4ch[1:4] * 255.0, 0, 255).astype(np.uint8)
        
        categories[i] = cat_id
        views[i] = view_id

        if (i + 1) % 500 == 0 or (i + 1) == num_samples:
            print(f"  Processed {i + 1}/{num_samples} samples...")

    print(f"[DataGen Tri-View] Saving compressed dataset to {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=inputs,
        targets_voxels=targets_voxels,
        categories=categories,
        views=views
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[DataGen Tri-View] Generated {num_samples} multi-task samples! Size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tri-View 64^3 Multi-Task Voxel Dataset Generator")
    parser.add_argument("--samples", type=int, default=2700, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="Tools/DataGen/dataset_triview_64.npz", help="Output path")
    args = parser.parse_args()

    generate_triview_dataset(num_samples=args.samples, output_path=args.out)
