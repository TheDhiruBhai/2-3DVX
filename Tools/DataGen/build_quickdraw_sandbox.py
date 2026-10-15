"""
Quick, Draw! 50-Class Sandbox Pipeline
Downloads 50 diverse creative sandbox categories from Google Cloud Storage via HTTP Range requests,
preprocesses them into the 2-Channel normalized [1, 2, 64, 64] format, trains the Universal Sandbox Model,
and exports the ONNX model and semantic palette configuration for Unity.
"""

import os
import sys
import json
import time
import math
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 50 CURATED SANDBOX CATEGORIES & SEMANTIC PALETTES
# ==============================================================================

SANDBOX_CLASSES = [
    # Weapons & Tools (0..15)
    {"name": "sword", "label": "Sword", "family": "Weapons", "color": [190, 195, 205], "accent": [230, 180, 50]},
    {"name": "axe", "label": "Battleaxe", "family": "Weapons", "color": [120, 130, 140], "accent": [115, 75, 45]},
    {"name": "rifle", "label": "Rifle", "family": "Weapons", "color": [70, 75, 80], "accent": [130, 85, 50]},
    {"name": "cannon", "label": "Cannon", "family": "Weapons", "color": [50, 55, 60], "accent": [180, 140, 60]},
    {"name": "knife", "label": "Dagger", "family": "Weapons", "color": [210, 215, 220], "accent": [40, 40, 45]},
    {"name": "boomerang", "label": "Boomerang", "family": "Weapons", "color": [160, 105, 60], "accent": [220, 70, 50]},
    {"name": "hammer", "label": "War Hammer", "family": "Tools", "color": [130, 135, 140], "accent": [110, 70, 40]},
    {"name": "screwdriver", "label": "Screwdriver", "family": "Tools", "color": [230, 190, 40], "accent": [180, 185, 190]},
    {"name": "saw", "label": "Hand Saw", "family": "Tools", "color": [200, 205, 210], "accent": [220, 100, 40]},
    {"name": "pliers", "label": "Pliers", "family": "Tools", "color": [210, 50, 50], "accent": [150, 155, 160]},
    {"name": "flashlight", "label": "Flashlight", "family": "Tools", "color": [45, 50, 55], "accent": [255, 240, 100]},
    {"name": "umbrella", "label": "Umbrella", "family": "Gear", "color": [60, 140, 220], "accent": [40, 40, 45]},
    {"name": "key", "label": "Golden Key", "family": "Gear", "color": [235, 195, 55], "accent": [255, 230, 120]},
    {"name": "anvil", "label": "Blacksmith Anvil", "family": "Tools", "color": [65, 70, 75], "accent": [90, 95, 100]},
    {"name": "helmet", "label": "Knight Helmet", "family": "Armor", "color": [170, 175, 185], "accent": [200, 60, 60]},
    {"name": "crown", "label": "Royal Crown", "family": "Armor", "color": [240, 200, 50], "accent": [220, 40, 50]},

    # Treasures & Celestial (16..20)
    {"name": "diamond", "label": "Magic Diamond", "family": "Treasure", "color": [90, 225, 240], "accent": [220, 250, 255]},
    {"name": "star", "label": "Shining Star", "family": "Celestial", "color": [255, 225, 60], "accent": [255, 245, 160]},
    {"name": "sun", "label": "Radiant Sun", "family": "Celestial", "color": [255, 180, 40], "accent": [255, 220, 70]},
    {"name": "cloud", "label": "Fluffy Cloud", "family": "Celestial", "color": [230, 240, 250], "accent": [190, 210, 235]},
    {"name": "mountain", "label": "Snowy Mountain", "family": "Nature", "color": [110, 115, 125], "accent": [245, 250, 255]},

    # Vehicles & Transport (21..26)
    {"name": "airplane", "label": "Airplane", "family": "Vehicles", "color": [235, 240, 245], "accent": [50, 120, 210]},
    {"name": "helicopter", "label": "Helicopter", "family": "Vehicles", "color": [75, 110, 65], "accent": [40, 40, 40]},
    {"name": "car", "label": "Sports Car", "family": "Vehicles", "color": [225, 50, 50], "accent": [40, 40, 45]},
    {"name": "bicycle", "label": "Bicycle", "family": "Vehicles", "color": [50, 195, 200], "accent": [30, 30, 35]},
    {"name": "sailboat", "label": "Sailboat", "family": "Vehicles", "color": [245, 245, 250], "accent": [140, 85, 45]},
    {"name": "skateboard", "label": "Skateboard", "family": "Vehicles", "color": [180, 120, 60], "accent": [80, 220, 100]},

    # Creatures & Living Beings (27..33)
    {"name": "duck", "label": "Rubber Ducky", "family": "Creatures", "color": [245, 215, 45], "accent": [240, 120, 35]},
    {"name": "cat", "label": "Cat Figurine", "family": "Creatures", "color": [230, 140, 65], "accent": [245, 185, 195]},
    {"name": "dog", "label": "Dog Figurine", "family": "Creatures", "color": [175, 125, 75], "accent": [50, 50, 55]},
    {"name": "bear", "label": "Grizzly Bear", "family": "Creatures", "color": [105, 70, 45], "accent": [60, 40, 25]},
    {"name": "fish", "label": "Tropical Fish", "family": "Creatures", "color": [60, 185, 210], "accent": [245, 130, 50]},
    {"name": "bird", "label": "Songbird", "family": "Creatures", "color": [225, 65, 65], "accent": [245, 185, 45]},
    {"name": "frog", "label": "Tree Frog", "family": "Creatures", "color": [80, 185, 70], "accent": [230, 200, 40]},

    # Food & Produce (34..39)
    {"name": "apple", "label": "Red Apple", "family": "Food", "color": [215, 45, 45], "accent": [85, 150, 50]},
    {"name": "banana", "label": "Yellow Banana", "family": "Food", "color": [245, 225, 55], "accent": [130, 110, 40]},
    {"name": "pizza", "label": "Pizza Slice", "family": "Food", "color": [240, 185, 65], "accent": [200, 45, 45]},
    {"name": "donut", "label": "Glazed Donut", "family": "Food", "color": [230, 120, 160], "accent": [245, 235, 100]},
    {"name": "ice cream", "label": "Ice Cream Cone", "family": "Food", "color": [110, 220, 175], "accent": [190, 130, 70]},
    {"name": "hamburger", "label": "Burger", "family": "Food", "color": [215, 160, 85], "accent": [120, 65, 35]},

    # Nature & Plants (40..43)
    {"name": "tree", "label": "Pine Tree", "family": "Nature", "color": [55, 145, 65], "accent": [115, 75, 45]},
    {"name": "flower", "label": "Garden Flower", "family": "Nature", "color": [235, 75, 120], "accent": [245, 210, 55]},
    {"name": "cactus", "label": "Desert Cactus", "family": "Nature", "color": [75, 155, 80], "accent": [220, 65, 95]},
    {"name": "mushroom", "label": "Magic Mushroom", "family": "Nature", "color": [220, 50, 50], "accent": [245, 245, 245]},
    {"name": "campfire", "label": "Campfire", "family": "Nature", "color": [245, 120, 35], "accent": [105, 65, 40]},

    # Electronics & Furniture (44..49)
    {"name": "television", "label": "Retro TV", "family": "Tech", "color": [125, 85, 55], "accent": [50, 55, 65]},
    {"name": "radio", "label": "Boombox Radio", "family": "Tech", "color": [160, 165, 170], "accent": [45, 45, 50]},
    {"name": "guitar", "label": "Acoustic Guitar", "family": "Tech", "color": [200, 125, 60], "accent": [200, 205, 210]},
    {"name": "chair", "label": "Wooden Chair", "family": "Furniture", "color": [155, 105, 55], "accent": [115, 75, 40]},
    {"name": "clock", "label": "Wall Clock", "family": "Tech", "color": [240, 240, 245], "accent": [200, 160, 50]},

    # Out-of-Distribution Noise (50)
    {"name": "ood_scribble", "label": "Scribble / Noise", "family": "OOD", "color": [150, 150, 150], "accent": [80, 80, 80]}
]

NUM_SANDBOX_CLASSES = len(SANDBOX_CLASSES) # 51 classes (0..50)

# ==============================================================================
# FAST STREAMING DOWNLOAD VIA HTTP RANGE (Only ~1.5 MB per category!)
# ==============================================================================

def fetch_class_samples(cls_info, samples_per_class=1200):
    c_name = cls_info["name"]
    if c_name == "ood_scribble":
        return None

    url_name = c_name.replace(" ", "%20")
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{url_name}.npy"
    byte_limit = 1024 + samples_per_class * 784

    req = urllib.request.Request(url, headers={'Range': f'bytes=0-{byte_limit}'})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read()
        header_len = int.from_bytes(raw[8:10], 'little')
        offset = 10 + header_len
        data = raw[offset:offset + samples_per_class * 784]
        arr = np.frombuffer(data, dtype=np.uint8).reshape((-1, 28, 28))
        return arr[:samples_per_class]
    except Exception as e:
        print(f"Error fetching {c_name}: {e}")
        return None

# ==============================================================================
# 2-CHANNEL PREPROCESSING: AABB Centering (52x52) + Chamfer Euclidean SDF
# ==============================================================================

def preprocess_28x28_to_2ch64(img28, target_box=52, canvas_dim=64, d_max=16.0):
    """Upscales 28x28 QuickDraw raster to 64x64 2-Channel SDF."""
    active_y, active_x = np.where(img28 > 40)
    if len(active_x) == 0:
        ch0 = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
        ch1 = np.ones((canvas_dim, canvas_dim), dtype=np.float32)
        return ch0, ch1

    min_x, max_x = int(np.min(active_x)), int(np.max(active_x))
    min_y, max_y = int(np.min(active_y)), int(np.max(active_y))
    w = max_x - min_x + 1
    h = max_y - min_y + 1

    crop = img28[min_y:max_y+1, min_x:max_x+1]
    scale = float(target_box) / max(w, h)
    new_w = max(1, min(target_box, int(round(w * scale))))
    new_h = max(1, min(target_box, int(round(h * scale))))

    # Zoom crop
    zoom_y = new_h / float(h)
    zoom_x = new_w / float(w)
    resized = ndimage.zoom(crop.astype(np.float32), (zoom_y, zoom_x), order=1)
    resized = np.clip(resized / 255.0, 0.0, 1.0)

    actual_h, actual_w = resized.shape
    ox = (canvas_dim - actual_w) // 2
    oy = (canvas_dim - actual_h) // 2

    ch0 = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
    ch0[oy:oy+actual_h, ox:ox+actual_w] = (resized > 0.2).astype(np.float32)

    # Chamfer Euclidean Distance Transform
    dist = ndimage.distance_transform_edt(ch0 == 0)
    ch1 = np.clip(dist / d_max, 0.0, 1.0).astype(np.float32)

    return ch0, ch1

def generate_random_scribble_2ch(canvas_dim=64):
    """Generates synthetic OOD scribble for class 50."""
    ch0 = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
    pts = [np.array([random.randint(15, 48), random.randint(15, 48)])]
    for _ in range(random.randint(8, 25)):
        step = np.random.randn(2) * random.uniform(3, 8)
        next_pt = np.clip(pts[-1] + step, 4, canvas_dim - 5)
        pts.append(next_pt)
    for j in range(len(pts) - 1):
        p1 = tuple(pts[j].astype(int))
        p2 = tuple(pts[j+1].astype(int))
        # Draw line
        length = max(1, int(np.hypot(p2[0]-p1[0], p2[1]-p1[1])))
        xs = np.linspace(p1[0], p2[0], length).astype(int)
        ys = np.linspace(p1[1], p2[1], length).astype(int)
        ch0[ys, xs] = 1.0
    dist = ndimage.distance_transform_edt(ch0 == 0)
    ch1 = np.clip(dist / 16.0, 0.0, 1.0).astype(np.float32)
    return ch0, ch1

# ==============================================================================
# BUILD SANDBOX DATASET
# ==============================================================================

def build_sandbox_dataset(output_path="Tools/DataGen/dataset_sandbox_50.npz", samples_per_class=1000):
    print(f"\n[Sandbox DataGen] Downloading & preprocessing 50 QuickDraw classes ({samples_per_class} samples/class)...")
    t0 = time.time()

    valid_classes = [c for c in SANDBOX_CLASSES if c["name"] != "ood_scribble"]
    results = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_cls = {executor.submit(fetch_class_samples, c, samples_per_class): c for c in valid_classes}
        for future in future_to_cls:
            c = future_to_cls[future]
            res = future.result()
            if res is not None:
                results[c["name"]] = res
                print(f"  ✔ Downloaded {c['name']:<14}: {len(res)} samples")
            else:
                print(f"  ✖ Failed {c['name']}")

    total_classes = len(SANDBOX_CLASSES)
    total_samples = len(valid_classes) * samples_per_class + samples_per_class

    inputs = np.zeros((total_samples, 2, 64, 64), dtype=np.uint8)
    categories = np.zeros(total_samples, dtype=np.int64)

    idx = 0
    for c_id, c_info in enumerate(SANDBOX_CLASSES):
        c_name = c_info["name"]
        if c_name == "ood_scribble":
            # Generate OOD scribbles
            for _ in range(samples_per_class):
                ch0, ch1 = generate_random_scribble_2ch(64)
                inputs[idx, 0] = (ch0 * 255.0).astype(np.uint8)
                inputs[idx, 1] = (ch1 * 255.0).astype(np.uint8)
                categories[idx] = c_id
                idx += 1
            continue

        raw_samples = results[c_name]
        for s in range(len(raw_samples)):
            ch0, ch1 = preprocess_28x28_to_2ch64(raw_samples[s])
            inputs[idx, 0] = (ch0 * 255.0).astype(np.uint8)
            inputs[idx, 1] = (ch1 * 255.0).astype(np.uint8)
            categories[idx] = c_id
            idx += 1

    np.savez_compressed(output_path, inputs=inputs[:idx], categories=categories[:idx])
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Sandbox DataGen] Built {idx} samples across {total_classes} classes in {time.time()-t0:.1f}s! File: {output_path} ({size_mb:.1f} MB)")
    return output_path

# ==============================================================================
# UNIVERSAL SANDBOX NEURAL NETWORK ARCHITECTURE
# ==============================================================================

class UniversalSandboxNet(nn.Module):
    """
    Lightweight 50-Class Sandbox Recognizer + 2.5D Curvature Depth Head:
    - Input: [1, 2, 64, 64] (Stroke + SDF)
    - Output:
      1. category_logits: [1, 51]
      2. latent_z: [1, 512]
      3. inflation_depth: [1, 1, 64, 64] (Continuous 2.5D depth heightmap for organic 3D voxel volume)
    """
    def __init__(self, num_classes=51, latent_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),   # -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),# -> 2x2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_latent = nn.Linear(512 * 2 * 2, latent_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Volumetric 2.5D Depth Inflation Trunk (Generates continuous depth heightmap from latent code)
        self.fc_depth = nn.Linear(latent_dim, 128 * 4 * 4)
        self.depth_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # -> 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat_flat = feat.view(feat.size(0), -1)
        z = self.fc_latent(feat_flat)
        logits = self.classifier(z)

        d_init = self.fc_depth(z).view(-1, 128, 4, 4)
        depth_map = self.depth_decoder(d_init)

        return logits, depth_map, z

# ==============================================================================
# TRAIN & EXPORT SANDBOX MODEL
# ==============================================================================

class SandboxDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.inputs = data['inputs']
        self.categories = data['categories']
        self.length = len(self.categories)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inp = torch.from_numpy(self.inputs[idx].astype(np.float32) / 255.0)
        cat = torch.tensor(self.categories[idx], dtype=torch.long)
        return inp, cat

def train_sandbox_model(dataset_path="Tools/DataGen/dataset_sandbox_50.npz", epochs=10, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("Training Universal 50-Class Sandbox Model on RTX 4050 GPU")
    print("=" * 70)

    dataset = SandboxDataset(dataset_path)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = UniversalSandboxNet(num_classes=NUM_SANDBOX_CLASSES, latent_dim=512).to(device)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_depth = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for b_in, b_cat in train_loader:
            b_in, b_cat = b_in.to(device), b_cat.to(device)
            optimizer.zero_grad()
            p_logits, p_depth, _ = model(b_in)

            # Target depth heightmap: smooth inflated profile bounded by stroke mask
            stroke_mask = b_in[:, 0:1]
            target_depth = stroke_mask * 0.85 # 2.5D normalized half-thickness

            loss_cls = criterion_cls(p_logits, b_cat)
            loss_d = criterion_depth(p_depth * stroke_mask, target_depth)
            loss = loss_cls + 2.0 * loss_d

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for b_in, b_cat in val_loader:
                b_in, b_cat = b_in.to(device), b_cat.to(device)
                p_logits, _, _ = model(b_in)
                pred_top5 = torch.topk(p_logits, 5, dim=1).indices
                correct_top1 += (pred_top5[:, 0] == b_cat).sum().item()
                correct_top5 += (pred_top5 == b_cat.unsqueeze(1)).any(dim=1).sum().item()
                total += len(b_cat)

        top1 = (correct_top1 / total) * 100.0
        top5 = (correct_top5 / total) * 100.0
        print(f"Epoch [{ep:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Top-1: {top1:5.2f}% | Val Top-5: {top5:5.2f}%")

    # Export ONNX for Unity Sentis
    os.makedirs("Assets/ML/Models", exist_ok=True)
    onnx_path = "Assets/ML/Models/draw2voxel_sandbox.onnx"
    print(f"\n[ONNX Export] Exporting Universal Sandbox Model to: {onnx_path}...")
    model.eval().to("cpu")
    dummy = torch.randn(1, 2, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_sketch_2ch'],
        output_names=['category_logits', 'inflation_depth', 'latent_z'],
        dynamo=False
    )
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[ONNX Export] Successfully exported {onnx_path} ({size_mb:.2f} MB)")

    # Export Sandbox Metadata JSON for Unity
    meta_path = "Assets/ML/Models/sandbox_categories.json"
    with open(meta_path, "w") as f:
        json.dump(SANDBOX_CLASSES, f, indent=2)
    print(f"[Sandbox Metadata] Saved 50-class catalog to: {meta_path}")

if __name__ == "__main__":
    npz_file = "Tools/DataGen/dataset_sandbox_50.npz"
    if not os.path.exists(npz_file):
        build_sandbox_dataset(npz_file, samples_per_class=1000)
    train_sandbox_model(npz_file, epochs=10, batch_size=64, lr=1e-3)
