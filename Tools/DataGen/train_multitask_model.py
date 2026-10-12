"""
Multi-Task 2D-to-3D Voxel Reconstruction & Perspective Disambiguation Model
AntiGravity Architecture Spike Deliverable

Features:
1. Multi-Task Heads:
   - Category Classifier Head (Softmax + Confidence Threshold tau)
   - View Angle Disambiguation Head (0: Front XY, 1: Side ZY, 2: Top XZ)
   - 4-Channel 3D Voxel Decoder (Occupancy + RGB Colors) at 64x64x64 resolution
2. Out-of-Distribution (OOD) Handling:
   - Centroid computation in latent space z in R^512
   - Nearest-centroid snapping or scribble rejection
3. Multi-Output ONNX Export for Unity Sentis / Inference Engine
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CATEGORY_NAMES = {
    0: "Sword",
    1: "Shield",
    2: "Blaster",
    3: "Axe",
    4: "TeddyBear",
    5: "RubberDuck",
    6: "ToyRobot",
    7: "ToyBunny",
    8: "Gamepad",
    9: "RetroTV",
    10: "GameBoy",
    11: "OOD_Reject"
}

VIEW_NAMES = {
    0: "Front (XY)",
    1: "Side (ZY)",
    2: "Top (XZ)"
}

# ==============================================================================
# MULTI-TASK NEURAL ARCHITECTURE
# ==============================================================================

class AntiGravityMultiTaskNet(nn.Module):
    def __init__(self, latent_dim=512, num_classes=12):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 1. 2D Convolutional Encoder: 2x64x64 -> Latent z in R^512
        # Channel 0: Normalized Stroke Mask, Channel 1: Truncated Distance Field (SDF)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # -> 2x2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_latent = nn.Linear(512 * 2 * 2, latent_dim)

        # 2. Classification Head A: Category Logits (11 canonical + 1 OOD_Reject)
        self.category_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_classes)
        )

        # 3. Classification Head B: Perspective View Disambiguation (Front vs Side vs Top)
        self.view_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 3)
        )

        # 4. 3D Deconvolutional Decoder: Latent z -> 64x64x64 Voxel Space
        # Shared trunk up to 32x32x32 resolution
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4 * 4)
        self.decoder_trunk = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), # -> 8x8x8
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 16x16x16
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 32x32x32
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Separate Head 1: Voxel Occupancy [B, 1, 64, 64, 64] (Sigmoid probability)
        self.head_occ = nn.Sequential(
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),   # -> 64x64x64
            nn.Sigmoid()
        )

        # Separate Head 2: Voxel RGB Colors [B, 3, 64, 64, 64] (Sigmoid [0..1] color space)
        self.head_rgb = nn.Sequential(
            nn.ConvTranspose3d(16, 3, kernel_size=4, stride=2, padding=1),   # -> 64x64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat_flat = feat.view(feat.size(0), -1)
        z = self.fc_latent(feat_flat)

        # Multi-task outputs
        category_logits = self.category_head(z)
        view_logits = self.view_head(z)

        vol_init = self.fc_dec(z).view(-1, 128, 4, 4, 4)
        vol_feat = self.decoder_trunk(vol_init)
        voxel_occ = self.head_occ(vol_feat)
        voxel_rgb = self.head_rgb(vol_feat)

        return voxel_occ, voxel_rgb, category_logits, view_logits, z

# ==============================================================================
# MULTI-TASK LOSS FUNCTION
# ==============================================================================

class MultiTaskVoxelLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()
        self.ce_cat = nn.CrossEntropyLoss()
        self.ce_view = nn.CrossEntropyLoss()

    def forward(self, pred_occ, pred_rgb, pred_cat, pred_view, tgt_vox, tgt_cat, tgt_view):
        tgt_occ = tgt_vox[:, 0:1]
        tgt_rgb = tgt_vox[:, 1:4]

        # 1. Occupancy Loss: Focal + Soft Dice + BCE
        eps = 1e-7
        p_clamped = pred_occ.clamp(eps, 1.0 - eps)
        pos_focal = - 0.75 * tgt_occ * ((1.0 - p_clamped) ** 2) * torch.log(p_clamped)
        neg_focal = - 0.25 * (1.0 - tgt_occ) * (p_clamped ** 2) * torch.log(1.0 - p_clamped)
        focal_loss = (pos_focal + neg_focal).mean()

        p_flat = pred_occ.reshape(-1)
        t_flat = tgt_occ.reshape(-1)
        inter = (p_flat * t_flat).sum()
        dice = (2. * inter + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        bce_loss = self.bce(pred_occ, tgt_occ)
        occ_loss = focal_loss + bce_loss + 2.5 * dice_loss

        # 2. Color Loss: Masked MSE only on solid voxels
        mask = (tgt_occ > 0.5).expand_as(pred_rgb)
        if mask.sum() > 0:
            rgb_loss = ((pred_rgb - tgt_rgb) ** 2 * mask.float()).sum() / (mask.sum().float() + 1e-6)
        else:
            rgb_loss = torch.tensor(0.0, device=pred_occ.device)

        # 3. Category & View Cross-Entropy Losses
        cat_loss = self.ce_cat(pred_cat, tgt_cat)

        # Only penalize view error on canonical objects (ignore for OOD random scribbles)
        canonical_mask = (tgt_cat != 11)
        if canonical_mask.sum() > 0:
            view_loss = self.ce_view(pred_view[canonical_mask], tgt_view[canonical_mask])
        else:
            view_loss = torch.tensor(0.0, device=pred_occ.device)

        total_loss = occ_loss + 0.5 * rgb_loss + 0.3 * cat_loss + 0.3 * view_loss
        return total_loss, occ_loss, rgb_loss, cat_loss, view_loss

# ==============================================================================
# DATASET & CENTROID EXTRACTION
# ==============================================================================

class TriViewDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.inputs = torch.from_numpy(data['inputs'])
        self.targets_voxels = torch.from_numpy(data['targets_voxels'])
        self.categories = torch.from_numpy(data['categories']).long()
        self.views = torch.from_numpy(data['views']).long()
        print(f"[Dataset] Loaded {len(self.inputs)} tri-view samples from {npz_path} (inputs shape: {self.inputs.shape}, dtype: {self.inputs.dtype})")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Normalize 2-channel input sketch to [0.0, 1.0]
        if self.inputs.dtype == torch.uint8:
            inp = self.inputs[idx].float() / 255.0
        else:
            inp = self.inputs[idx].float()

        # Channel 0: Occupancy (0 or 1), Channels 1..3: RGB [0.0, 1.0]
        if self.targets_voxels.dtype == torch.uint8:
            tgt_occ = self.targets_voxels[idx, 0:1].float()
            tgt_rgb = self.targets_voxels[idx, 1:4].float() / 255.0
            tgt_vox = torch.cat([tgt_occ, tgt_rgb], dim=0)
        else:
            tgt_vox = self.targets_voxels[idx].float()

        return (
            inp,
            tgt_vox,
            self.categories[idx],
            self.views[idx]
        )

def compute_latent_centroids(model, dataloader, device, num_classes=12):
    """Computes class centroids and radii in latent space z for OOD rejection & snapping."""
    model.eval()
    centroids = {k: [] for k in range(num_classes)}

    with torch.no_grad():
        for x, _, cat, _ in dataloader:
            x = x.to(device)
            _, _, _, _, z = model(x)
            z_cpu = z.cpu().numpy()
            cat_cpu = cat.numpy()
            for b in range(len(cat_cpu)):
                c_id = cat_cpu[b]
                centroids[c_id].append(z_cpu[b])

    centroid_data = {}
    for c_id in range(num_classes):
        if len(centroids[c_id]) == 0:
            continue
        vectors = np.array(centroids[c_id])
        mean_vec = np.mean(vectors, axis=0)
        dists = np.linalg.norm(vectors - mean_vec, axis=1)
        radius_95 = float(np.percentile(dists, 95))
        centroid_data[c_id] = {
            "name": CATEGORY_NAMES.get(c_id, f"Class_{c_id}"),
            "centroid": mean_vec.tolist(),
            "radius_95": radius_95,
            "mean_dist": float(np.mean(dists))
        }

    return centroid_data

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_multitask(dataset_path, epochs=15, batch_size=32, lr=1e-3, onnx_out="Assets/ML/Models/draw2voxel_multitask.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train Multi-Task] Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    dataset = TriViewDataset(dataset_path)
    val_size = max(int(0.12 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = AntiGravityMultiTaskNet(latent_dim=512, num_classes=12).to(device)
    criterion = MultiTaskVoxelLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    best_iou = 0.0

    print(f"[Train Multi-Task] Training for {epochs} epochs (Train: {train_size}, Val: {val_size})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for b_in, b_vox, b_cat, b_view in train_loader:
            b_in = b_in.to(device, non_blocking=True)
            b_vox = b_vox.to(device, non_blocking=True)
            b_cat = b_cat.to(device, non_blocking=True)
            b_view = b_view.to(device, non_blocking=True)

            optimizer.zero_grad()
            p_occ, p_rgb, p_cat, p_view, _ = model(b_in)
            loss, _, _, _, _ = criterion(p_occ, p_rgb, p_cat, p_view, b_vox, b_cat, b_view)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * b_in.size(0)

        scheduler.step()
        train_loss = total_train_loss / train_size

        # Validation
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        val_cat_correct = 0
        val_view_correct = 0
        total_val_samples = 0
        canonical_val_samples = 0

        with torch.no_grad():
            for v_in, v_vox, v_cat, v_view in val_loader:
                v_in = v_in.to(device, non_blocking=True)
                v_vox = v_vox.to(device, non_blocking=True)
                v_cat = v_cat.to(device, non_blocking=True)
                v_view = v_view.to(device, non_blocking=True)

                p_occ, p_rgb, p_cat, p_view, _ = model(v_in)
                loss, _, _, _, _ = criterion(p_occ, p_rgb, p_cat, p_view, v_vox, v_cat, v_view)
                val_loss += loss.item() * v_in.size(0)

                # Accuracies
                val_cat_correct += (p_cat.argmax(dim=1) == v_cat).sum().item()

                can_mask = (v_cat != 11)
                if can_mask.sum() > 0:
                    val_view_correct += (p_view[can_mask].argmax(dim=1) == v_view[can_mask]).sum().item()
                    canonical_val_samples += can_mask.sum().item()

                # Voxel IoU on occupancy channel
                p_bin = (p_occ[:, 0] > 0.5).float()
                t_bin = (v_vox[:, 0] > 0.5).float()
                for b in range(v_in.size(0)):
                    inter = (p_bin[b] * t_bin[b]).sum().item()
                    union = ((p_bin[b] + t_bin[b]) > 0.5).float().sum().item()
                    val_iou_sum += inter / union if union > 0 else 1.0

                total_val_samples += v_in.size(0)

        mean_val_loss = val_loss / total_val_samples
        mean_val_iou = val_iou_sum / total_val_samples
        cat_acc = val_cat_correct / total_val_samples
        view_acc = val_view_correct / max(canonical_val_samples, 1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {train_loss:.4f} (Val: {mean_val_loss:.4f}) | "
              f"Voxel IoU: {mean_val_iou*100:.1f}% | "
              f"Category Acc: {cat_acc*100:.1f}% | "
              f"View Acc: {view_acc*100:.1f}%")

        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            ckpt_path = onnx_out.replace(".onnx", ".pt")
            torch.save(model.state_dict(), ckpt_path)

    # Reload best weights and move to CPU
    ckpt_path = onnx_out.replace(".onnx", ".pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to("cpu").eval()
    print(f"\n[Train Multi-Task] Training complete! Best Voxel IoU: {best_iou*100:.2f}%")

    # Compute Latent Centroids for OOD Rejection & Snapping
    centroids = compute_latent_centroids(model, val_loader, "cpu", num_classes=12)
    centroid_json_path = onnx_out.replace(".onnx", "_centroids.json")
    with open(centroid_json_path, "w") as f:
        json.dump(centroids, f, indent=2)
    print(f"[Centroids] Exported latent centroids to {centroid_json_path}")

    # Export to multi-output ONNX (full multitask model)
    export_multitask_onnx(model, onnx_out)

    # Export standalone 3D decoder ONNX model (for canonical snapping and hybrid latent blending)
    decoder_out = onnx_out.replace(".onnx", "").replace("_multitask", "") + "_decoder.onnx"
    export_decoder_onnx(model, decoder_out)

class AntiGravityDecoderNet(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc_dec = nn.Linear(latent_dim, 128 * 4 * 4 * 4)
        self.decoder_trunk = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head_occ = nn.Sequential(
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.head_rgb = nn.Sequential(
            nn.ConvTranspose3d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        vol_init = self.fc_dec(z).view(-1, 128, 4, 4, 4)
        vol_feat = self.decoder_trunk(vol_init)
        voxel_occ = self.head_occ(vol_feat)
        voxel_rgb = self.head_rgb(vol_feat)
        return voxel_occ, voxel_rgb

def export_decoder_onnx(full_model, decoder_out):
    print(f"[ONNX Export] Exporting standalone 3D decoder to {decoder_out} for latent blending...")
    full_model.eval().to("cpu")

    decoder_net = AntiGravityDecoderNet(latent_dim=full_model.latent_dim)
    decoder_net.fc_dec.load_state_dict(full_model.fc_dec.state_dict())
    decoder_net.decoder_trunk.load_state_dict(full_model.decoder_trunk.state_dict())
    decoder_net.head_occ.load_state_dict(full_model.head_occ.state_dict())
    decoder_net.head_rgb.load_state_dict(full_model.head_rgb.state_dict())
    decoder_net.eval()

    dummy_z = torch.randn(1, full_model.latent_dim, device="cpu")
    os.makedirs(os.path.dirname(os.path.abspath(decoder_out)), exist_ok=True)

    torch.onnx.export(
        decoder_net,
        dummy_z,
        decoder_out,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['latent_z'],
        output_names=['voxel_occupancy', 'voxel_rgb'],
        dynamo=False
    )

    import onnx
    onnx_model = onnx.load(decoder_out)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(decoder_out) / (1024 * 1024)
    print(f"[ONNX Export] Successfully verified 3D decoder ONNX model! File size: {file_size_mb:.2f} MB")
    print("  Decoder Graph Inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input])
    print("  Decoder Graph Outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output])

def export_multitask_onnx(model, onnx_out):
    print(f"[ONNX Export] Exporting multi-task model to {onnx_out} for Unity Sentis...")
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 2, 64, 64, device="cpu")

    os.makedirs(os.path.dirname(os.path.abspath(onnx_out)), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_out,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_sketch_2ch'],
        output_names=['voxel_occupancy', 'voxel_rgb', 'category_logits', 'view_logits', 'latent_z'],
        dynamo=False
    )

    import onnx
    onnx_model = onnx.load(onnx_out)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_out) / (1024 * 1024)
    print(f"[ONNX Export] Successfully verified ONNX model! File size: {file_size_mb:.2f} MB")
    print("  Graph Inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input])
    print("  Graph Outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Tools/DataGen/dataset_triview_64.npz")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="Assets/ML/Models/draw2voxel_multitask.onnx")
    args = parser.parse_args()

    train_multitask(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, onnx_out=args.out)

