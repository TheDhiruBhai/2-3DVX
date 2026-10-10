"""
Training & ONNX Export Pipeline for Multi-Category Draw2Voxel Model
Supports:
- Category 0: Weapons (Melee & Firearms)
- Category 1: Toys (Animals & Living Figures)
- Category 2: Electronics & Gadgets
Topology:
- Input: [B, 1, 64, 64]
- Encoder: 4 Conv2D blocks (stride 2) -> Latent z in R^256
- Latent Projection: Linear 256 -> (64 * 4 * 4 * 4)
- Decoder: 3 ConvTranspose3D blocks -> [B, 1, 32, 32, 32]
- Loss: Combined BCE + Soft Dice Loss for sparse 3D occupancy
- Export: Assets/ML/Models/draw2voxel_v1.onnx (Static shape for Unity Sentis)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATEGORY_NAMES = {
    0: "Weapons",
    1: "Toys",
    2: "Electronics"
}

class Draw2VoxelNet(nn.Module):
    def __init__(self, latent_dim=256):
        super(Draw2VoxelNet, self).__init__()
        self.latent_dim = latent_dim
        
        # 1. 2D Encoder: [B, 1, 64, 64] -> Latent z
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> 32x32
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
        )
        
        # Flatten: 256 * 4 * 4 = 4096 -> 256
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        
        # 2. Latent Projection: 256 -> 64 * 4 * 4 * 4 = 4096
        self.fc_dec = nn.Linear(latent_dim, 64 * 4 * 4 * 4)
        
        # 3. 3D Decoder: [B, 64, 4, 4, 4] -> [B, 1, 32, 32, 32]
        # 4x4x4 -> 8x8x8 -> 16x16x16 -> 32x32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), # -> 8x8x8
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1), # -> 16x16x16
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),  # -> 32x32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat_flat = feat.view(feat.size(0), -1)
        z = self.fc_enc(feat_flat)
        
        vol_init = self.fc_dec(z)
        vol_init = vol_init.view(vol_init.size(0), 64, 4, 4, 4)
        
        voxels = self.decoder(vol_init)
        return voxels

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice
        
        return bce_loss + 2.0 * dice_loss

class VoxelDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.inputs = torch.from_numpy(data['inputs']).float()
        self.targets = torch.from_numpy(data['targets']).float()
        if 'categories' in data:
            self.categories = torch.from_numpy(data['categories']).long()
        else:
            self.categories = torch.zeros(len(self.inputs), dtype=torch.long)
        print(f"[Dataset] Loaded {len(self.inputs)} multi-category samples from {npz_path}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.categories[idx]

def train_model(dataset_path, epochs=20, batch_size=64, lr=1e-3, onnx_out="Assets/ML/Models/draw2voxel_v1.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    dataset = VoxelDataset(dataset_path)
    val_size = max(int(0.1 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = Draw2VoxelNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    best_iou = 0.0
    print(f"[Train] Training for {epochs} epochs (Train: {train_size}, Val: {val_size})...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_in, batch_target, _ in train_loader:
            batch_in = batch_in.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(batch_in)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_in.size(0)
            
        scheduler.step()
        train_loss = running_loss / train_size
        
        # Validation across categories
        model.eval()
        val_loss = 0.0
        cat_iou_sum = {0: 0.0, 1: 0.0, 2: 0.0}
        cat_counts = {0: 0, 1: 0, 2: 0}
        
        with torch.no_grad():
            for val_in, val_target, val_cat in val_loader:
                val_in = val_in.to(device, non_blocking=True)
                val_target = val_target.to(device, non_blocking=True)
                pred = model(val_in)
                loss = criterion(pred, val_target)
                val_loss += loss.item() * val_in.size(0)
                
                # Per-sample IoU
                pred_bin = (pred > 0.5).float()
                for b in range(val_in.size(0)):
                    p_b = pred_bin[b]
                    t_b = val_target[b]
                    c_id = val_cat[b].item()
                    
                    inter = (p_b * t_b).sum().item()
                    union = ((p_b + t_b) > 0.5).float().sum().item()
                    sample_iou = inter / union if union > 0 else 1.0
                    cat_iou_sum[c_id] += sample_iou
                    cat_counts[c_id] += 1
                
        val_loss = val_loss / val_size
        cat_mean_iou = {c: (cat_iou_sum[c] / max(1, cat_counts[c])) for c in cat_iou_sum}
        mean_iou = sum(cat_iou_sum.values()) / max(1, sum(cat_counts.values()))
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"Mean IoU: {mean_iou*100:.1f}% | "
              f"Weapons: {cat_mean_iou[0]*100:.1f}% | "
              f"Toys: {cat_mean_iou[1]*100:.1f}% | "
              f"Electronics: {cat_mean_iou[2]*100:.1f}%")
        
        if mean_iou > best_iou:
            best_iou = mean_iou
            ckpt_path = onnx_out.replace(".onnx", ".pt")
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n[Train] Training complete! Best Validation IoU: {best_iou*100:.2f}%")
    
    # Reload best weights
    ckpt_path = onnx_out.replace(".onnx", ".pt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    
    # Export to ONNX with static shape
    export_to_onnx(model, onnx_out)
    
    # Generate visual validation preview
    visualize_multi_category_predictions(model, val_loader, "Tools/DataGen/multicategory_samples_preview.png")

def export_to_onnx(model, onnx_out):
    print(f"[ONNX Export] Exporting model to {onnx_out} for Unity Sentis...")
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 1, 64, 64, device="cpu")
    
    os.makedirs(os.path.dirname(os.path.abspath(onnx_out)), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out,
        export_params=True,
        opset_version=14, # Sentis fully supports opset 14
        do_constant_folding=True,
        input_names=['input_sketch'],
        output_names=['output_voxels'],
        dynamo=False
    )
    
    import onnx
    onnx_model = onnx.load(onnx_out)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_out) / (1024 * 1024)
    print(f"[ONNX Export] Successfully verified ONNX model! File size: {file_size_mb:.2f} MB")
    print(f"  Input: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input]}")
    print(f"  Output: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output]}")

def visualize_multi_category_predictions(model, val_loader, out_img_path):
    print(f"[Preview] Generating preview image at {out_img_path}...")
    model.eval().to("cpu")
    
    # Collect 2 samples from each category
    samples = {0: [], 1: [], 2: []}
    for val_in, val_target, val_cat in val_loader:
        for b in range(val_in.size(0)):
            cid = val_cat[b].item()
            if len(samples[cid]) < 2:
                samples[cid].append((val_in[b:b+1], val_target[b:b+1]))
        if all(len(s) >= 2 for s in samples.values()):
            break
            
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    cat_keys = [0, 1, 2]
    
    for row_idx, cid in enumerate(cat_keys):
        cat_name = CATEGORY_NAMES[cid]
        for col_pair in range(2):
            if col_pair >= len(samples[cid]):
                continue
            inp, tgt = samples[cid][col_pair]
            with torch.no_grad():
                pred = model(inp).numpy()[0, 0]
            inp_img = inp.numpy()[0, 0]
            tgt_proj = tgt.numpy()[0, 0].max(axis=2)
            pred_proj = (pred > 0.5).max(axis=2)
            
            # Left subcol: 2D Input
            ax_in = axes[row_idx, col_pair * 2]
            ax_in.imshow(inp_img.T, cmap='gray_r', origin='lower')
            ax_in.set_title(f"[{cat_name}] 2D Input", fontsize=10)
            ax_in.axis('off')
            
            # Right subcol: 3D Reconstructed Projection
            ax_out = axes[row_idx, col_pair * 2 + 1]
            ax_out.imshow(pred_proj.T, cmap='viridis', origin='lower')
            ax_out.set_title(f"3D Voxel Reconstructed", fontsize=10)
            ax_out.axis('off')
            
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=120)
    plt.close()
    print(f"[Preview] Successfully saved {out_img_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Tools/DataGen/dataset_multicategory.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="Assets/ML/Models/draw2voxel_v1.onnx")
    args = parser.parse_args()

    train_model(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, onnx_out=args.out)
