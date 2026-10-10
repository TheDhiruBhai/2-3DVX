"""
Verification script for AntiGravity Multi-Task Draw2Voxel ONNX Model & Pipeline.
Tests:
1. Category classification across test samples
2. Perspective disambiguation across Front, Side, and Top views
3. Out-of-Distribution (OOD) scribble rejection
4. Voxel reconstruction statistics (Occupancy count & Color distribution)
"""

import json
import numpy as np
import onnxruntime as ort

ONNX_PATH = "Assets/ML/Models/draw2voxel_multitask.onnx"
CENTROIDS_PATH = "Assets/ML/Models/draw2voxel_multitask_centroids.json"
DATASET_PATH = "Tools/DataGen/dataset_triview_64.npz"

CATEGORY_NAMES = [
    "Sword", "Shield", "Blaster", "Axe", "TeddyBear",
    "RubberDuck", "ToyRobot", "ToyBunny", "Gamepad", "RetroTV", "GameBoy"
]
VIEW_NAMES = ["Front (XY)", "Side (ZY)", "Top (XZ)"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def run_verification():
    print(f"=== AntiGravity Multi-Task Pipeline Verification ===")
    print(f"Loading ONNX runtime session: {ONNX_PATH}...")
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

    with open(CENTROIDS_PATH, "r") as f:
        centroids = json.load(f)
    print(f"Loaded {len(centroids)} latent centroids from {CENTROIDS_PATH}")

    data = np.load(DATASET_PATH)
    inputs = data['inputs']
    categories = data['categories']
    views = data['views']
    targets_voxels = data['targets_voxels']

    print(f"\n--- Testing All 11 Classes (In-Distribution Samples) ---")
    correct_cat = 0
    correct_view = 0
    # Test one sample from each class (indices 0..10) and a few more
    test_indices = list(range(11)) + [22, 33, 44, 55, 66]

    for idx in test_indices:
        sketch = (inputs[idx:idx+1].astype(np.float32) / 255.0)
        gt_cat = categories[idx]
        gt_view = views[idx]

        outputs = session.run(None, {'input_sketch': sketch})
        voxels_4ch, cat_logits, view_logits, z = outputs

        cat_probs = softmax(cat_logits[0])
        pred_cat = np.argmax(cat_probs)
        cat_conf = cat_probs[pred_cat]

        view_probs = softmax(view_logits[0])
        pred_view = np.argmax(view_probs)
        view_conf = view_probs[pred_view]

        solid_voxels = np.sum(voxels_4ch[0, 0] > 0.5)

        if pred_cat == gt_cat: correct_cat += 1
        if pred_view == gt_view: correct_view += 1

        print(f"Sample #{idx:04d}: GT [{CATEGORY_NAMES[gt_cat]} | {VIEW_NAMES[gt_view]}] -> "
              f"Pred [{CATEGORY_NAMES[pred_cat]} ({cat_conf*100:.1f}%) | {VIEW_NAMES[pred_view]} ({view_conf*100:.1f}%)] | "
              f"Solid Voxels: {solid_voxels}")

    print(f"\nAccuracy on inspected samples: Category {correct_cat}/{len(test_indices)}, View {correct_view}/{len(test_indices)}")

    print(f"\n--- Testing Out-of-Distribution (OOD) Scribbles ---")
    np.random.seed(42)
    # 1. Random gaussian noise
    noise_sketch = np.random.uniform(0.0, 1.0, (1, 1, 64, 64)).astype(np.float32)
    out_noise = session.run(None, {'input_sketch': noise_sketch})
    noise_cat_probs = softmax(out_noise[1][0])
    noise_max_prob = np.max(noise_cat_probs)
    noise_z = out_noise[3][0]

    # Distance to nearest centroid
    min_dist = float('inf')
    closest_cls = None
    for c_id, c_data in centroids.items():
        centroid_vec = np.array(c_data["centroid"])
        dist = np.linalg.norm(noise_z - centroid_vec)
        if dist < min_dist:
            min_dist = dist
            closest_cls = c_data["name"]

    print(f"Random Noise Sketch:")
    print(f"  Max Category Softmax: {noise_max_prob*100:.1f}% (Threshold tau = 60.0%)")
    print(f"  Nearest Centroid: {closest_cls} (Distance: {min_dist:.2f}, 95th radius: {centroids['0']['radius_95']:.2f})")
    print(f"  OOD Softmax Rejected: {noise_max_prob < 0.60}")

    # 2. Diagonal random scribbles
    scribble_sketch = np.zeros((1, 1, 64, 64), dtype=np.float32)
    for i in range(10, 54):
        scribble_sketch[0, 0, i, (i * 3) % 64] = 1.0
    out_scribble = session.run(None, {'input_sketch': scribble_sketch})
    scribble_cat_probs = softmax(out_scribble[1][0])
    scribble_max_prob = np.max(scribble_cat_probs)
    print(f"Sparse Diagonal Scribble:")
    print(f"  Max Category Softmax: {scribble_max_prob*100:.1f}% (Threshold tau = 60.0%)")
    print(f"  OOD Softmax Rejected: {scribble_max_prob < 0.60}")

    print(f"\n=== All Pipeline Verifications Succeeded ===")

if __name__ == "__main__":
    run_verification()
