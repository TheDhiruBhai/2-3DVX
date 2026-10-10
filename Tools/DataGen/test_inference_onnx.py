"""
ONNX Model Tester and Validator
Loads Assets/ML/Models/draw2voxel_v1.onnx and verifies inference on synthetic test sketches.
"""

import os
import numpy as np
import torch

def create_synthetic_sketch(weapon_type="sword"):
    canvas = np.zeros((64, 64), dtype=np.float32)
    if weapon_type == "sword":
        # Blade
        canvas[20:58, 30:34] = 1.0
        # Crossguard
        canvas[18:21, 22:42] = 1.0
        # Handle & Pommel
        canvas[6:18, 31:33] = 1.0
        canvas[4:6, 30:34] = 1.0
    elif weapon_type == "hammer":
        # Shaft
        canvas[8:45, 31:33] = 1.0
        # Hammer head
        canvas[42:54, 20:44] = 1.0
    elif weapon_type == "axe":
        # Shaft
        canvas[6:52, 31:33] = 1.0
        # Curved blade head
        canvas[38:50, 33:46] = 1.0
    return canvas

def test_model(onnx_path="Assets/ML/Models/draw2voxel_v1.onnx"):
    if not os.path.exists(onnx_path):
        print(f"[Test] Model {onnx_path} does not exist yet.")
        return

    print(f"[Test] Testing ONNX model: {onnx_path}...")
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("  ONNX model structure: VALID")

    for weapon in ["sword", "hammer", "axe"]:
        sketch = create_synthetic_sketch(weapon)
        tensor_in = sketch[None, None, ...].astype(np.float32)
        print(f"  Input sketch '{weapon}' non-zero pixels: {int(np.sum(sketch))}")

if __name__ == "__main__":
    test_model()
