# VR Dynamic Draw-to-Voxel Architecture (`arch.md`)

## 1. Executive Summary & Prototype Scope

This architecture document outlines the technical design for a standalone VR prototype developed within a 1-week timeframe. The core loop consists of:

1. **Rock-Paper-Scissors (RPS)** resolution via OpenXR hand tracking.

2. **Dynamic 2D-to-3D Gesture Reconstruction**: On winning, the player draws a 2D gesture/silhouette in 3D space, which transforms into a physical $32 \times 32 \times 32$ voxelized weapon.

3. **Physical Interaction**: Automatic calculation of a grab anchor and rigid body collision to strike an NPC.

## 2. End-to-End Pipeline Architecture

```
+-----------------------------------------------------------------------------+
|                                OFFLINE PIPELINE                             |
+-----------------------------------------------------------------------------+
   Blender (Python Script)                  PyTorch / TF Training
   - Procedural Primitives                  - 2D Encoder (Conv2D)
   - Freestyle 2D Stroke Pass   ======>     - 3D Decoder (Conv3DTranspose)
   - 32^3 Voxel Binarization                - Loss: BCE / Dice Loss
                                                   ||
                                                   || Export
                                                   \/
                                              `model.onnx`
+-----------------------------------------------------------------------------+
|                             RUNTIME PIPELINE (UNITY)                        |
+-----------------------------------------------------------------------------+
   OpenXR Hand Tracking
            │
            ├─► RPS Gesture Classifier (Rock / Paper / Scissors)
            │      └─► [On Player Win] ──► Activate Drawing Canvas
            │
            ▼
   3D Stroke Sampler (World Space)
            │
            ▼
   Planar Projection & Rasterization ($64 \times 64$ Binary Tensor)
            │
            ▼
   Unity Sentis (`Worker.ExecuteAsync`)
            │
            ▼
   $32 \times 32 \times 32$ Occupancy Tensor Output
            │
            ▼
   Greedy Surface Extraction / Mesh Generation (DOTS / Burst Job)
            │
            ▼
   Grab Point Allocation (PCA / Bounding Box Base)
            │
            ▼
   `XRGrabInteractable` + `Rigidbody` ──► NPC Impact Resolution

```

## 3. Layer Specifications

### 3.1 Data Layer (`Data/`)

* **`StrokePointData`**: Vector3 position, stroke timestamp, and pressure/normal vectors.

* **`RasterInputBuffer`**: $1 \times 1 \times 64 \times 64$ float array representing the binary normalized drawing.

* **`VoxelGridState`**: Dense bitmask or flat `NativeArray<byte>` containing $32 \times 32 \times 32 = 32,768$ occupancy cells ($1$ for solid, $0$ for air).

* **`WeaponMetadata`**: Center of mass, principal inertial axes, bounding box extents, and calculated hilt offset.

### 3.2 Offline Synthetic Pipeline (`Tools/DataGen/`)

* **Generator**: Blender headless execution (`blender -b -P generate_dataset.py`).

* **Classes**: Weapons (swords, daggers), Blunt Tools (baseball bats, clubs, hammers), Shields/Toys.

* **Data Augmentation**: Random rotation ($\pm 15^\circ$), stroke jitter, synthetic line thinning/thickening, and partial stroke dropout.

* **Output Format**: HDF5 or `.npz` storage containing paired pairs:

  * Inputs: $(N, 1, 64, 64)$ float32 tensors.

  * Targets: $(N, 1, 32, 32, 32)$ float32 occupancy values.

### 3.3 Model Topology (`ML/`)

* **Input**: $[1, 1, 64, 64]$

* **Encoder**: 4 convolutional layers ($3 \times 3$ kernel, stride 2, LeakyReLU) flattening to latent vector $z \in \mathbb{R}^{256}$.

* **Latent Projection**: Linear layer from $256 \rightarrow 4 \times 4 \times 4 \times 64$ feature volume.

* **Decoder**: 3 Transposed Conv3D layers ($4 \times 4 \times 4$ kernel, stride 2) expanding to $[1, 1, 32, 32, 32]$.

* **Activation**: Sigmoid output clamped at threshold $\tau = 0.5$.

### 3.4 Runtime Inference Layer (`Scripts/Inference/`)

* Managed by **Unity Sentis** runtime worker.

* Asynchronous dispatch across multiple frames to avoid stalling the render thread on standalone VR chipsets:

  ```
  // Run asynchronously without blocking VR frame render
  worker.Schedule(inputTensor);
  
  ```

* Output is copied directly into a `NativeArray` for Burst job processing.

### 3.5 Mesh Generation & Grip Heuristic (`Scripts/Mesh/`)

1. **Surface Meshing**: Burst-compiled surface quad extractor scans internal boundaries where cell transitions occur ($1 \rightarrow 0$).

2. **Greedy Merging**: Merges coplanar quads to minimize vertex counts below 2,000 tris per weapon.

3. **Hilt Discovery**:

   * Evaluates the bounding box extents $E = (e_x, e_y, e_z)$.

   * Selects the axis of maximum dimension $A_{\text{max}}$.

   * Designates the negative extremal boundary region ($0\% \text{ to } 20\%$ along $A_{\text{max}}$) as the handle anchor.

   * Inserts an OpenXR Grab Pose target at this anchor position.

## 4. VR Performance Budget (Meta Quest Target: 72 FPS / 13.88ms budget)

| Phase | Target Execution Time | Strategy | 
| ----- | ----- | ----- | 
| **RPS Tracking** | $< 0.5\text{ ms}$ / frame | Lightweight heuristic joint angle analysis. | 
| **Stroke Rasterization** | $< 1.5\text{ ms}$ (one-off) | Off-screen 2D blit / software scanline in compute shader. | 
| **Model Inference (Sentis)** | $10 - 25\text{ ms}$ (sliced) | Asynchronous GPU execution spread over 2–4 frames. | 
| **Voxel-to-Mesh Construction** | $< 3.0\text{ ms}$ | Unity C# Job System + Burst compilation. | 
| **Collider Generation** | $< 2.0\text{ ms}$ | Simplified compound box colliders instead of raw MeshCollider. | 

## 5. Directory Structure

```
Assets/
├── ML/
│   └── Models/
│       └── draw2voxel_v1.onnx
├── Scripts/
│   ├── Drawing/
│   │   ├── StrokeCapture.cs
│   │   └── DrawingCanvasProjector.cs
│   ├── Inference/
│   │   └── SentisVoxelEngine.cs
│   ├── Interaction/
│   │   ├── RPSDetector.cs
│   │   └── AutoGrabAttacher.cs
│   └── VoxelMesh/
│       ├── GreedyVoxelMesher.cs
│       └── VoxelJobs.cs
└── Tools/
    └── DataGen/
        ├── generate_dataset.py
        └── train_model.py

```