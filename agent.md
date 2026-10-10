# Agent Context: VR Dynamic 2D-to-3D Gesture Combat Prototype

## Project Summary
A VR prototype where a player plays Rock-Paper-Scissors via hand gestures. Upon winning a round, the player draws a 2D gesture or weapon outline in 3D space, which dynamically converts into a physical 3D voxelized mesh to strike the NPC.

## Architecture Guidelines
* **Engine & Runtime**: Unity (OpenXR / Meta Quest).
* **Code Paradigm**: Data-Oriented Technology Stack (DOTS / ECS) with Burst compilation where applicable; hybrid Unity Sentis for deep learning inference.
* **Inference Pipeline**: 
  1. Capture user stroke points in Unity $\rightarrow$ normalize to 2D binary raster grid ($64 \times 64$).
  2. Pass tensor to Unity Sentis running an ONNX model.
  3. Model outputs a $32 \times 32 \times 32$ occupancy voxel grid.
  4. Extract surface voxels / run greedy meshing $\rightarrow$ instantiate collision mesh.
  5. Compute bounding box / principal axes to determine default handle/grab transform automatically.

## Pipeline Modules
1. **Synthetic Data Generation (Blender API)**:
   * Procedural Blender Python scripts generate 3D reference primitives across categories (weapons, bats, blades, blunt tools).
   * Script renders orthogonal silhouette line drawings (Freestyle/workbench edge pass) to form input-output paired datasets.
2. **Model Training (PyTorch / Scikit/TF)**:
   * Supervised 2D-to-3D reconstruction (encoder-decoder CNN or 3D U-Net).
   * Export architecture to `.onnx` with fixed/dynamic batch size for Sentis.
3. **Unity Integration**:
   * XR Hand Tracking interaction layer for RPS gesture recognition.
   * Runtime stroke capture $\rightarrow$ Sentis engine execution $\rightarrow$ dynamic mesh assignment with XR Grab Interactable.

## Agent Instructions
* Always adhere to clean, decoupled modules.
* Avoid heavy allocations in the per-frame game loop; run voxel extraction and tensor dispatch asynchronously to respect VR frame budgets ($72\text{ Hz} / 90\text{ Hz}$).
* Keep code snippets complete, typed, and idiomatic.