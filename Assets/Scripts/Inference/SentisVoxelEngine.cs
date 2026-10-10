using System;
using System.Collections;
using UnityEngine;
using Unity.Collections;

#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
using Unity.InferenceEngine;
#else
using Unity.Sentis;
#endif

namespace VRDraw2Voxel.Inference
{
    /// <summary>
    /// Executes the 2D-to-3D Draw2Voxel ONNX model asynchronously using Unity Sentis / Inference Engine.
    /// Spreads tensor execution across multiple frames to protect VR frame budgets (72 FPS / 90 FPS).
    /// Compatible with both Unity 6 (com.unity.ai.inference) and earlier Sentis versions.
    /// </summary>
    public class SentisVoxelEngine : MonoBehaviour
    {
        [Header("Model Asset")]
        [SerializeField] private ModelAsset modelAsset;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;

        [Header("Inference Settings")]
        [Range(0.1f, 0.9f)]
        [SerializeField] private float occupancyThreshold = 0.5f;

        private Model _runtimeModel;

#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
        private Worker _worker;
#else
        private IWorker _worker;
#endif

        public const int InputWidth = 64;
        public const int InputHeight = 64;
        public const int VoxelDim = 32;
        public const int TotalVoxels = VoxelDim * VoxelDim * VoxelDim;

        public event Action<byte[]> OnVoxelGridReady;

        private void Awake()
        {
            InitializeEngine();
        }

        private void InitializeEngine()
        {
#if UNITY_EDITOR
            if (modelAsset == null)
            {
                modelAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/ML/Models/draw2voxel_v1.onnx");
                if (modelAsset != null)
                {
                    Debug.Log("[SentisVoxelEngine] Auto-loaded draw2voxel_v1.onnx from AssetDatabase.");
                }
            }
#endif

            if (modelAsset == null)
            {
                Debug.LogWarning("[SentisVoxelEngine] ModelAsset not assigned in inspector. Awaiting runtime assignment.");
                return;
            }

            try
            {
                _runtimeModel = ModelLoader.Load(modelAsset);
                _worker = new Worker(_runtimeModel, backend);
                Debug.Log($"[SentisVoxelEngine] Initialized inference worker on backend: {backend}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[SentisVoxelEngine] Backend {backend} failed ({ex.Message}), falling back to CPU.");
                backend = BackendType.CPU;
                _worker = new Worker(_runtimeModel, backend);
                Debug.Log($"[SentisVoxelEngine] Initialized inference worker on backend: {backend}");
            }
        }

        public void LoadModelAsset(ModelAsset asset)
        {
            modelAsset = asset;
            _worker?.Dispose();
            InitializeEngine();
        }

        /// <summary>
        /// Asynchronously processes a 64x64 binary raster input buffer and returns a 32x32x32 occupancy grid.
        /// </summary>
        public void PredictVoxelsAsync(float[] rasterBuffer, Action<byte[]> callback = null)
        {
            if (_worker == null)
            {
                Debug.LogError("[SentisVoxelEngine] Inference worker is not initialized.");
                return;
            }

            if (rasterBuffer == null || rasterBuffer.Length != InputWidth * InputHeight)
            {
                Debug.LogError($"[SentisVoxelEngine] Invalid raster buffer length: {rasterBuffer?.Length}. Expected {InputWidth * InputHeight}.");
                return;
            }

            StartCoroutine(ExecuteInferenceCoroutine(rasterBuffer, callback));
        }

        private IEnumerator ExecuteInferenceCoroutine(float[] rasterBuffer, Action<byte[]> callback)
        {
            TensorShape inputShape = new TensorShape(1, 1, InputHeight, InputWidth);
            float[] rawFloats;

#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
            // Unity 6 Inference Engine (Tensor<float>)
            using Tensor<float> inputTensor = new Tensor<float>(inputShape, rasterBuffer);
            _worker.Schedule(inputTensor);

            // Yield frame to let GPU compute without stalling VR render thread
            yield return null;

            Tensor<float> outputTensor = _worker.PeekOutput() as Tensor<float>;
            if (outputTensor == null)
            {
                Debug.LogError("[SentisVoxelEngine] Failed to retrieve output tensor.");
                yield break;
            }

            using Tensor<float> cpuTensor = outputTensor.ReadbackAndClone();
            rawFloats = cpuTensor != null ? cpuTensor.DownloadToArray() : new float[TotalVoxels];
#else
            // Legacy Sentis (TensorFloat)
            using TensorFloat inputTensor = new TensorFloat(inputShape, rasterBuffer);
            _worker.Schedule(inputTensor);

            yield return null;

            TensorFloat outputTensor = _worker.PeekOutput() as TensorFloat;
            if (outputTensor == null)
            {
                Debug.LogError("[SentisVoxelEngine] Failed to retrieve output tensor.");
                yield break;
            }

            var readback = outputTensor.ReadbackAndCloneAsync();
            while (!readback.isDone)
            {
                yield return null;
            }

            using TensorFloat cpuTensor = readback.result;
            rawFloats = cpuTensor.ToReadOnlyArray();
#endif

            // Threshold floats into binary voxel bytes (0 or 1)
            byte[] occupancyGrid = new byte[TotalVoxels];
            for (int i = 0; i < TotalVoxels; i++)
            {
                occupancyGrid[i] = rawFloats[i] >= occupancyThreshold ? (byte)1 : (byte)0;
            }

            Debug.Log($"[SentisVoxelEngine] Voxel inference complete. Generated {TotalVoxels} voxels.");
            OnVoxelGridReady?.Invoke(occupancyGrid);
            callback?.Invoke(occupancyGrid);
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
        }
    }
}
