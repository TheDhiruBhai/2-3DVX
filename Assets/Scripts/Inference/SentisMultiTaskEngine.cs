using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;

#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
using Unity.InferenceEngine;
#else
using Unity.Sentis;
#endif

namespace VRDraw2Voxel.Inference
{
    public enum ReconstructedCategory
    {
        Sword = 0,
        Shield = 1,
        Blaster = 2,
        Axe = 3,
        TeddyBear = 4,
        RubberDuck = 5,
        ToyRobot = 6,
        ToyBunny = 7,
        Gamepad = 8,
        RetroTV = 9,
        GameBoy = 10,
        Unknown = -1
    }

    public enum PerspectiveView
    {
        Front_XY = 0,
        Side_ZY = 1,
        Top_XZ = 2
    }

    [System.Serializable]
    public struct MultiTaskInferenceResult
    {
        public ReconstructedCategory category;
        public float categoryConfidence;
        public PerspectiveView detectedView;
        public float viewConfidence;
        public bool isOutOfDistribution;
        public byte[] occupancyGrid; // 64x64x64 binary (0 or 1)
        public Color32[] voxelColors; // 64x64x64 RGB colors
        public float latencyMs;
    }

    /// <summary>
    /// Executes the multi-task AntiGravity Draw2Voxel ONNX model in Unity Sentis / Inference Engine.
    /// Disambiguates perspective (Front/Side/Top), classifies object category with OOD rejection,
    /// and extracts a 64x64x64 colored 3D voxel grid within VR frame budgets (<15 ms).
    /// </summary>
    public class SentisMultiTaskEngine : MonoBehaviour
    {
        [Header("Model Configuration")]
        [SerializeField] private ModelAsset multiTaskModelAsset;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;

        [Header("OOD & Perspective Rejection")]
        [Tooltip("Minimum softmax probability required to accept drawing as valid shape.")]
        [Range(0.3f, 0.95f)]
        [SerializeField] private float oodConfidenceThreshold = 0.60f;

        [Header("Occupancy & Island Post-Processing")]
        [Tooltip("Binary occupancy threshold for voxel decoder. Higher values eliminate boundary floating dust.")]
        [Range(0.2f, 0.85f)]
        [SerializeField] private float occupancyThreshold = 0.58f;

        [Tooltip("Enable 3D connected component analysis to reject disconnected voxel islands.")]
        [SerializeField] private bool filterDisconnectedIslands = true;

        [Tooltip("Minimum voxel cluster size to retain. Disconnected islands smaller than this are purged.")]
        [Range(1, 32)]
        [SerializeField] private int minClusterSize = 8;

        private Model _runtimeModel;
#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
        private Worker _worker;
#else
        private IWorker _worker;
#endif

        public const int InputRes = 64;
        public const int VoxelDim = 64;
        public const int TotalVoxels = VoxelDim * VoxelDim * VoxelDim; // 262,144 voxels

        public event Action<MultiTaskInferenceResult> OnInferenceComplete;

        private void Awake()
        {
            InitializeEngine();
        }

        public void InitializeEngine()
        {
#if UNITY_EDITOR
            if (multiTaskModelAsset == null)
            {
                multiTaskModelAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/ML/Models/draw2voxel_multitask.onnx");
            }
#endif

            if (multiTaskModelAsset == null)
            {
                Debug.LogWarning("[SentisMultiTaskEngine] Multi-task ModelAsset not assigned.");
                return;
            }

            try
            {
                _runtimeModel = ModelLoader.Load(multiTaskModelAsset);
                _worker = new Worker(_runtimeModel, backend);
                Debug.Log($"[SentisMultiTaskEngine] Initialized multi-task worker on backend: {backend}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[SentisMultiTaskEngine] Backend {backend} failed ({ex.Message}), falling back to CPU.");
                backend = BackendType.CPU;
                _worker = new Worker(_runtimeModel, backend);
            }
        }

        /// <summary>
        /// Dispatches a 64x64 float input sketch to the multi-task network.
        /// Asynchronously yields across frames to maintain VR frame rates.
        /// </summary>
        public void PredictMultiTaskAsync(float[] rasterBuffer, Action<MultiTaskInferenceResult> callback = null)
        {
            if (_worker == null)
            {
                Debug.LogError("[SentisMultiTaskEngine] Inference worker is not initialized.");
                return;
            }

            if (rasterBuffer == null || rasterBuffer.Length != InputRes * InputRes)
            {
                Debug.LogError($"[SentisMultiTaskEngine] Invalid raster buffer length: {rasterBuffer?.Length}. Expected {InputRes * InputRes}.");
                return;
            }

            StartCoroutine(ExecuteMultiTaskCoroutine(rasterBuffer, callback));
        }

        private IEnumerator ExecuteMultiTaskCoroutine(float[] rasterBuffer, Action<MultiTaskInferenceResult> callback)
        {
            float startTime = Time.realtimeSinceStartup;
            TensorShape inputShape = new TensorShape(1, 1, InputRes, InputRes);

#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
            using Tensor<float> inputTensor = new Tensor<float>(inputShape, rasterBuffer);
            _worker.Schedule(inputTensor);

            // Yield frame so GPU compute executes without blocking VR render thread
            yield return null;

            // Retrieve multi-output tensors
            Tensor<float> voxelsTensor = _worker.PeekOutput("output_voxels_4ch") as Tensor<float>;
            Tensor<float> catLogitsTensor = _worker.PeekOutput("category_logits") as Tensor<float>;
            Tensor<float> viewLogitsTensor = _worker.PeekOutput("view_logits") as Tensor<float>;

            if (voxelsTensor == null || catLogitsTensor == null || viewLogitsTensor == null)
            {
                Debug.LogError("[SentisMultiTaskEngine] Failed to retrieve one or more output tensors.");
                yield break;
            }

            // Readback logits (very small tensors: 3 floats each)
            using Tensor<float> catCpu = catLogitsTensor.ReadbackAndClone();
            using Tensor<float> viewCpu = viewLogitsTensor.ReadbackAndClone();
            float[] catLogits = catCpu.DownloadToArray();
            float[] viewLogits = viewCpu.DownloadToArray();

            // Readback 4-channel 64^3 voxel tensor
            using Tensor<float> voxCpu = voxelsTensor.ReadbackAndClone();
            float[] rawVoxels4Ch = voxCpu.DownloadToArray();
#else
            using TensorFloat inputTensor = new TensorFloat(inputShape, rasterBuffer);
            _worker.Schedule(inputTensor);
            yield return null;

            TensorFloat voxelsTensor = _worker.PeekOutput("output_voxels_4ch") as TensorFloat;
            TensorFloat catLogitsTensor = _worker.PeekOutput("category_logits") as TensorFloat;
            TensorFloat viewLogitsTensor = _worker.PeekOutput("view_logits") as TensorFloat;

            float[] catLogits = catLogitsTensor.ToReadOnlyArray();
            float[] viewLogits = viewLogitsTensor.ToReadOnlyArray();
            float[] rawVoxels4Ch = voxelsTensor.ToReadOnlyArray();
#endif

            // 1. Evaluate Category Softmax & OOD Rejection
            float[] catSoftmax = ComputeSoftmax(catLogits);
            int bestCatIdx = 0;
            float bestCatProb = catSoftmax[0];
            for (int i = 1; i < catSoftmax.Length; i++)
            {
                if (catSoftmax[i] > bestCatProb)
                {
                    bestCatProb = catSoftmax[i];
                    bestCatIdx = i;
                }
            }

            bool isOOD = bestCatProb < oodConfidenceThreshold;
            ReconstructedCategory cat = isOOD ? ReconstructedCategory.Unknown : (ReconstructedCategory)bestCatIdx;

            // 2. Evaluate View Disambiguation Softmax
            float[] viewSoftmax = ComputeSoftmax(viewLogits);
            int bestViewIdx = 0;
            float bestViewProb = viewSoftmax[0];
            for (int i = 1; i < viewSoftmax.Length; i++)
            {
                if (viewSoftmax[i] > bestViewProb)
                {
                    bestViewProb = viewSoftmax[i];
                    bestViewIdx = i;
                }
            }
            PerspectiveView detectedView = (PerspectiveView)bestViewIdx;

            // 3. Process 64^3 Occupancy & Color Palettes
            // Channel 0: Occupancy, Channel 1: R, Channel 2: G, Channel 3: B
            byte[] occupancyGrid = new byte[TotalVoxels];
            Color32[] voxelColors = new Color32[TotalVoxels];

            int chStride = TotalVoxels;
            int rOffset = chStride;
            int gOffset = chStride * 2;
            int bOffset = chStride * 3;

            for (int i = 0; i < TotalVoxels; i++)
            {
                float occ = rawVoxels4Ch[i];
                if (occ >= occupancyThreshold)
                {
                    occupancyGrid[i] = 1;
                    byte r = (byte)Mathf.Clamp((int)(rawVoxels4Ch[rOffset + i] * 255.0f), 0, 255);
                    byte g = (byte)Mathf.Clamp((int)(rawVoxels4Ch[gOffset + i] * 255.0f), 0, 255);
                    byte b = (byte)Mathf.Clamp((int)(rawVoxels4Ch[bOffset + i] * 255.0f), 0, 255);
                    voxelColors[i] = new Color32(r, g, b, 255);
                }
                else
                {
                    occupancyGrid[i] = 0;
                }
            }

            // 4. Reject disconnected floating voxel islands (connected component analysis)
            if (filterDisconnectedIslands && minClusterSize > 1)
            {
                int prunedVoxels = FilterDisconnectedIslands(occupancyGrid, voxelColors, minClusterSize);
                if (prunedVoxels > 0)
                {
                    Debug.Log($"[SentisMultiTaskEngine] Pruned {prunedVoxels} disconnected floating island voxels (< {minClusterSize} cluster size).");
                }
            }

            float latency = (Time.realtimeSinceStartup - startTime) * 1000f;
            MultiTaskInferenceResult result = new MultiTaskInferenceResult
            {
                category = cat,
                categoryConfidence = bestCatProb,
                detectedView = detectedView,
                viewConfidence = bestViewProb,
                isOutOfDistribution = isOOD,
                occupancyGrid = occupancyGrid,
                voxelColors = voxelColors,
                latencyMs = latency
            };

            Debug.Log($"[SentisMultiTaskEngine] Done: Cat={cat} ({bestCatProb*100:F1}%), View={detectedView} ({bestViewProb*100:F1}%), OOD={isOOD}, Latency={latency:F1}ms");
            OnInferenceComplete?.Invoke(result);
            callback?.Invoke(result);
        }

        /// <summary>
        /// Performs 3D connected component analysis (BFS flood fill) to eliminate floating voxel dust
        /// and disconnected artifacts smaller than minSize voxels.
        /// </summary>
        private int FilterDisconnectedIslands(byte[] occupancy, Color32[] colors, int minSize)
        {
            if (minSize <= 1) return 0;

            bool[] visited = new bool[TotalVoxels];
            int prunedCount = 0;
            Queue<int> queue = new Queue<int>(256);
            List<int> currentCluster = new List<int>(minSize * 2);

            // 6-connectivity neighbor offsets in 3D (X*4096 + Y*64 + Z)
            int[] dx = { 1, -1, 0, 0, 0, 0 };
            int[] dy = { 0, 0, 1, -1, 0, 0 };
            int[] dz = { 0, 0, 0, 0, 1, -1 };

            for (int i = 0; i < TotalVoxels; i++)
            {
                if (occupancy[i] == 0 || visited[i]) continue;

                currentCluster.Clear();
                queue.Clear();

                visited[i] = true;
                queue.Enqueue(i);
                currentCluster.Add(i);

                while (queue.Count > 0)
                {
                    int currIdx = queue.Dequeue();
                    int cx = currIdx / (VoxelDim * VoxelDim);
                    int rem = currIdx % (VoxelDim * VoxelDim);
                    int cy = rem / VoxelDim;
                    int cz = rem % VoxelDim;

                    for (int n = 0; n < 6; n++)
                    {
                        int nx = cx + dx[n];
                        int ny = cy + dy[n];
                        int nz = cz + dz[n];

                        if (nx < 0 || nx >= VoxelDim || ny < 0 || ny >= VoxelDim || nz < 0 || nz >= VoxelDim)
                            continue;

                        int nIdx = nx * (VoxelDim * VoxelDim) + ny * VoxelDim + nz;
                        if (occupancy[nIdx] > 0 && !visited[nIdx])
                        {
                            visited[nIdx] = true;
                            queue.Enqueue(nIdx);
                            currentCluster.Add(nIdx);
                        }
                    }
                }

                if (currentCluster.Count < minSize)
                {
                    for (int c = 0; c < currentCluster.Count; c++)
                    {
                        int pruneIdx = currentCluster[c];
                        occupancy[pruneIdx] = 0;
                        colors[pruneIdx] = default;
                        prunedCount++;
                    }
                }
            }

            return prunedCount;
        }

        private float[] ComputeSoftmax(float[] logits)
        {
            float maxLogit = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > maxLogit) maxLogit = logits[i];
            }

            float sumExp = 0f;
            float[] exp = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                exp[i] = Mathf.Exp(logits[i] - maxLogit);
                sumExp += exp[i];
            }

            for (int i = 0; i < logits.Length; i++)
            {
                exp[i] /= sumExp;
            }
            return exp;
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
        }
    }
}
