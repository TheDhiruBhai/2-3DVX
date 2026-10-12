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
        OOD_Reject = 11,
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
    /// feeds a 2-channel (Stroke + Distance Field) input tensor, applies GATOR Visual-Hull prior masking,
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

        [Header("Visual-Hull Prior (GATOR)")]
        [Tooltip("Constrains decoded voxels to the extruded 2D sketch bounding envelope to eliminate empty space hallucinations.")]
        [SerializeField] private bool applyVisualHullPrior = true;

        [Tooltip("Margin dilation applied to the 2D bounding envelope before extrusion.")]
        [Range(0, 5)]
        [SerializeField] private int envelopeDilation = 2;

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
        /// Dispatches a 2-channel [1, 2, 64, 64] float input sketch to the multi-task network.
        /// Accepts either 8192 floats (Channel 0 Stroke + Channel 1 Distance Field) or 4096 floats (auto-computes Ch1).
        /// Asynchronously yields across frames to maintain VR frame rates.
        /// </summary>
        public void PredictMultiTaskAsync(float[] rasterBuffer, Action<MultiTaskInferenceResult> callback = null)
        {
            if (_worker == null)
            {
                Debug.LogError("[SentisMultiTaskEngine] Inference worker is not initialized.");
                return;
            }

            if (rasterBuffer == null || (rasterBuffer.Length != InputRes * InputRes && rasterBuffer.Length != 2 * InputRes * InputRes))
            {
                Debug.LogError($"[SentisMultiTaskEngine] Invalid raster buffer length: {rasterBuffer?.Length}. Expected {InputRes * InputRes} or {2 * InputRes * InputRes}.");
                return;
            }

            float[] buffer2Ch;
            if (rasterBuffer.Length == 2 * InputRes * InputRes)
            {
                buffer2Ch = rasterBuffer;
            }
            else
            {
                buffer2Ch = BuildTwoChannelBuffer(rasterBuffer);
            }

            StartCoroutine(ExecuteMultiTaskCoroutine(buffer2Ch, callback));
        }

        private IEnumerator ExecuteMultiTaskCoroutine(float[] buffer2Ch, Action<MultiTaskInferenceResult> callback)
        {
            float startTime = Time.realtimeSinceStartup;
            TensorShape inputShape = new TensorShape(1, 2, InputRes, InputRes);

#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
            using Tensor<float> inputTensor = new Tensor<float>(inputShape, buffer2Ch);
            _worker.Schedule(inputTensor);

            // Yield frame so GPU compute executes without blocking VR render thread
            yield return null;

            // Retrieve multi-output tensors: separate voxel_occupancy & voxel_rgb, or legacy output_voxels_4ch
            Tensor<float> occTensor = _worker.PeekOutput("voxel_occupancy") as Tensor<float>;
            Tensor<float> rgbTensor = _worker.PeekOutput("voxel_rgb") as Tensor<float>;
            Tensor<float> legacyVoxTensor = _worker.PeekOutput("output_voxels_4ch") as Tensor<float>;
            Tensor<float> catLogitsTensor = _worker.PeekOutput("category_logits") as Tensor<float>;
            Tensor<float> viewLogitsTensor = _worker.PeekOutput("view_logits") as Tensor<float>;

            if ((occTensor == null && legacyVoxTensor == null) || catLogitsTensor == null || viewLogitsTensor == null)
            {
                Debug.LogError("[SentisMultiTaskEngine] Failed to retrieve one or more output tensors.");
                yield break;
            }

            using Tensor<float> catCpu = catLogitsTensor.ReadbackAndClone();
            using Tensor<float> viewCpu = viewLogitsTensor.ReadbackAndClone();
            float[] catLogits = catCpu.DownloadToArray();
            float[] viewLogits = viewCpu.DownloadToArray();

            float[] rawOcc = null;
            float[] rawRgb = null;
            float[] rawLegacyVox4Ch = null;

            if (occTensor != null && rgbTensor != null)
            {
                using Tensor<float> occCpu = occTensor.ReadbackAndClone();
                using Tensor<float> rgbCpu = rgbTensor.ReadbackAndClone();
                rawOcc = occCpu.DownloadToArray();
                rawRgb = rgbCpu.DownloadToArray();
            }
            else
            {
                using Tensor<float> voxCpu = legacyVoxTensor.ReadbackAndClone();
                rawLegacyVox4Ch = voxCpu.DownloadToArray();
            }
#else
            using TensorFloat inputTensor = new TensorFloat(inputShape, buffer2Ch);
            _worker.Schedule(inputTensor);
            yield return null;

            TensorFloat occTensor = _worker.PeekOutput("voxel_occupancy") as TensorFloat;
            TensorFloat rgbTensor = _worker.PeekOutput("voxel_rgb") as TensorFloat;
            TensorFloat legacyVoxTensor = _worker.PeekOutput("output_voxels_4ch") as TensorFloat;
            TensorFloat catLogitsTensor = _worker.PeekOutput("category_logits") as TensorFloat;
            TensorFloat viewLogitsTensor = _worker.PeekOutput("view_logits") as TensorFloat;

            float[] catLogits = catLogitsTensor.ToReadOnlyArray();
            float[] viewLogits = viewLogitsTensor.ToReadOnlyArray();
            float[] rawOcc = occTensor != null ? occTensor.ToReadOnlyArray() : null;
            float[] rawRgb = rgbTensor != null ? rgbTensor.ToReadOnlyArray() : null;
            float[] rawLegacyVox4Ch = legacyVoxTensor != null ? legacyVoxTensor.ToReadOnlyArray() : null;
#endif

            // 1. Evaluate Category Softmax & OOD Rejection (12 classes: 0..10 canonical, 11 OOD_Reject)
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

            bool isOOD = (bestCatIdx == (int)ReconstructedCategory.OOD_Reject) || (bestCatProb < oodConfidenceThreshold);
            ReconstructedCategory cat = isOOD ? ReconstructedCategory.OOD_Reject : (ReconstructedCategory)bestCatIdx;

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
            byte[] occupancyGrid = new byte[TotalVoxels];
            Color32[] voxelColors = new Color32[TotalVoxels];

            if (rawOcc != null && rawRgb != null)
            {
                int rOffset = 0;
                int gOffset = TotalVoxels;
                int bOffset = TotalVoxels * 2;

                for (int i = 0; i < TotalVoxels; i++)
                {
                    float occ = rawOcc[i];
                    if (occ >= occupancyThreshold)
                    {
                        occupancyGrid[i] = 1;
                        byte r = (byte)Mathf.Clamp((int)(rawRgb[rOffset + i] * 255.0f), 0, 255);
                        byte g = (byte)Mathf.Clamp((int)(rawRgb[gOffset + i] * 255.0f), 0, 255);
                        byte b = (byte)Mathf.Clamp((int)(rawRgb[bOffset + i] * 255.0f), 0, 255);
                        voxelColors[i] = new Color32(r, g, b, 255);
                    }
                    else
                    {
                        occupancyGrid[i] = 0;
                    }
                }
            }
            else if (rawLegacyVox4Ch != null)
            {
                int chStride = TotalVoxels;
                int rOffset = chStride;
                int gOffset = chStride * 2;
                int bOffset = chStride * 3;

                for (int i = 0; i < TotalVoxels; i++)
                {
                    float occ = rawLegacyVox4Ch[i];
                    if (occ >= occupancyThreshold)
                    {
                        occupancyGrid[i] = 1;
                        byte r = (byte)Mathf.Clamp((int)(rawLegacyVox4Ch[rOffset + i] * 255.0f), 0, 255);
                        byte g = (byte)Mathf.Clamp((int)(rawLegacyVox4Ch[gOffset + i] * 255.0f), 0, 255);
                        byte b = (byte)Mathf.Clamp((int)(rawLegacyVox4Ch[bOffset + i] * 255.0f), 0, 255);
                        voxelColors[i] = new Color32(r, g, b, 255);
                    }
                    else
                    {
                        occupancyGrid[i] = 0;
                    }
                }
            }

            // 3.5. GATOR Visual-Hull Intersection Masking
            // Multiplies decoded 3D occupancy by the extruded 2D envelope to eliminate empty space hallucinations
            if (applyVisualHullPrior && !isOOD)
            {
                float[] ch0Stroke = new float[InputRes * InputRes];
                Array.Copy(buffer2Ch, 0, ch0Stroke, 0, InputRes * InputRes);
                byte[] envelope2D = ComputeVisualHullEnvelope2D(ch0Stroke, envelopeDilation);
                int culledByHull = ApplyVisualHullMasking(occupancyGrid, envelope2D, detectedView);
                if (culledByHull > 0)
                {
                    Debug.Log($"[SentisMultiTaskEngine] GATOR Visual-Hull prior masked {culledByHull} voxels outside 2D contour envelope.");
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

        // ==============================================================================
        // GATOR VISUAL-HULL PRIOR & 2-CHANNEL PIPELINE HELPERS
        // ==============================================================================

        /// <summary>
        /// Computes the 2D bounding envelope (filled silhouette) from a stroke mask using flood-fill and morphological dilation.
        /// </summary>
        public static byte[] ComputeVisualHullEnvelope2D(float[] strokeMask, int marginDilation = 2)
        {
            byte[] env = new byte[InputRes * InputRes];
            bool[] isStroke = new bool[InputRes * InputRes];
            for (int i = 0; i < isStroke.Length; i++)
            {
                if (strokeMask[i] > 0.2f) isStroke[i] = true;
            }

            // Dilate stroke by 2 pixels to bridge small unclosed stroke gaps
            bool[] closedStroke = DilateBinary2D(isStroke, 2);

            // Flood-fill exterior from all 4 borders
            bool[] exterior = new bool[InputRes * InputRes];
            Queue<int> q = new Queue<int>(256);

            for (int x = 0; x < InputRes; x++)
            {
                int top = x * InputRes + 0;
                int btm = x * InputRes + (InputRes - 1);
                if (!closedStroke[top] && !exterior[top]) { exterior[top] = true; q.Enqueue(top); }
                if (!closedStroke[btm] && !exterior[btm]) { exterior[btm] = true; q.Enqueue(btm); }
            }
            for (int y = 0; y < InputRes; y++)
            {
                int left = 0 * InputRes + y;
                int right = (InputRes - 1) * InputRes + y;
                if (!closedStroke[left] && !exterior[left]) { exterior[left] = true; q.Enqueue(left); }
                if (!closedStroke[right] && !exterior[right]) { exterior[right] = true; q.Enqueue(right); }
            }

            int[] dx = { 1, -1, 0, 0 };
            int[] dy = { 0, 0, 1, -1 };

            while (q.Count > 0)
            {
                int cur = q.Dequeue();
                int cx = cur / InputRes;
                int cy = cur % InputRes;

                for (int d = 0; d < 4; d++)
                {
                    int nx = cx + dx[d];
                    int ny = cy + dy[d];
                    if (nx >= 0 && nx < InputRes && ny >= 0 && ny < InputRes)
                    {
                        int nIdx = nx * InputRes + ny;
                        if (!closedStroke[nIdx] && !exterior[nIdx])
                        {
                            exterior[nIdx] = true;
                            q.Enqueue(nIdx);
                        }
                    }
                }
            }

            // Interior envelope is everything not reachable from border
            bool[] interior = new bool[InputRes * InputRes];
            for (int i = 0; i < interior.Length; i++)
            {
                interior[i] = !exterior[i];
            }

            // Dilate envelope by margin
            bool[] dilatedEnvelope = marginDilation > 0 ? DilateBinary2D(interior, marginDilation) : interior;
            for (int i = 0; i < env.Length; i++)
            {
                env[i] = dilatedEnvelope[i] ? (byte)1 : (byte)0;
            }
            return env;
        }

        private static bool[] DilateBinary2D(bool[] src, int iterations)
        {
            bool[] cur = (bool[])src.Clone();
            bool[] next = new bool[src.Length];

            for (int it = 0; it < iterations; it++)
            {
                Array.Copy(cur, next, src.Length);
                for (int x = 0; x < InputRes; x++)
                {
                    for (int y = 0; y < InputRes; y++)
                    {
                        if (cur[x * InputRes + y])
                        {
                            if (x > 0) next[(x - 1) * InputRes + y] = true;
                            if (x < InputRes - 1) next[(x + 1) * InputRes + y] = true;
                            if (y > 0) next[x * InputRes + (y - 1)] = true;
                            if (y < InputRes - 1) next[x * InputRes + (y + 1)] = true;
                        }
                    }
                }
                bool[] tmp = cur;
                cur = next;
                next = tmp;
            }
            return cur;
        }

        private int ApplyVisualHullMasking(byte[] occupancy, byte[] envelope2D, PerspectiveView view)
        {
            int culledCount = 0;
            for (int x = 0; x < VoxelDim; x++)
            {
                for (int y = 0; y < VoxelDim; y++)
                {
                    for (int z = 0; z < VoxelDim; z++)
                    {
                        int flatIdx = x * (VoxelDim * VoxelDim) + y * VoxelDim + z;
                        if (occupancy[flatIdx] == 0) continue;

                        int u = 0, v = 0;
                        switch (view)
                        {
                            case PerspectiveView.Front_XY:
                                u = x;
                                v = y;
                                break;
                            case PerspectiveView.Side_ZY:
                                u = z;
                                v = y;
                                break;
                            case PerspectiveView.Top_XZ:
                                u = x;
                                v = z;
                                break;
                        }

                        if (u < 0 || u >= InputRes || v < 0 || v >= InputRes || envelope2D[u * InputRes + v] == 0)
                        {
                            occupancy[flatIdx] = 0;
                            culledCount++;
                        }
                    }
                }
            }
            return culledCount;
        }

        /// <summary>
        /// Builds a 2-channel normalized buffer [Channel 0: Stroke, Channel 1: SDF] from a single 64x64 stroke buffer.
        /// </summary>
        public static float[] BuildTwoChannelBuffer(float[] singleChannelStroke)
        {
            float[] buffer2Ch = new float[2 * InputRes * InputRes];
            Array.Copy(singleChannelStroke, 0, buffer2Ch, 0, InputRes * InputRes);

            // Compute 2-pass Chamfer distance transform for Channel 1
            float[] dist = new float[InputRes * InputRes];
            float maxVal = 1e6f;
            for (int i = 0; i < dist.Length; i++)
            {
                dist[i] = singleChannelStroke[i] > 0.2f ? 0.0f : maxVal;
            }

            float sqrt2 = 1.41421356f;
            // Forward pass
            for (int x = 0; x < InputRes; x++)
            {
                for (int y = 0; y < InputRes; y++)
                {
                    int idx = x * InputRes + y;
                    if (dist[idx] > 0f)
                    {
                        float best = dist[idx];
                        if (x > 0) best = Mathf.Min(best, dist[(x - 1) * InputRes + y] + 1.0f);
                        if (y > 0) best = Mathf.Min(best, dist[x * InputRes + (y - 1)] + 1.0f);
                        if (x > 0 && y > 0) best = Mathf.Min(best, dist[(x - 1) * InputRes + (y - 1)] + sqrt2);
                        if (x < InputRes - 1 && y > 0) best = Mathf.Min(best, dist[(x + 1) * InputRes + (y - 1)] + sqrt2);
                        dist[idx] = best;
                    }
                }
            }

            // Backward pass
            for (int x = InputRes - 1; x >= 0; x--)
            {
                for (int y = InputRes - 1; y >= 0; y--)
                {
                    int idx = x * InputRes + y;
                    float best = dist[idx];
                    if (x < InputRes - 1) best = Mathf.Min(best, dist[(x + 1) * InputRes + y] + 1.0f);
                    if (y < InputRes - 1) best = Mathf.Min(best, dist[x * InputRes + (y + 1)] + 1.0f);
                    if (x < InputRes - 1 && y < InputRes - 1) best = Mathf.Min(best, dist[(x + 1) * InputRes + (y + 1)] + sqrt2);
                    if (x > 0 && y < InputRes - 1) best = Mathf.Min(best, dist[(x - 1) * InputRes + (y + 1)] + sqrt2);
                    dist[idx] = best;
                }
            }

            // Normalize truncated distance field to [0.0, 1.0] (d_max = 16.0f)
            float dMax = 16.0f;
            int ch1Offset = InputRes * InputRes;
            for (int i = 0; i < dist.Length; i++)
            {
                buffer2Ch[ch1Offset + i] = Mathf.Clamp01(dist[i] / dMax);
            }

            return buffer2Ch;
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
