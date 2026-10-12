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

    public enum SemanticFamily
    {
        Weapons = 0,
        Toys = 1,
        Electronics = 2,
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

        // Hybrid / Majority Ambiguity Fields
        public ReconstructedCategory secondaryCategory;
        public float secondaryConfidence;
        public bool isHybridSketch;
        public float hybridBlendAlpha;
        public SemanticFamily family;
        public float cumulativeTop3Confidence;
    }

    /// <summary>
    /// Executes the multi-task AntiGravity Draw2Voxel ONNX model in Unity Sentis / Inference Engine.
    /// Disambiguates perspective (Front/Side/Top), classifies object category with OOD rejection,
    /// performs majority canonical labeling and soft latent blending for hybrid sketches,
    /// feeds a 2-channel (Stroke + Distance Field) input tensor, and renders full volumetric 3D assets.
    /// </summary>
    public class SentisMultiTaskEngine : MonoBehaviour
    {
        [Header("Model Configuration")]
        [SerializeField] private ModelAsset multiTaskModelAsset;
        [SerializeField] private ModelAsset decoderModelAsset;
        [SerializeField] private TextAsset centroidsJsonAsset;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;

        [Header("Majority Decision & Hybrid Sketch Blending")]
        [Tooltip("If top class confidence is above this threshold, treat as a clear majority decision.")]
        [Range(0.40f, 0.90f)]
        [SerializeField] private float clearMajorityThreshold = 0.55f;

        [Tooltip("Maximum difference between top-1 and top-2 class probabilities to consider them ambiguous/hybrid.")]
        [Range(0.05f, 0.40f)]
        [SerializeField] private float hybridAmbiguityMargin = 0.20f;

        [Tooltip("Cumulative top-3 probability required to accept creative ambiguous drawings (overrides OOD rejection).")]
        [Range(0.50f, 0.95f)]
        [SerializeField] private float cumulativeTop3AcceptThreshold = 0.70f;

        [Tooltip("Maximum probability below which the distribution is considered completely flat / pure noise.")]
        [Range(0.10f, 0.40f)]
        [SerializeField] private float flatDistributionEntropyThreshold = 0.25f;

        [Tooltip("Enable latent blending between top-2 class centroids when hybrid/ambiguous drawing is detected within the same family.")]
        [SerializeField] private bool enableLatentBlending = true;

        [Tooltip("Decode canonical centroid embedding for the majority class to guarantee smooth, volumetric geometry.")]
        [SerializeField] private bool useCanonicalEmbeddingDecoding = true;

        [Header("OOD & Perspective Rejection")]
        [Tooltip("Minimum softmax probability required to accept drawing as valid shape.")]
        [Range(0.3f, 0.95f)]
        [SerializeField] private float oodConfidenceThreshold = 0.60f;

        [Header("Visual-Hull Prior (GATOR)")]
        [Tooltip("Constrains decoded voxels to the extruded 2D sketch bounding envelope. Kept false by default to preserve full volumetric rounded 3D contours.")]
        [SerializeField] private bool applyVisualHullPrior = false;

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
        private Model _runtimeDecoderModel;
#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
        private Worker _worker;
        private Worker _decoderWorker;
#else
        private IWorker _worker;
        private IWorker _decoderWorker;
#endif

        private Dictionary<int, float[]> _classCentroids;

        public const int InputRes = 64;
        public const int VoxelDim = 64;
        public const int TotalVoxels = VoxelDim * VoxelDim * VoxelDim; // 262,144 voxels

        public event Action<MultiTaskInferenceResult> OnInferenceComplete;

        public static SemanticFamily GetSemanticFamily(ReconstructedCategory cat)
        {
            switch (cat)
            {
                case ReconstructedCategory.Sword:
                case ReconstructedCategory.Shield:
                case ReconstructedCategory.Blaster:
                case ReconstructedCategory.Axe:
                    return SemanticFamily.Weapons;

                case ReconstructedCategory.TeddyBear:
                case ReconstructedCategory.RubberDuck:
                case ReconstructedCategory.ToyRobot:
                case ReconstructedCategory.ToyBunny:
                    return SemanticFamily.Toys;

                case ReconstructedCategory.Gamepad:
                case ReconstructedCategory.RetroTV:
                case ReconstructedCategory.GameBoy:
                    return SemanticFamily.Electronics;

                default:
                    return SemanticFamily.Unknown;
            }
        }

        private void Awake()
        {
            InitializeEngine();
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
            _decoderWorker?.Dispose();
        }

        public void InitializeEngine()
        {
#if UNITY_EDITOR
            if (multiTaskModelAsset == null)
            {
                multiTaskModelAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/ML/Models/draw2voxel_multitask.onnx");
            }
            if (decoderModelAsset == null)
            {
                decoderModelAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/ML/Models/draw2voxel_decoder.onnx");
            }
            if (centroidsJsonAsset == null)
            {
                centroidsJsonAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/ML/Models/draw2voxel_multitask_centroids.json");
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

            if (decoderModelAsset != null)
            {
                try
                {
                    _runtimeDecoderModel = ModelLoader.Load(decoderModelAsset);
                    _decoderWorker = new Worker(_runtimeDecoderModel, backend);
                    Debug.Log($"[SentisMultiTaskEngine] Initialized standalone decoder worker on backend: {backend}");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[SentisMultiTaskEngine] Decoder backend {backend} failed ({ex.Message}), falling back to CPU.");
                    _decoderWorker = new Worker(_runtimeDecoderModel, BackendType.CPU);
                }
            }

            LoadCentroids();
        }

        private void LoadCentroids()
        {
            _classCentroids = new Dictionary<int, float[]>();
            if (centroidsJsonAsset == null) return;

            string json = centroidsJsonAsset.text;
            if (string.IsNullOrEmpty(json)) return;

            try
            {
                for (int c = 0; c < 12; c++)
                {
                    string marker = $"\"{c}\":";
                    int idx = json.IndexOf(marker, StringComparison.Ordinal);
                    if (idx < 0) continue;

                    int cMarker = json.IndexOf("\"centroid\":", idx, StringComparison.Ordinal);
                    if (cMarker < 0) continue;

                    int startBracket = json.IndexOf('[', cMarker);
                    int endBracket = json.IndexOf(']', startBracket);
                    if (startBracket < 0 || endBracket < 0) continue;

                    string arrayStr = json.Substring(startBracket + 1, endBracket - startBracket - 1);
                    string[] tokens = arrayStr.Split(new char[] { ',', ' ', '\r', '\n', '\t' }, StringSplitOptions.RemoveEmptyEntries);

                    float[] vec = new float[tokens.Length];
                    for (int k = 0; k < tokens.Length; k++)
                    {
                        if (float.TryParse(tokens[k], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float val))
                        {
                            vec[k] = val;
                        }
                    }

                    if (vec.Length == 512)
                    {
                        _classCentroids[c] = vec;
                    }
                }
                Debug.Log($"[SentisMultiTaskEngine] Successfully parsed {_classCentroids.Count} class centroids in R^512.");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[SentisMultiTaskEngine] Error parsing centroids JSON: {ex.Message}");
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

            // Retrieve multi-output tensors safely: separate voxel_occupancy & voxel_rgb, or legacy output_voxels_4ch
            Tensor<float> occTensor = HasOutput(_runtimeModel, "voxel_occupancy") ? _worker.PeekOutput("voxel_occupancy") as Tensor<float> : null;
            Tensor<float> rgbTensor = HasOutput(_runtimeModel, "voxel_rgb") ? _worker.PeekOutput("voxel_rgb") as Tensor<float> : null;
            Tensor<float> legacyVoxTensor = HasOutput(_runtimeModel, "output_voxels_4ch") ? _worker.PeekOutput("output_voxels_4ch") as Tensor<float> : null;
            Tensor<float> catLogitsTensor = HasOutput(_runtimeModel, "category_logits") ? _worker.PeekOutput("category_logits") as Tensor<float> : null;
            Tensor<float> viewLogitsTensor = HasOutput(_runtimeModel, "view_logits") ? _worker.PeekOutput("view_logits") as Tensor<float> : null;
            Tensor<float> latentTensor = HasOutput(_runtimeModel, "latent_z") ? _worker.PeekOutput("latent_z") as Tensor<float> : null;

            if ((occTensor == null && legacyVoxTensor == null) || catLogitsTensor == null || viewLogitsTensor == null)
            {
                Debug.LogError("[SentisMultiTaskEngine] Failed to retrieve one or more output tensors.");
                yield break;
            }

            using Tensor<float> catCpu = catLogitsTensor.ReadbackAndClone();
            using Tensor<float> viewCpu = viewLogitsTensor.ReadbackAndClone();
            float[] catLogits = catCpu.DownloadToArray();
            float[] viewLogits = viewCpu.DownloadToArray();

            float[] latentZ = null;
            if (latentTensor != null)
            {
                using Tensor<float> latentCpu = latentTensor.ReadbackAndClone();
                latentZ = latentCpu.DownloadToArray();
            }

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
            else if (legacyVoxTensor != null)
            {
                using Tensor<float> voxCpu = legacyVoxTensor.ReadbackAndClone();
                rawLegacyVox4Ch = voxCpu.DownloadToArray();
            }
#else
            using TensorFloat inputTensor = new TensorFloat(inputShape, buffer2Ch);
            _worker.Schedule(inputTensor);
            yield return null;

            TensorFloat occTensor = HasOutput(_runtimeModel, "voxel_occupancy") ? _worker.PeekOutput("voxel_occupancy") as TensorFloat : null;
            TensorFloat rgbTensor = HasOutput(_runtimeModel, "voxel_rgb") ? _worker.PeekOutput("voxel_rgb") as TensorFloat : null;
            TensorFloat legacyVoxTensor = HasOutput(_runtimeModel, "output_voxels_4ch") ? _worker.PeekOutput("output_voxels_4ch") as TensorFloat : null;
            TensorFloat catLogitsTensor = HasOutput(_runtimeModel, "category_logits") ? _worker.PeekOutput("category_logits") as TensorFloat : null;
            TensorFloat viewLogitsTensor = HasOutput(_runtimeModel, "view_logits") ? _worker.PeekOutput("view_logits") as TensorFloat : null;
            TensorFloat latentTensor = HasOutput(_runtimeModel, "latent_z") ? _worker.PeekOutput("latent_z") as TensorFloat : null;

            float[] catLogits = catLogitsTensor != null ? catLogitsTensor.ToReadOnlyArray() : new float[0];
            float[] viewLogits = viewLogitsTensor != null ? viewLogitsTensor.ToReadOnlyArray() : new float[0];
            float[] latentZ = latentTensor != null ? latentTensor.ToReadOnlyArray() : null;
            float[] rawOcc = occTensor != null ? occTensor.ToReadOnlyArray() : null;
            float[] rawRgb = rgbTensor != null ? rgbTensor.ToReadOnlyArray() : null;
            float[] rawLegacyVox4Ch = legacyVoxTensor != null ? legacyVoxTensor.ToReadOnlyArray() : null;
#endif

            // 1. Evaluate Softmax Probabilities & Sort Class Predictions
            float[] catProbs = ComputeSoftmax(catLogits);
            int[] sortedIndices = new int[catProbs.Length];
            for (int i = 0; i < sortedIndices.Length; i++) sortedIndices[i] = i;
            Array.Sort(sortedIndices, (a, b) => catProbs[b].CompareTo(catProbs[a]));

            int top1Idx = sortedIndices[0];
            float top1Prob = catProbs[top1Idx];

            int top2Idx = sortedIndices.Length > 1 ? sortedIndices[1] : top1Idx;
            float top2Prob = sortedIndices.Length > 1 ? catProbs[top2Idx] : 0f;

            int top3Idx = sortedIndices.Length > 2 ? sortedIndices[2] : top2Idx;
            float top3Prob = sortedIndices.Length > 2 ? catProbs[top3Idx] : 0f;

            float cumulativeTop3 = top1Prob + top2Prob + top3Prob;

            // 2. Differentiate Creative Ambiguous Shapes vs. Pure Noise / Flat Entropy
            bool isPureNoise = (top1Prob < flatDistributionEntropyThreshold) ||
                               (top1Idx == (int)ReconstructedCategory.OOD_Reject && top1Prob > 0.50f);
            bool isCreativeAmbiguous = !isPureNoise && (cumulativeTop3 >= cumulativeTop3AcceptThreshold);

            bool isOOD = false;
            if (isPureNoise)
            {
                isOOD = true;
            }
            else if (!isCreativeAmbiguous && top1Prob < oodConfidenceThreshold)
            {
                isOOD = true;
            }

            ReconstructedCategory majorityCat = isOOD ? ReconstructedCategory.OOD_Reject : (ReconstructedCategory)top1Idx;

            // 3. Majority Decision Rule & Latent Blending
            bool isHybrid = false;
            float blendAlpha = 1.0f;
            ReconstructedCategory secondaryCat = (ReconstructedCategory)top2Idx;
            SemanticFamily family = GetSemanticFamily(majorityCat);

            float[] finalZ = latentZ;

            if (!isOOD && enableLatentBlending && _classCentroids != null)
            {
                SemanticFamily fam1 = GetSemanticFamily((ReconstructedCategory)top1Idx);
                SemanticFamily fam2 = GetSemanticFamily((ReconstructedCategory)top2Idx);

                bool top2Close = Mathf.Abs(top1Prob - top2Prob) <= hybridAmbiguityMargin;
                bool sameFamily = (fam1 == fam2) && (fam1 != SemanticFamily.Unknown);

                if (top2Close && sameFamily && _classCentroids.ContainsKey(top1Idx) && _classCentroids.ContainsKey(top2Idx))
                {
                    // Compute blended latent code:
                    // z_blend = alpha * z_C1 + (1 - alpha) * z_C2 where alpha = P1 / (P1 + P2)
                    blendAlpha = top1Prob / (top1Prob + top2Prob + 1e-6f);
                    float[] z1 = _classCentroids[top1Idx];
                    float[] z2 = _classCentroids[top2Idx];

                    finalZ = new float[512];
                    for (int d = 0; d < 512; d++)
                    {
                        finalZ[d] = blendAlpha * z1[d] + (1.0f - blendAlpha) * z2[d];
                    }

                    isHybrid = true;
                    Debug.Log($"[SentisMultiTaskEngine] Hybrid sketch detected in family {fam1}! Blending {((ReconstructedCategory)top1Idx)} ({blendAlpha*100:F0}%) and {((ReconstructedCategory)top2Idx)} ({(1-blendAlpha)*100:F0}%).");
                }
                else if (useCanonicalEmbeddingDecoding && _classCentroids.ContainsKey(top1Idx))
                {
                    // Snap to canonical majority centroid embedding
                    finalZ = _classCentroids[top1Idx];
                }
            }

            // 4. Standalone 3D Decoder Execution on Blended or Canonical Latent
            if (!isOOD && _decoderWorker != null && finalZ != null && (isHybrid || useCanonicalEmbeddingDecoding))
            {
                TensorShape zShape = new TensorShape(1, 512);
#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
                using Tensor<float> zTensor = new Tensor<float>(zShape, finalZ);
                _decoderWorker.Schedule(zTensor);
                yield return null;

                Tensor<float> decOccTensor = HasOutput(_runtimeDecoderModel, "voxel_occupancy") ? _decoderWorker.PeekOutput("voxel_occupancy") as Tensor<float> : null;
                Tensor<float> decRgbTensor = HasOutput(_runtimeDecoderModel, "voxel_rgb") ? _decoderWorker.PeekOutput("voxel_rgb") as Tensor<float> : null;

                if (decOccTensor != null && decRgbTensor != null)
                {
                    using Tensor<float> dOccCpu = decOccTensor.ReadbackAndClone();
                    using Tensor<float> dRgbCpu = decRgbTensor.ReadbackAndClone();
                    rawOcc = dOccCpu.DownloadToArray();
                    rawRgb = dRgbCpu.DownloadToArray();
                }
#else
                using TensorFloat zTensor = new TensorFloat(zShape, finalZ);
                _decoderWorker.Schedule(zTensor);
                yield return null;

                TensorFloat decOccTensor = HasOutput(_runtimeDecoderModel, "voxel_occupancy") ? _decoderWorker.PeekOutput("voxel_occupancy") as TensorFloat : null;
                TensorFloat decRgbTensor = HasOutput(_runtimeDecoderModel, "voxel_rgb") ? _decoderWorker.PeekOutput("voxel_rgb") as TensorFloat : null;

                if (decOccTensor != null && decRgbTensor != null)
                {
                    rawOcc = decOccTensor.ToReadOnlyArray();
                    rawRgb = decRgbTensor.ToReadOnlyArray();
                }
#endif
            }

            // 5. Evaluate View Disambiguation Softmax
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
                category = majorityCat,
                categoryConfidence = top1Prob,
                detectedView = detectedView,
                viewConfidence = bestViewProb,
                isOutOfDistribution = isOOD,
                occupancyGrid = occupancyGrid,
                voxelColors = voxelColors,
                latencyMs = latency,
                secondaryCategory = secondaryCat,
                secondaryConfidence = top2Prob,
                isHybridSketch = isHybrid,
                hybridBlendAlpha = blendAlpha,
                family = family,
                cumulativeTop3Confidence = cumulativeTop3
            };

            Debug.Log($"[SentisMultiTaskEngine] Done: Cat={majorityCat} ({top1Prob*100:F1}%), Hybrid={isHybrid} (Alpha={blendAlpha*100:F0}%), View={detectedView} ({bestViewProb*100:F1}%), OOD={isOOD}, Latency={latency:F1}ms");
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

        private bool HasOutput(Model model, string outputName)
        {
            if (model == null || model.outputs == null) return false;
            for (int i = 0; i < model.outputs.Count; i++)
            {
                if (model.outputs[i].name == outputName) return true;
            }
            return false;
        }
    }
}
