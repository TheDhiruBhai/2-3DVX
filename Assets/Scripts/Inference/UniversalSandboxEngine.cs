using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Unity.Sentis;
using VRDraw2Voxel.VoxelMesh;

namespace VRDraw2Voxel.Inference
{
    [Serializable]
    public class SandboxCategoryInfo
    {
        public string name;
        public string label;
        public string family;
        public int[] color;   // RGB [0..255]
        public int[] accent;  // RGB [0..255]
    }

    [Serializable]
    public class SandboxPresetItem
    {
        public string name;
        public string label;
        public string family;
        public float confidence;
        public string data;   // Base64 encoded 64x64 uint8 stroke mask
    }

    public struct SandboxInferenceResult
    {
        public int categoryId;
        public string categoryName;
        public string label;
        public string family;
        public float confidence;
        public List<KeyValuePair<string, float>> top5;
        public byte[] occupancyGrid;
        public Color32[] voxelColors;
        public float latencyMs;
        public bool isNoise;
    }

    /// <summary>
    /// Universal 50-Class Sandbox Engine for AntiGravity 2-3DVX:
    /// Powered by the Quick, Draw! universal dataset. Recognizes 50 creative sandbox classes
    /// (Weapons, Tools, Vehicles, Creatures, Food, Nature, Tech, Armor, Treasures)
    /// and inflates 2D player sketches into interactive 3D voxel props with real physics.
    /// </summary>
    public class UniversalSandboxEngine : MonoBehaviour
    {
        [Header("Model & Catalog Assets")]
        [SerializeField] private ModelAsset sandboxModelAsset;
        [SerializeField] private TextAsset categoriesJsonAsset;
        [SerializeField] private TextAsset presetsJsonAsset;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;

        [Header("Volumetric Infill Settings")]
        [Tooltip("Maximum half-thickness in voxels for inflated 3D volume (e.g. 10 voxels = 20 voxel thickness).")]
        [Range(4, 20)]
        [SerializeField] private int maxHalfThickness = 12;

        [Tooltip("Minimum thickness to guarantee solid structural integrity in physics simulation.")]
        [Range(1, 6)]
        [SerializeField] private int minHalfThickness = 2;

        [Header("Physics & Prop Spawning")]
        [Tooltip("Scale of individual voxels in meters (0.015 = 1.5cm).")]
        [SerializeField] private float voxelScale = 0.015f;
        [SerializeField] private float propMass = 1.0f;
        [SerializeField] private Material vertexColorMaterial;

        private Model _runtimeModel;
#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
        private Worker _worker;
#else
        private IWorker _worker;
#endif
        private List<SandboxCategoryInfo> _catalog;
        private List<SandboxPresetItem> _presets;
        private readonly List<GameObject> _spawnedProps = new List<GameObject>();
        private GameObject _sandboxFloor;

        public const int InputRes = 64;
        public const int VoxelDim = 64;
        public const int TotalVoxels = VoxelDim * VoxelDim * VoxelDim;

        public event Action<SandboxInferenceResult> OnSandboxInferenceComplete;

        public List<SandboxPresetItem> Presets => _presets;
        public int SpawnedPropCount => _spawnedProps.Count;

        private void Awake()
        {
            InitializeEngine();
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
        }

        public void InitializeEngine()
        {
#if UNITY_EDITOR
            if (sandboxModelAsset == null)
            {
                sandboxModelAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/ML/Models/draw2voxel_sandbox.onnx");
            }
            if (categoriesJsonAsset == null)
            {
                categoriesJsonAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/ML/Models/sandbox_categories.json");
            }
            if (presetsJsonAsset == null)
            {
                presetsJsonAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/ML/Models/sandbox_presets.json");
            }
            if (vertexColorMaterial == null)
            {
                vertexColorMaterial = UnityEditor.AssetDatabase.LoadAssetAtPath<Material>("Assets/Materials/VoxelVertexColorMat.mat");
            }
#endif

            LoadCatalog();
            LoadPresets();

            if (sandboxModelAsset == null)
            {
                Debug.LogWarning("[UniversalSandboxEngine] Sandbox ModelAsset not assigned.");
                return;
            }

            try
            {
                _runtimeModel = ModelLoader.Load(sandboxModelAsset);
                _worker = new Worker(_runtimeModel, backend);
                Debug.Log($"[UniversalSandboxEngine] Initialized 50-Class Sandbox Worker on backend: {backend}");
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[UniversalSandboxEngine] GPU Compute failed ({ex.Message}), falling back to CPU.");
                backend = BackendType.CPU;
                _worker = new Worker(_runtimeModel, backend);
            }
        }

        private void LoadCatalog()
        {
            if (categoriesJsonAsset == null) return;
            try
            {
                string jsonWrapped = "{\"items\":" + categoriesJsonAsset.text + "}";
                Wrapper wrapper = JsonUtility.FromJson<Wrapper>(jsonWrapped);
                _catalog = wrapper.items;
                Debug.Log($"[UniversalSandboxEngine] Loaded {_catalog.Count} creative sandbox categories from catalog.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[UniversalSandboxEngine] Failed to parse sandbox_categories.json: {ex.Message}");
            }
        }

        private void LoadPresets()
        {
            if (presetsJsonAsset == null) return;
            try
            {
                string jsonWrapped = "{\"items\":" + presetsJsonAsset.text + "}";
                PresetWrapper wrapper = JsonUtility.FromJson<PresetWrapper>(jsonWrapped);
                _presets = wrapper.items;
                Debug.Log($"[UniversalSandboxEngine] Loaded {_presets.Count} authentic Quick, Draw! presets.");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[UniversalSandboxEngine] Failed to parse sandbox_presets.json: {ex.Message}");
            }
        }

        [Serializable]
        private class Wrapper
        {
            public List<SandboxCategoryInfo> items;
        }

        [Serializable]
        private class PresetWrapper
        {
            public List<SandboxPresetItem> items;
        }

        public byte[] GetPresetMask(string presetName)
        {
            if (_presets == null) return null;
            var match = _presets.Find(p => p.name.Equals(presetName, StringComparison.OrdinalIgnoreCase) ||
                                           p.label.Equals(presetName, StringComparison.OrdinalIgnoreCase));
            if (match == null || string.IsNullOrEmpty(match.data)) return null;

            try
            {
                return Convert.FromBase64String(match.data);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[UniversalSandboxEngine] Failed to decode preset mask '{presetName}': {ex.Message}");
                return null;
            }
        }

        public void PredictSandboxAsync(float[] buffer2Ch, Action<SandboxInferenceResult> callback)
        {
            if (_worker == null)
            {
                Debug.LogError("[UniversalSandboxEngine] Worker not initialized!");
                return;
            }

            StartCoroutine(ExecuteSandboxCoroutine(buffer2Ch, callback));
        }

        private IEnumerator ExecuteSandboxCoroutine(float[] buffer2Ch, Action<SandboxInferenceResult> callback)
        {
            float startTime = Time.realtimeSinceStartup;
            TensorShape inputShape = new TensorShape(1, 2, InputRes, InputRes);

#if UNITY_6000_0_OR_NEWER || UNITY_INFERENCE_ENGINE
            using Tensor<float> inputTensor = new Tensor<float>(inputShape, buffer2Ch);
            _worker.Schedule(inputTensor);
            yield return null;

            Tensor<float> catTensor = _worker.PeekOutput("category_logits") as Tensor<float>;
            Tensor<float> depthTensor = _worker.PeekOutput("inflation_depth") as Tensor<float>;

            if (catTensor == null)
            {
                Debug.LogError("[UniversalSandboxEngine] Failed to peek category_logits tensor.");
                yield break;
            }

            using Tensor<float> catCpu = catTensor.ReadbackAndClone();
            float[] catLogits = catCpu.DownloadToArray();

            float[] depthMap = null;
            if (depthTensor != null)
            {
                using Tensor<float> depthCpu = depthTensor.ReadbackAndClone();
                depthMap = depthCpu.DownloadToArray();
            }
#else
            using TensorFloat inputTensor = new TensorFloat(inputShape, buffer2Ch);
            _worker.Schedule(inputTensor);
            yield return null;

            TensorFloat catTensor = _worker.PeekOutput("category_logits") as TensorFloat;
            TensorFloat depthTensor = _worker.PeekOutput("inflation_depth") as TensorFloat;

            float[] catLogits = catTensor != null ? catTensor.ToReadOnlyArray() : new float[0];
            float[] depthMap = depthTensor != null ? depthTensor.ToReadOnlyArray() : null;
#endif

            // 1. Evaluate Softmax & Top-5 Candidates
            float[] probs = ComputeSoftmax(catLogits);
            int[] sortedIndices = new int[probs.Length];
            for (int i = 0; i < sortedIndices.Length; i++) sortedIndices[i] = i;
            Array.Sort(sortedIndices, (a, b) => probs[b].CompareTo(probs[a]));

            int top1 = sortedIndices[0];
            float top1Prob = probs[top1];

            List<KeyValuePair<string, float>> top5List = new List<KeyValuePair<string, float>>();
            for (int k = 0; k < Mathf.Min(5, sortedIndices.Length); k++)
            {
                int cIdx = sortedIndices[k];
                string name = (_catalog != null && cIdx < _catalog.Count) ? _catalog[cIdx].label : $"Class_{cIdx}";
                top5List.Add(new KeyValuePair<string, float>(name, probs[cIdx]));
            }

            SandboxCategoryInfo catInfo = (_catalog != null && top1 < _catalog.Count) ? _catalog[top1] : null;
            string catName = catInfo != null ? catInfo.name : $"Class_{top1}";
            string catLabel = catInfo != null ? catInfo.label : catName;
            string family = catInfo != null ? catInfo.family : "General";
            bool isNoise = (catName == "ood_scribble") || (top1Prob < 0.25f);

            // 2. Intelligent Morphological Closing & Interior Flood-Fill
            float[] strokeMask = new float[InputRes * InputRes];
            Array.Copy(buffer2Ch, 0, strokeMask, 0, InputRes * InputRes);

            // Dilate by 1 pixel to seal micro-gaps for closed contours
            bool[] dilated = new bool[InputRes * InputRes];
            for (int x = 0; x < InputRes; x++)
            {
                for (int y = 0; y < InputRes; y++)
                {
                    int p = x * InputRes + y;
                    if (strokeMask[p] > 0.25f)
                    {
                        dilated[p] = true;
                        if (x > 0) dilated[(x - 1) * InputRes + y] = true;
                        if (x < InputRes - 1) dilated[(x + 1) * InputRes + y] = true;
                        if (y > 0) dilated[x * InputRes + (y - 1)] = true;
                        if (y < InputRes - 1) dilated[x * InputRes + (y + 1)] = true;
                    }
                }
            }

            // Flood-fill exterior from 4 canvas borders
            bool[] exterior = new bool[InputRes * InputRes];
            Queue<int> q = new Queue<int>(1024);

            for (int x = 0; x < InputRes; x++)
            {
                int p0 = x * InputRes + 0;
                int p1 = x * InputRes + (InputRes - 1);
                if (!dilated[p0] && !exterior[p0]) { exterior[p0] = true; q.Enqueue(p0); }
                if (!dilated[p1] && !exterior[p1]) { exterior[p1] = true; q.Enqueue(p1); }
            }
            for (int y = 0; y < InputRes; y++)
            {
                int p0 = 0 * InputRes + y;
                int p1 = (InputRes - 1) * InputRes + y;
                if (!dilated[p0] && !exterior[p0]) { exterior[p0] = true; q.Enqueue(p0); }
                if (!dilated[p1] && !exterior[p1]) { exterior[p1] = true; q.Enqueue(p1); }
            }

            while (q.Count > 0)
            {
                int p = q.Dequeue();
                int px = p / InputRes;
                int py = p % InputRes;

                if (px > 0) { int np = (px - 1) * InputRes + py; if (!dilated[np] && !exterior[np]) { exterior[np] = true; q.Enqueue(np); } }
                if (px < InputRes - 1) { int np = (px + 1) * InputRes + py; if (!dilated[np] && !exterior[np]) { exterior[np] = true; q.Enqueue(np); } }
                if (py > 0) { int np = px * InputRes + (py - 1); if (!dilated[np] && !exterior[np]) { exterior[np] = true; q.Enqueue(np); } }
                if (py < InputRes - 1) { int np = px * InputRes + (py + 1); if (!dilated[np] && !exterior[np]) { exterior[np] = true; q.Enqueue(np); } }
            }

            // Build solid 2D footprint (Stroke OR enclosed interior)
            bool[] solid2D = new bool[InputRes * InputRes];
            int solidPixels = 0;
            for (int i = 0; i < solid2D.Length; i++)
            {
                if (strokeMask[i] > 0.25f || !exterior[i])
                {
                    solid2D[i] = true;
                    solidPixels++;
                }
            }

            // Compute distance from boundary inside the solid shape for spherical/rounded bulging
            float[] distInside = new float[InputRes * InputRes];
            float maxDist = 1.0f;
            float maxVal = 1e6f;
            for (int i = 0; i < distInside.Length; i++)
            {
                distInside[i] = solid2D[i] ? maxVal : 0.0f;
            }

            float sqrt2 = 1.41421356f;
            for (int x = 0; x < InputRes; x++)
            {
                for (int y = 0; y < InputRes; y++)
                {
                    int idx = x * InputRes + y;
                    if (distInside[idx] > 0f)
                    {
                        float best = distInside[idx];
                        if (x > 0) best = Mathf.Min(best, distInside[(x - 1) * InputRes + y] + 1.0f);
                        if (y > 0) best = Mathf.Min(best, distInside[x * InputRes + (y - 1)] + 1.0f);
                        if (x > 0 && y > 0) best = Mathf.Min(best, distInside[(x - 1) * InputRes + (y - 1)] + sqrt2);
                        if (x < InputRes - 1 && y > 0) best = Mathf.Min(best, distInside[(x + 1) * InputRes + (y - 1)] + sqrt2);
                        distInside[idx] = best;
                    }
                }
            }
            for (int x = InputRes - 1; x >= 0; x--)
            {
                for (int y = InputRes - 1; y >= 0; y--)
                {
                    int idx = x * InputRes + y;
                    if (distInside[idx] > 0f)
                    {
                        float best = distInside[idx];
                        if (x < InputRes - 1) best = Mathf.Min(best, distInside[(x + 1) * InputRes + y] + 1.0f);
                        if (y < InputRes - 1) best = Mathf.Min(best, distInside[x * InputRes + (y + 1)] + 1.0f);
                        if (x < InputRes - 1 && y < InputRes - 1) best = Mathf.Min(best, distInside[(x + 1) * InputRes + (y + 1)] + sqrt2);
                        if (x > 0 && y < InputRes - 1) best = Mathf.Min(best, distInside[(x - 1) * InputRes + (y + 1)] + sqrt2);
                        distInside[idx] = best;
                        if (best > maxDist && best < maxVal) maxDist = best;
                    }
                }
            }

            // 3. Universal 3D Volumetric Infilling (Inflate 2D sketch into 3D voxel grid)
            byte[] occupancyGrid = new byte[TotalVoxels];
            Color32[] voxelColors = new Color32[TotalVoxels];

            Color32 primaryCol = catInfo != null && catInfo.color != null && catInfo.color.Length >= 3
                ? new Color32((byte)catInfo.color[0], (byte)catInfo.color[1], (byte)catInfo.color[2], 255)
                : new Color32(200, 200, 210, 255);

            Color32 accentCol = catInfo != null && catInfo.accent != null && catInfo.accent.Length >= 3
                ? new Color32((byte)catInfo.accent[0], (byte)catInfo.accent[1], (byte)catInfo.accent[2], 255)
                : new Color32(240, 180, 50, 255);

            int solidCount = 0;
            int zCenter = VoxelDim / 2;

            for (int x = 0; x < InputRes; x++)
            {
                for (int y = 0; y < InputRes; y++)
                {
                    int p2d = x * InputRes + y;
                    if (!solid2D[p2d]) continue;

                    float rawD = (depthMap != null && p2d < depthMap.Length) ? depthMap[p2d] : 0.6f;
                    float curvatureD = Mathf.Clamp01(distInside[p2d] / maxDist);
                    float combinedD = 0.45f * rawD + 0.55f * curvatureD;

                    int halfThickness = Mathf.Clamp(Mathf.RoundToInt(combinedD * maxHalfThickness), minHalfThickness, maxHalfThickness);

                    int zMin = Mathf.Clamp(zCenter - halfThickness, 0, VoxelDim - 1);
                    int zMax = Mathf.Clamp(zCenter + halfThickness, 0, VoxelDim - 1);

                    // Stylized accent highlights (rims, outlines, handles, tips)
                    bool isStrokeEdge = strokeMask[p2d] > 0.25f && curvatureD < 0.35f;
                    bool isAccent = isStrokeEdge || (y < 12) || (x < 10) || (combinedD > 0.85f);
                    Color32 pixelColor = isAccent ? accentCol : primaryCol;

                    for (int z = zMin; z <= zMax; z++)
                    {
                        int flatIdx = x * (VoxelDim * VoxelDim) + y * VoxelDim + z;
                        occupancyGrid[flatIdx] = 1;
                        voxelColors[flatIdx] = pixelColor;
                        solidCount++;
                    }
                }
            }

            float latency = (Time.realtimeSinceStartup - startTime) * 1000f;

            SandboxInferenceResult result = new SandboxInferenceResult
            {
                categoryId = top1,
                categoryName = catName,
                label = catLabel,
                family = family,
                confidence = top1Prob,
                top5 = top5List,
                occupancyGrid = occupancyGrid,
                voxelColors = voxelColors,
                latencyMs = latency,
                isNoise = isNoise
            };

            Debug.Log($"[UniversalSandboxEngine] Classified: '{catLabel}' ({family}, {top1Prob * 100f:F1}%) | Voxels: {solidCount} | Latency: {latency:F1}ms");
            OnSandboxInferenceComplete?.Invoke(result);
            callback?.Invoke(result);
        }

        public void EnsureSandboxFloor()
        {
            if (_sandboxFloor != null) return;

            _sandboxFloor = GameObject.Find("SandboxArenaFloor");
            if (_sandboxFloor == null)
            {
                _sandboxFloor = GameObject.CreatePrimitive(PrimitiveType.Cube);
                _sandboxFloor.name = "SandboxArenaFloor";
                _sandboxFloor.transform.position = new Vector3(0, -0.65f, 0);
                _sandboxFloor.transform.localScale = new Vector3(4.5f, 0.1f, 4.5f);

                var rend = _sandboxFloor.GetComponent<MeshRenderer>();
                if (rend != null)
                {
                    Shader sh = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
                    if (sh != null)
                    {
                        Material mat = new Material(sh);
                        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", new Color(0.12f, 0.13f, 0.17f));
                        else if (mat.HasProperty("_Color")) mat.SetColor("_Color", new Color(0.12f, 0.13f, 0.17f));
                        rend.sharedMaterial = mat;
                    }
                }
            }
        }

        public GameObject SpawnPhysicalProp(SandboxInferenceResult result, Vector3 spawnPosition)
        {
            EnsureSandboxFloor();

            // Auto-recycle oldest prop if count exceeds 25
            if (_spawnedProps.Count >= 25)
            {
                if (_spawnedProps[0] != null) Destroy(_spawnedProps[0]);
                _spawnedProps.RemoveAt(0);
            }

            GameObject propObj = new GameObject($"Prop_{result.label}_{Time.frameCount}");
            propObj.transform.position = spawnPosition;
            propObj.transform.rotation = Quaternion.Euler(
                UnityEngine.Random.Range(-10f, 10f),
                UnityEngine.Random.Range(0f, 360f),
                UnityEngine.Random.Range(-10f, 10f)
            );

            MeshFilter mf = propObj.AddComponent<MeshFilter>();
            MeshRenderer mr = propObj.AddComponent<MeshRenderer>();

            Mesh mesh = BuildMeshFromVoxelData(result.occupancyGrid, result.voxelColors);
            mf.sharedMesh = mesh;

            if (vertexColorMaterial != null)
            {
                mr.sharedMaterial = vertexColorMaterial;
            }
            else
            {
                Shader sh = Shader.Find("Custom/VoxelVertexColor") ?? Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Standard");
                if (sh != null) mr.sharedMaterial = new Material(sh);
            }

            if (enablePhysicsOnSpawn && mesh != null && mesh.bounds.size.magnitude > 0.01f)
            {
                BoxCollider col = propObj.AddComponent<BoxCollider>();
                col.size = mesh.bounds.size;
                col.center = mesh.bounds.center;

                Rigidbody rb = propObj.AddComponent<Rigidbody>();
                rb.mass = propMass;
                rb.linearDamping = 0.2f;
                rb.angularDamping = 0.5f;

                // Gentle playful spin
                rb.AddTorque(UnityEngine.Random.insideUnitSphere * 1.5f, ForceMode.Impulse);
            }

            _spawnedProps.Add(propObj);
            Debug.Log($"[UniversalSandboxEngine] Spawned physics prop: {propObj.name} at {spawnPosition}");
            return propObj;
        }

        public void ClearSpawnedProps()
        {
            foreach (var prop in _spawnedProps)
            {
                if (prop != null) Destroy(prop);
            }
            _spawnedProps.Clear();
            Debug.Log("[UniversalSandboxEngine] Cleared all spawned sandbox props.");
        }

        private Mesh BuildMeshFromVoxelData(byte[] occupancy, Color32[] colors)
        {
            if (occupancy == null || colors == null || occupancy.Length != TotalVoxels) return null;

            NativeArray<byte> nativeOcc = new NativeArray<byte>(occupancy, Allocator.TempJob);
            NativeArray<Color32> nativeColors = new NativeArray<Color32>(colors, Allocator.TempJob);

            NativeList<float3> vertices = new NativeList<float3>(Allocator.TempJob);
            NativeList<int> triangles = new NativeList<int>(Allocator.TempJob);
            NativeList<float3> normals = new NativeList<float3>(Allocator.TempJob);
            NativeList<Color32> outColors = new NativeList<Color32>(Allocator.TempJob);

            var job = new ColoredVoxelJobs.ExtractColoredSurfaceQuadsJob
            {
                OccupancyGrid = nativeOcc,
                VoxelColors = nativeColors,
                VoxelScale = voxelScale,
                OutVertices = vertices,
                OutTriangles = triangles,
                OutNormals = normals,
                OutColors = outColors
            };

            JobHandle handle = job.Schedule();
            handle.Complete();

            Mesh mesh = new Mesh { name = "ProceduralSandboxProp" };

            if (vertices.Length > 0)
            {
                Vector3[] managedVerts = new Vector3[vertices.Length];
                Vector3[] managedNorms = new Vector3[normals.Length];
                Color32[] managedColors = new Color32[outColors.Length];
                int[] managedTris = new int[triangles.Length];

                for (int i = 0; i < vertices.Length; i++)
                {
                    managedVerts[i] = vertices[i];
                    managedNorms[i] = normals[i];
                    managedColors[i] = outColors[i];
                }
                for (int i = 0; i < triangles.Length; i++)
                {
                    managedTris[i] = triangles[i];
                }

                mesh.vertices = managedVerts;
                mesh.normals = managedNorms;
                mesh.colors32 = managedColors;
                mesh.triangles = managedTris;
                mesh.RecalculateBounds();

                // Centering around mesh origin for clean physics rotation
                Vector3 center = mesh.bounds.center;
                for (int i = 0; i < managedVerts.Length; i++)
                {
                    managedVerts[i] -= center;
                }
                mesh.vertices = managedVerts;
                mesh.RecalculateBounds();
            }

            nativeOcc.Dispose();
            nativeColors.Dispose();
            vertices.Dispose();
            triangles.Dispose();
            normals.Dispose();
            outColors.Dispose();

            return mesh;
        }

        private float[] ComputeSoftmax(float[] logits)
        {
            if (logits == null || logits.Length == 0) return new float[0];
            float max = logits[0];
            for (int i = 1; i < logits.Length; i++) if (logits[i] > max) max = logits[i];

            float sum = 0f;
            float[] probs = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                probs[i] = Mathf.Exp(logits[i] - max);
                sum += probs[i];
            }
            if (sum > 1e-6f)
            {
                for (int i = 0; i < logits.Length; i++) probs[i] /= sum;
            }
            return probs;
        }
    }
}
