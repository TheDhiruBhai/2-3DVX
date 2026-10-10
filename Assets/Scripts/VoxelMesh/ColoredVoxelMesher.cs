using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using VRDraw2Voxel.Inference;

namespace VRDraw2Voxel.VoxelMesh
{
    /// <summary>
    /// Generates physical Unity GameObjects from 64x64x64 colored voxel grids
    /// using Burst-compiled ColoredVoxelJobs with vertex color rendering.
    /// </summary>
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class ColoredVoxelMesher : MonoBehaviour
    {
        [Header("Mesh Settings")]
        [SerializeField] private float voxelScale = 0.015f; // 1.5 cm per voxel -> ~96 cm weapon length
        [SerializeField] private Material vertexColorMaterial;
        [SerializeField] private SentisMultiTaskEngine multiTaskEngine;

        private MeshFilter _meshFilter;
        private MeshRenderer _meshRenderer;
        private Mesh _mesh;

        public event Action<GameObject, Bounds, MultiTaskInferenceResult> OnColoredMeshConstructed;

        private void Awake()
        {
            _meshFilter = GetComponent<MeshFilter>();
            _meshRenderer = GetComponent<MeshRenderer>();
            _mesh = new Mesh { name = "ProceduralColoredVoxelItem" };
            _meshFilter.sharedMesh = _mesh;

            if (vertexColorMaterial != null)
            {
                _meshRenderer.sharedMaterial = vertexColorMaterial;
            }
            else
            {
                // Use shader that natively supports per-vertex colors in URP
                Shader shader = Shader.Find("Universal Render Pipeline/Particles/Lit")
                             ?? Shader.Find("Universal Render Pipeline/Unlit")
                             ?? Shader.Find("Sprites/Default")
                             ?? Shader.Find("Universal Render Pipeline/Lit")
                             ?? Shader.Find("Standard");
                if (shader != null)
                {
                    Material mat = new Material(shader) { name = "VoxelVertexColorMat" };
                    if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", Color.white);
                    if (mat.HasProperty("_Color")) mat.SetColor("_Color", Color.white);
                    if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", 0.5f);
                    _meshRenderer.sharedMaterial = mat;
                }
            }

            if (multiTaskEngine == null)
            {
                multiTaskEngine = FindFirstObjectByType<SentisMultiTaskEngine>();
            }

            if (multiTaskEngine != null)
            {
                multiTaskEngine.OnInferenceComplete += GenerateFromInferenceResult;
            }
        }

        private void OnDestroy()
        {
            if (multiTaskEngine != null)
            {
                multiTaskEngine.OnInferenceComplete -= GenerateFromInferenceResult;
            }
        }

        public void GenerateFromInferenceResult(MultiTaskInferenceResult result)
        {
            if (result.isOutOfDistribution)
            {
                Debug.LogWarning("[ColoredVoxelMesher] Skipping mesh generation: Input sketch was rejected as Out-Of-Distribution!");
                return;
            }

            GenerateColoredMesh(result.occupancyGrid, result.voxelColors);
            OnColoredMeshConstructed?.Invoke(gameObject, _mesh.bounds, result);
        }

        public void GenerateColoredMesh(byte[] occupancy, Color32[] colors)
        {
            if (occupancy == null || colors == null || occupancy.Length != ColoredVoxelJobs.Dim * ColoredVoxelJobs.Dim * ColoredVoxelJobs.Dim)
            {
                Debug.LogError("[ColoredVoxelMesher] Invalid grid dimensions for 64^3 mesher.");
                return;
            }

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
            handle.Complete(); // Executes within frame budget

            _mesh.Clear();
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

                _mesh.vertices = managedVerts;
                _mesh.normals = managedNorms;
                _mesh.colors32 = managedColors;
                _mesh.triangles = managedTris;
                _mesh.RecalculateBounds();

                // Auto-center mesh vertices around centroid for smooth 360 showcase rotation
                Vector3 center = _mesh.bounds.center;
                for (int i = 0; i < managedVerts.Length; i++)
                {
                    managedVerts[i] -= center;
                }
                _mesh.vertices = managedVerts;
                _mesh.RecalculateBounds();

                Debug.Log($"[ColoredVoxelMesher] Generated colored mesh: {vertices.Length} verts, {triangles.Length/3} tris.");
            }

            nativeOcc.Dispose();
            nativeColors.Dispose();
            vertices.Dispose();
            triangles.Dispose();
            normals.Dispose();
            outColors.Dispose();
        }
    }
}
