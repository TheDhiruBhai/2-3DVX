using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using VRDraw2Voxel.Inference;

namespace VRDraw2Voxel.VoxelMesh
{
    /// <summary>
    /// Converts a 32x32x32 occupancy grid into an optimized 3D physical GameObject
    /// using Burst-compiled C# Jobs, fitting within the 3.0 ms VR budget.
    /// </summary>
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
    public class GreedyVoxelMesher : MonoBehaviour
    {
        [Header("Mesh Settings")]
        [SerializeField] private float voxelScale = 0.025f; // 2.5 cm per voxel -> ~80 cm weapon length
        [SerializeField] private Material weaponMaterial;
        [SerializeField] private SentisVoxelEngine voxelEngine;

        private MeshFilter _meshFilter;
        private MeshRenderer _meshRenderer;
        private Mesh _mesh;

        public event Action<GameObject, Bounds> OnWeaponMeshConstructed;

        private void Awake()
        {
            _meshFilter = GetComponent<MeshFilter>();
            _meshRenderer = GetComponent<MeshRenderer>();
            _mesh = new Mesh { name = "ProceduralVoxelWeapon" };
            _meshFilter.sharedMesh = _mesh;

            if (weaponMaterial != null)
            {
                _meshRenderer.sharedMaterial = weaponMaterial;
            }
            else
            {
                Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard") ?? Shader.Find("Diffuse");
                if (shader != null)
                {
                    Material defaultMat = new Material(shader)
                    {
                        color = new Color(0.95f, 0.8f, 0.35f) // Golden PS2 achievement aesthetic
                    };
                    if (defaultMat.HasProperty("_Metallic")) defaultMat.SetFloat("_Metallic", 0.85f);
                    if (defaultMat.HasProperty("_Smoothness")) defaultMat.SetFloat("_Smoothness", 0.7f);
                    _meshRenderer.sharedMaterial = defaultMat;
                }
            }

            if (voxelEngine != null)
            {
                voxelEngine.OnVoxelGridReady += GenerateMeshFromOccupancy;
            }
        }

        private void OnDestroy()
        {
            if (voxelEngine != null)
            {
                voxelEngine.OnVoxelGridReady -= GenerateMeshFromOccupancy;
            }
        }

        public void GenerateMeshFromOccupancy(byte[] occupancyBytes)
        {
            if (occupancyBytes == null || occupancyBytes.Length != VoxelJobs.Dim * VoxelJobs.Dim * VoxelJobs.Dim)
            {
                Debug.LogError("[GreedyVoxelMesher] Invalid occupancy byte array.");
                return;
            }

            NativeArray<byte> nativeGrid = new NativeArray<byte>(occupancyBytes, Allocator.TempJob);
            NativeList<float3> vertices = new NativeList<float3>(Allocator.TempJob);
            NativeList<int> triangles = new NativeList<int>(Allocator.TempJob);
            NativeList<float3> normals = new NativeList<float3>(Allocator.TempJob);

            var job = new VoxelJobs.ExtractSurfaceQuadsJob
            {
                OccupancyGrid = nativeGrid,
                VoxelScale = voxelScale,
                OutVertices = vertices,
                OutTriangles = triangles,
                OutNormals = normals
            };

            JobHandle handle = job.Schedule();
            handle.Complete(); // Completed well within frame budget

            // Apply to Unity Mesh
            _mesh.Clear();
            if (vertices.Length > 0)
            {
                Vector3[] managedVerts = new Vector3[vertices.Length];
                Vector3[] managedNorms = new Vector3[normals.Length];
                int[] managedTris = new int[triangles.Length];

                for (int i = 0; i < vertices.Length; i++)
                {
                    managedVerts[i] = vertices[i];
                    managedNorms[i] = normals[i];
                }
                for (int i = 0; i < triangles.Length; i++)
                {
                    managedTris[i] = triangles[i];
                }

                _mesh.vertices = managedVerts;
                _mesh.normals = managedNorms;
                _mesh.triangles = managedTris;
                _mesh.RecalculateBounds();

                // Auto-center mesh around its geometric pivot for perfect 360 showcase rotation
                Vector3 center = _mesh.bounds.center;
                for (int i = 0; i < managedVerts.Length; i++)
                {
                    managedVerts[i] -= center;
                }
                _mesh.vertices = managedVerts;
                _mesh.RecalculateBounds();

                // Setup Compound Box Collider or main BoxCollider for physics
                SetupPhysicsColliders(_mesh.bounds);

                Debug.Log($"[GreedyVoxelMesher] Generated mesh with {vertices.Length} vertices, {triangles.Length / 3} triangles.");
                OnWeaponMeshConstructed?.Invoke(gameObject, _mesh.bounds);
            }

            // Cleanup NativeArrays
            nativeGrid.Dispose();
            vertices.Dispose();
            triangles.Dispose();
            normals.Dispose();
        }

        private void SetupPhysicsColliders(Bounds bounds)
        {
            BoxCollider box = GetComponent<BoxCollider>();
            if (box == null)
            {
                box = gameObject.AddComponent<BoxCollider>();
            }
            box.center = bounds.center;
            box.size = bounds.size;
        }
    }
}
