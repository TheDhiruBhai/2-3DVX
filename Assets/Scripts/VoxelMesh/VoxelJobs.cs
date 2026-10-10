using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace VRDraw2Voxel.VoxelMesh
{
    /// <summary>
    /// Burst-compiled Jobs for high-performance surface quad extraction from a 32x32x32 occupancy grid.
    /// Runs well within the 3.0 ms budget on Meta Quest chipsets.
    /// </summary>
    public static class VoxelJobs
    {
        public const int Dim = 32;

        public static int ToFlatIndex(int x, int y, int z)
        {
            return x * (Dim * Dim) + y * Dim + z;
        }

        [BurstCompile]
        public struct ExtractSurfaceQuadsJob : IJob
        {
            [ReadOnly] public NativeArray<byte> OccupancyGrid;
            public float VoxelScale;

            public NativeList<float3> OutVertices;
            public NativeList<int> OutTriangles;
            public NativeList<float3> OutNormals;

            public void Execute()
            {
                // Iterate through all 32x32x32 voxels
                for (int z = 0; z < Dim; z++)
                {
                    for (int y = 0; y < Dim; y++)
                    {
                        for (int x = 0; x < Dim; x++)
                        {
                            int idx = ToFlatIndex(x, y, z);
                            if (OccupancyGrid[idx] == 0) continue;

                            float3 center = new float3(
                                (x - Dim * 0.5f) * VoxelScale,
                                (y - Dim * 0.5f) * VoxelScale,
                                (z - Dim * 0.5f) * VoxelScale
                            );
                            float half = VoxelScale * 0.5f;

                            // Check all 6 face neighbors. If neighbor is air (0), add quad face.
                            
                            // +X Face
                            if (x + 1 >= Dim || OccupancyGrid[ToFlatIndex(x + 1, y, z)] == 0)
                            {
                                AddQuad(
                                    center + new float3(half, -half, -half),
                                    center + new float3(half, half, -half),
                                    center + new float3(half, half, half),
                                    center + new float3(half, -half, half),
                                    new float3(1, 0, 0)
                                );
                            }

                            // -X Face
                            if (x - 1 < 0 || OccupancyGrid[ToFlatIndex(x - 1, y, z)] == 0)
                            {
                                AddQuad(
                                    center + new float3(-half, -half, half),
                                    center + new float3(-half, half, half),
                                    center + new float3(-half, half, -half),
                                    center + new float3(-half, -half, -half),
                                    new float3(-1, 0, 0)
                                );
                            }

                            // +Y Face
                            if (y + 1 >= Dim || OccupancyGrid[ToFlatIndex(x, y + 1, z)] == 0)
                            {
                                AddQuad(
                                    center + new float3(-half, half, -half),
                                    center + new float3(-half, half, half),
                                    center + new float3(half, half, half),
                                    center + new float3(half, half, -half),
                                    new float3(0, 1, 0)
                                );
                            }

                            // -Y Face
                            if (y - 1 < 0 || OccupancyGrid[ToFlatIndex(x, y - 1, z)] == 0)
                            {
                                AddQuad(
                                    center + new float3(-half, -half, half),
                                    center + new float3(-half, -half, -half),
                                    center + new float3(half, -half, -half),
                                    center + new float3(half, -half, half),
                                    new float3(0, -1, 0)
                                );
                            }

                            // +Z Face
                            if (z + 1 >= Dim || OccupancyGrid[ToFlatIndex(x, y, z + 1)] == 0)
                            {
                                AddQuad(
                                    center + new float3(half, -half, half),
                                    center + new float3(half, half, half),
                                    center + new float3(-half, half, half),
                                    center + new float3(-half, -half, half),
                                    new float3(0, 0, 1)
                                );
                            }

                            // -Z Face
                            if (z - 1 < 0 || OccupancyGrid[ToFlatIndex(x, y, z - 1)] == 0)
                            {
                                AddQuad(
                                    center + new float3(-half, -half, -half),
                                    center + new float3(-half, half, -half),
                                    center + new float3(half, half, -half),
                                    center + new float3(half, -half, -half),
                                    new float3(0, 0, -1)
                                );
                            }
                        }
                    }
                }
            }

            private void AddQuad(float3 v0, float3 v1, float3 v2, float3 v3, float3 norm)
            {
                int baseIdx = OutVertices.Length;
                OutVertices.Add(v0);
                OutVertices.Add(v1);
                OutVertices.Add(v2);
                OutVertices.Add(v3);

                OutNormals.Add(norm);
                OutNormals.Add(norm);
                OutNormals.Add(norm);
                OutNormals.Add(norm);

                OutTriangles.Add(baseIdx);
                OutTriangles.Add(baseIdx + 1);
                OutTriangles.Add(baseIdx + 2);

                OutTriangles.Add(baseIdx);
                OutTriangles.Add(baseIdx + 2);
                OutTriangles.Add(baseIdx + 3);
            }
        }
    }
}
