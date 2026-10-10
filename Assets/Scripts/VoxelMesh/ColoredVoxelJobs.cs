using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace VRDraw2Voxel.VoxelMesh
{
    /// <summary>
    /// Burst-compiled surface extraction for 64x64x64 colored voxel grids.
    /// Emits vertices, normals, triangles, and Color32 vertex colors,
    /// enabling real-time zero-texture rendering with native URP shaders.
    /// </summary>
    public static class ColoredVoxelJobs
    {
        public const int Dim = 64;

        public static int ToFlatIndex(int x, int y, int z)
        {
            return x * (Dim * Dim) + y * Dim + z;
        }

        [BurstCompile]
        public struct ExtractColoredSurfaceQuadsJob : IJob
        {
            [ReadOnly] public NativeArray<byte> OccupancyGrid;
            [ReadOnly] public NativeArray<Color32> VoxelColors;
            public float VoxelScale;

            public NativeList<float3> OutVertices;
            public NativeList<int> OutTriangles;
            public NativeList<float3> OutNormals;
            public NativeList<Color32> OutColors;

            public void Execute()
            {
                for (int x = 0; x < Dim; x++)
                {
                    for (int y = 0; y < Dim; y++)
                    {
                        for (int z = 0; z < Dim; z++)
                        {
                            int idx = ToFlatIndex(x, y, z);
                            if (OccupancyGrid[idx] == 0) continue;

                            Color32 color = VoxelColors[idx];
                            float3 center = new float3(
                                (x - Dim * 0.5f) * VoxelScale,
                                (y - Dim * 0.5f) * VoxelScale,
                                (z - Dim * 0.5f) * VoxelScale
                            );
                            float half = VoxelScale * 0.5f;

                            // +X Face
                            if (x + 1 >= Dim || OccupancyGrid[ToFlatIndex(x + 1, y, z)] == 0)
                            {
                                AddQuad(
                                    center + new float3(half, -half, -half),
                                    center + new float3(half, half, -half),
                                    center + new float3(half, half, half),
                                    center + new float3(half, -half, half),
                                    new float3(1, 0, 0),
                                    color
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
                                    new float3(-1, 0, 0),
                                    color
                                );
                            }

                            // +Y Face
                            if (y + 1 >= Dim || OccupancyGrid[ToFlatIndex(x, y + 1, z)] == 0)
                            {
                                AddQuad(
                                    center + new float3(-half, half, -half),
                                    center + new float3(half, half, -half),
                                    center + new float3(half, half, half),
                                    center + new float3(-half, half, half),
                                    new float3(0, 1, 0),
                                    color
                                );
                            }

                            // -Y Face
                            if (y - 1 < 0 || OccupancyGrid[ToFlatIndex(x, y - 1, z)] == 0)
                            {
                                AddQuad(
                                    center + new float3(-half, -half, half),
                                    center + new float3(half, -half, half),
                                    center + new float3(half, -half, -half),
                                    center + new float3(-half, -half, -half),
                                    new float3(0, -1, 0),
                                    color
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
                                    new float3(0, 0, 1),
                                    color
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
                                    new float3(0, 0, -1),
                                    color
                                );
                            }
                        }
                    }
                }
            }

            private void AddQuad(float3 v0, float3 v1, float3 v2, float3 v3, float3 normal, Color32 color)
            {
                int baseIndex = OutVertices.Length;
                OutVertices.Add(v0);
                OutVertices.Add(v1);
                OutVertices.Add(v2);
                OutVertices.Add(v3);

                OutNormals.Add(normal);
                OutNormals.Add(normal);
                OutNormals.Add(normal);
                OutNormals.Add(normal);

                OutColors.Add(color);
                OutColors.Add(color);
                OutColors.Add(color);
                OutColors.Add(color);

                OutTriangles.Add(baseIndex + 0);
                OutTriangles.Add(baseIndex + 1);
                OutTriangles.Add(baseIndex + 2);

                OutTriangles.Add(baseIndex + 0);
                OutTriangles.Add(baseIndex + 2);
                OutTriangles.Add(baseIndex + 3);
            }
        }
    }
}
