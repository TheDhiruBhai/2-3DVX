using UnityEngine;
using VRDraw2Voxel.Inference;
using VRDraw2Voxel.VoxelMesh;

namespace VRDraw2Voxel.Tests
{
    /// <summary>
    /// Test harness for testing the Draw2Voxel ONNX model directly inside the Unity Editor
    /// or runtime without requiring an active VR headset.
    /// </summary>
    public class ModelInferenceTester : MonoBehaviour
    {
        [Header("Components")]
        [SerializeField] private SentisVoxelEngine voxelEngine;
        [SerializeField] private GreedyVoxelMesher voxelMesher;

        [Header("Test Options")]
        [SerializeField] private bool testOnStart = true;
        [SerializeField] private TestPattern testShape = TestPattern.Sword;

        public enum TestPattern
        {
            Sword,
            Hammer,
            Cross
        }

        private void Start()
        {
            if (testOnStart)
            {
                RunTestInference();
            }
        }

        [ContextMenu("Run Test Inference")]
        public void RunTestInference()
        {
            if (voxelEngine == null)
            {
                voxelEngine = FindFirstObjectByType<SentisVoxelEngine>();
            }
            if (voxelMesher == null)
            {
                voxelMesher = FindFirstObjectByType<GreedyVoxelMesher>();
            }

            if (voxelEngine == null)
            {
                Debug.LogError("[ModelInferenceTester] No SentisVoxelEngine found in scene.");
                return;
            }

            Debug.Log($"[ModelInferenceTester] Generating synthetic {testShape} sketch for testing...");
            float[] testBuffer = GenerateTestSketch(testShape);

            float startTime = Time.realtimeSinceStartup;
            voxelEngine.PredictVoxelsAsync(testBuffer, (voxels) =>
            {
                float duration = (Time.realtimeSinceStartup - startTime) * 1000f;
                int solidCount = 0;
                for (int i = 0; i < voxels.Length; i++)
                {
                    if (voxels[i] > 0) solidCount++;
                }

                Debug.Log($"[ModelInferenceTester] SUCCESS! Prediction completed in {duration:F2} ms. Solid voxels: {solidCount} / {voxels.Length}.");
                if (voxelMesher != null)
                {
                    voxelMesher.GenerateMeshFromOccupancy(voxels);
                }
            });
        }

        private float[] GenerateTestSketch(TestPattern pattern)
        {
            float[] buffer = new float[64 * 64];

            switch (pattern)
            {
                case TestPattern.Sword:
                    // Blade
                    for (int y = 20; y <= 58; y++)
                    {
                        for (int x = 30; x <= 33; x++)
                        {
                            buffer[y * 64 + x] = 1.0f;
                        }
                    }
                    // Crossguard
                    for (int x = 22; x <= 41; x++)
                    {
                        for (int y = 18; y <= 20; y++)
                        {
                            buffer[y * 64 + x] = 1.0f;
                        }
                    }
                    // Handle & Pommel
                    for (int y = 6; y <= 18; y++)
                    {
                        buffer[y * 64 + 31] = 1.0f;
                        buffer[y * 64 + 32] = 1.0f;
                    }
                    break;

                case TestPattern.Hammer:
                    // Handle
                    for (int y = 8; y <= 45; y++)
                    {
                        buffer[y * 64 + 31] = 1.0f;
                        buffer[y * 64 + 32] = 1.0f;
                    }
                    // Head
                    for (int y = 42; y <= 54; y++)
                    {
                        for (int x = 20; x <= 44; x++)
                        {
                            buffer[y * 64 + x] = 1.0f;
                        }
                    }
                    break;

                case TestPattern.Cross:
                default:
                    for (int i = 12; i <= 52; i++)
                    {
                        buffer[i * 64 + 32] = 1.0f;
                        buffer[32 * 64 + i] = 1.0f;
                    }
                    break;
            }

            return buffer;
        }
    }
}
