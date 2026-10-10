using System.Collections.Generic;
using UnityEngine;

namespace VRDraw2Voxel.Drawing
{
    /// <summary>
    /// Projects 3D hand-drawn strokes onto a player-facing 2D plane and rasterizes them
    /// into a 1x1x64x64 binary input buffer for Sentis deep learning inference.
    /// </summary>
    public class DrawingCanvasProjector : MonoBehaviour
    {
        public const int RasterResolution = 64;

        [Header("Rasterization Settings")]
        [SerializeField] private int strokeBrushRadius = 2; // pixel radius for drawing stroke width

        /// <summary>
        /// Projects a series of 3D world stroke points onto a 2D plane facing the camera,
        /// normalizes the bounds, and returns a 64x64 float array (0.0 = background, 1.0 = stroke).
        /// </summary>
        public float[] RasterizeStrokes(List<Vector3> worldPoints, Transform observerTransform)
        {
            float[] buffer = new float[RasterResolution * RasterResolution];
            if (worldPoints == null || worldPoints.Count < 2)
            {
                Debug.LogWarning("[DrawingCanvasProjector] Insufficient points for rasterization.");
                return buffer;
            }

            // 1. Establish projection plane using camera's orientation
            Vector3 planeNormal = observerTransform != null ? observerTransform.forward : Vector3.forward;
            Vector3 planeUp = observerTransform != null ? observerTransform.up : Vector3.up;
            Vector3 planeRight = Vector3.Cross(planeUp, planeNormal).normalized;
            planeUp = Vector3.Cross(planeNormal, planeRight).normalized;

            // 2. Project 3D points to 2D plane coordinates
            Vector3 origin = worldPoints[0];
            List<Vector2> pts2D = new List<Vector2>(worldPoints.Count);
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;

            for (int i = 0; i < worldPoints.Count; i++)
            {
                Vector3 delta = worldPoints[i] - origin;
                float x = Vector3.Dot(delta, planeRight);
                float y = Vector3.Dot(delta, planeUp);
                pts2D.Add(new Vector2(x, y));

                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }

            // 3. Compute normalized bounds with aspect ratio preservation and padding
            float width = Mathf.Max(maxX - minX, 0.001f);
            float height = Mathf.Max(maxY - minY, 0.001f);
            float maxDim = Mathf.Max(width, height);
            
            float centerX = (minX + maxX) * 0.5f;
            float centerY = (minY + maxY) * 0.5f;

            int pad = 6; // Margin in 64x64 grid
            float drawArea = RasterResolution - (pad * 2);

            // 4. Rasterize line segments between consecutive points using Bresenham's algorithm
            for (int i = 0; i < pts2D.Count - 1; i++)
            {
                Vector2 p0 = pts2D[i];
                Vector2 p1 = pts2D[i + 1];

                int px0 = Mathf.RoundToInt(((p0.x - centerX) / maxDim) * drawArea + (RasterResolution * 0.5f));
                int py0 = Mathf.RoundToInt(((p0.y - centerY) / maxDim) * drawArea + (RasterResolution * 0.5f));
                int px1 = Mathf.RoundToInt(((p1.x - centerX) / maxDim) * drawArea + (RasterResolution * 0.5f));
                int py1 = Mathf.RoundToInt(((p1.y - centerY) / maxDim) * drawArea + (RasterResolution * 0.5f));

                DrawBresenhamLine(buffer, px0, py0, px1, py1, strokeBrushRadius);
            }

            return buffer;
        }

        private void DrawBresenhamLine(float[] buffer, int x0, int y0, int x1, int y1, int radius)
        {
            int dx = Mathf.Abs(x1 - x0);
            int dy = Mathf.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;

            while (true)
            {
                StampBrush(buffer, x0, y0, radius);

                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 > -dy)
                {
                    err -= dy;
                    x0 += sx;
                }
                if (e2 < dx)
                {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        private void StampBrush(float[] buffer, int cx, int cy, int radius)
        {
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx * dx + dy * dy <= radius * radius)
                    {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < RasterResolution && py >= 0 && py < RasterResolution)
                        {
                            buffer[py * RasterResolution + px] = 1.0f;
                        }
                    }
                }
            }
        }
    }
}
