using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace VRDraw2Voxel.Drawing
{
    /// <summary>
    /// Milestone 5 VR Tabletop Canvas & Spatial Drawing.
    /// Placed on an in-world table Quad. Tracks OpenXR hand joints:
    /// - Index tip to Thumb tip distance < 2.0 cm (Pinch Gesture)
    /// - Index tip distance to board surface < 1.5 cm (Hover/Touch)
    /// Interpolates continuous brush strokes into a dynamic 64x64 texture,
    /// enforcing the verified coordinate mapping I_Unity = [flipud(I)]^T.
    /// </summary>
    [RequireComponent(typeof(MeshRenderer), typeof(Collider))]
    public class VRTabletopCanvas : MonoBehaviour
    {
        [Header("Canvas Resolution")]
        public const int Resolution = 64;

        [Header("Hand Tracking & Spatial Pen")]
        [SerializeField] private Handedness drawingHand = Handedness.Right;
        [SerializeField] private float pinchThresholdMeters = 0.020f; // 2.0 cm
        [SerializeField] private float hoverThresholdMeters = 0.015f; // 1.5 cm
        [SerializeField] private int brushRadius = 1;                 // 1-2 pixels on 64x64
        [SerializeField] private Color32 inkColor = new Color32(255, 255, 255, 255);
        [SerializeField] private Color32 slateColor = new Color32(20, 24, 30, 255);

        [Header("Visual Feedback")]
        [SerializeField] private Transform brushCursorIndicator;
        [SerializeField] private Material canvasMaterial;

        // Events
        public event Action OnStrokeBegun;
        public event Action OnStrokeFinished;
        public event Action OnCanvasCleared;

        // Internal State
        private Texture2D _drawTexture;
        private Color32[] _pixelBuffer;
        private byte[] _strokeMask; // 64x64 uint8 (0 or 255)
        private Vector2Int? _lastDrawPixel;
        private bool _isPinchingAndHovering;
        private XRHandSubsystem _handSubsystem;
        private static readonly List<XRHandSubsystem> s_Subsystems = new List<XRHandSubsystem>();

        public Texture2D DrawTexture => _drawTexture;
        public bool IsDrawing => _isPinchingAndHovering;
        public int ActiveStrokePixels { get; private set; }

        private void Awake()
        {
            InitializeCanvasTexture();
        }

        private void OnEnable()
        {
            FindHandSubsystem();
        }

        private void Update()
        {
            if (_handSubsystem == null || !_handSubsystem.running)
            {
                FindHandSubsystem();
            }

            bool strokeDrawnThisFrame = false;

            if (_handSubsystem != null && _handSubsystem.running)
            {
                strokeDrawnThisFrame = PollVRHandDrawing();
            }
            else
            {
                strokeDrawnThisFrame = PollEditorMouseDrawing();
            }

            if (!strokeDrawnThisFrame)
            {
                if (_isPinchingAndHovering)
                {
                    _isPinchingAndHovering = false;
                    _lastDrawPixel = null;
                    OnStrokeFinished?.Invoke();
                }
                if (brushCursorIndicator != null)
                {
                    brushCursorIndicator.gameObject.SetActive(false);
                }
            }
        }

        private void InitializeCanvasTexture()
        {
            _drawTexture = new Texture2D(Resolution, Resolution, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };

            _pixelBuffer = new Color32[Resolution * Resolution];
            _strokeMask = new byte[Resolution * Resolution];

            ClearCanvas();

            MeshRenderer mr = GetComponent<MeshRenderer>();
            if (canvasMaterial != null)
            {
                mr.material = new Material(canvasMaterial);
            }
            mr.material.mainTexture = _drawTexture;
        }

        private void FindHandSubsystem()
        {
            SubsystemManager.GetSubsystems(s_Subsystems);
            for (int i = 0; i < s_Subsystems.Count; i++)
            {
                if (s_Subsystems[i].running)
                {
                    _handSubsystem = s_Subsystems[i];
                    break;
                }
            }
        }

        /// <summary>
        /// Reads OpenXR Hand joints: evaluates index-thumb pinch and distance to canvas quad.
        /// </summary>
        private bool PollVRHandDrawing()
        {
            XRHand hand = (drawingHand == Handedness.Right) ? _handSubsystem.rightHand : _handSubsystem.leftHand;
            if (!hand.isTracked) return false;

            XRHandJoint indexTip = hand.GetJoint(XRHandJointID.IndexTip);
            XRHandJoint thumbTip = hand.GetJoint(XRHandJointID.ThumbTip);

            if (!indexTip.TryGetPose(out Pose indexPose) || !thumbTip.TryGetPose(out Pose thumbPose))
            {
                return false;
            }

            // 1. Pinch Condition: Index Tip to Thumb Tip < 2.0 cm
            float pinchDist = Vector3.Distance(indexPose.position, thumbPose.position);
            bool isPinching = pinchDist <= pinchThresholdMeters;

            // 2. Hover Condition: Distance to Board Surface < 1.5 cm
            Vector3 localIndexPos = transform.InverseTransformPoint(indexPose.position);
            float distToSurface = Mathf.Abs(localIndexPos.z);
            bool isHovering = distToSurface <= hoverThresholdMeters &&
                              localIndexPos.x >= -0.5f && localIndexPos.x <= 0.5f &&
                              localIndexPos.y >= -0.5f && localIndexPos.y <= 0.5f;

            if (brushCursorIndicator != null)
            {
                brushCursorIndicator.gameObject.SetActive(isHovering);
                if (isHovering)
                {
                    Vector3 contactLocal = new Vector3(localIndexPos.x, localIndexPos.y, 0f);
                    brushCursorIndicator.position = transform.TransformPoint(contactLocal);
                }
            }

            if (isPinching && isHovering)
            {
                // Compute normalized UV coordinates on quad (local X in [-0.5, 0.5], local Y in [-0.5, 0.5])
                float u = localIndexPos.x + 0.5f;
                float v = localIndexPos.y + 0.5f;

                int px = Mathf.Clamp((int)(u * Resolution), 0, Resolution - 1);
                int py = Mathf.Clamp((int)(v * Resolution), 0, Resolution - 1);

                DrawAtPixel(px, py);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Mouse drawing simulation in Editor when VR headset is not connected.
        /// </summary>
        private bool PollEditorMouseDrawing()
        {
            if (!Input.GetMouseButton(0)) return false;

            Camera cam = Camera.main;
            if (cam == null) return false;

            Ray ray = cam.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                if (hit.collider == GetComponent<Collider>() || hit.transform == transform)
                {
                    Vector2 uv = hit.textureCoord;
                    int px = Mathf.Clamp((int)(uv.x * Resolution), 0, Resolution - 1);
                    int py = Mathf.Clamp((int)(uv.y * Resolution), 0, Resolution - 1);

                    if (brushCursorIndicator != null)
                    {
                        brushCursorIndicator.gameObject.SetActive(true);
                        brushCursorIndicator.position = hit.point;
                    }

                    DrawAtPixel(px, py);
                    return true;
                }
            }

            return false;
        }

        private void DrawAtPixel(int px, int py)
        {
            if (!_isPinchingAndHovering)
            {
                _isPinchingAndHovering = true;
                OnStrokeBegun?.Invoke();
                DrawStamp(px, py);
                _lastDrawPixel = new Vector2Int(px, py);
            }
            else
            {
                if (_lastDrawPixel.HasValue)
                {
                    InterpolateLine(_lastDrawPixel.Value.x, _lastDrawPixel.Value.y, px, py);
                }
                else
                {
                    DrawStamp(px, py);
                }
                _lastDrawPixel = new Vector2Int(px, py);
            }

            _drawTexture.SetPixels32(_pixelBuffer);
            _drawTexture.Apply(false);
        }

        private void DrawStamp(int cx, int cy)
        {
            int rSq = brushRadius * brushRadius;
            for (int dy = -brushRadius; dy <= brushRadius; dy++)
            {
                for (int dx = -brushRadius; dx <= brushRadius; dx++)
                {
                    if (dx * dx + dy * dy <= rSq)
                    {
                        int x = cx + dx;
                        int y = cy + dy;
                        if (x >= 0 && x < Resolution && y >= 0 && y < Resolution)
                        {
                            // In Texture2D, rows are indexed as y * Resolution + x
                            int idx = y * Resolution + x;
                            _pixelBuffer[idx] = inkColor;
                            _strokeMask[idx] = 255;
                            ActiveStrokePixels++;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Bresenham continuous line interpolation to avoid dotted gaps during rapid hand movement.
        /// </summary>
        private void InterpolateLine(int x0, int y0, int x1, int y1)
        {
            int dx = Mathf.Abs(x1 - x0);
            int dy = Mathf.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;

            while (true)
            {
                DrawStamp(x0, y0);
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

        /// <summary>
        /// Clears the slate back to default dark slate color.
        /// </summary>
        public void ClearCanvas()
        {
            for (int i = 0; i < _pixelBuffer.Length; i++)
            {
                _pixelBuffer[i] = slateColor;
                _strokeMask[i] = 0;
            }
            ActiveStrokePixels = 0;
            _lastDrawPixel = null;
            _isPinchingAndHovering = false;

            if (_drawTexture != null)
            {
                _drawTexture.SetPixels32(_pixelBuffer);
                _drawTexture.Apply(false);
            }
            OnCanvasCleared?.Invoke();
        }

        /// <summary>
        /// Extracts and preprocesses current drawing into a normalized 2-Channel [Stroke, SDF] array
        /// satisfying the verified Unity coordinate layout: Dim 2 = X (horizontal), Dim 3 = Y (vertical).
        /// </summary>
        public float[] PreprocessToTwoChannel64x64(out float[] strokeMask64, out float aspectRatio, out float fillDensity)
        {
            strokeMask64 = new float[Resolution * Resolution];
            float[] buffer2Ch = new float[2 * Resolution * Resolution];

            int minX = Resolution, maxX = -1, minY = Resolution, maxY = -1;
            int activeCount = 0;

            for (int y = 0; y < Resolution; y++)
            {
                for (int x = 0; x < Resolution; x++)
                {
                    if (_strokeMask[y * Resolution + x] > 100)
                    {
                        activeCount++;
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }
                }
            }

            if (activeCount == 0)
            {
                aspectRatio = 1.0f;
                fillDensity = 0.0f;
                for (int i = 0; i < Resolution * Resolution; i++)
                {
                    buffer2Ch[Resolution * Resolution + i] = 1.0f; // SDF Channel initialized to max distance
                }
                return buffer2Ch;
            }

            int w = maxX - minX + 1;
            int h = maxY - minY + 1;
            aspectRatio = (float)w / (float)h;
            fillDensity = (float)activeCount / (float)(w * h);

            int targetBox = 52;
            float scale = (float)targetBox / Mathf.Max(w, h);
            int newW = Mathf.Clamp(Mathf.RoundToInt(w * scale), 1, targetBox);
            int newH = Mathf.Clamp(Mathf.RoundToInt(h * scale), 1, targetBox);

            int ox = (Resolution - newW) / 2;
            int oy = (Resolution - newH) / 2;

            for (int dx = 0; dx < newW; dx++)
            {
                for (int dy = 0; dy < newH; dy++)
                {
                    int srcX = minX + Mathf.Clamp(Mathf.RoundToInt(dx * ((float)w / newW)), 0, w - 1);
                    int srcY = minY + Mathf.Clamp(Mathf.RoundToInt(dy * ((float)h / newH)), 0, h - 1);

                    if (_strokeMask[srcY * Resolution + srcX] > 100)
                    {
                        // Verified coordinate layout: Dim 2 is X, Dim 3 is Y
                        strokeMask64[(ox + dx) * Resolution + (oy + dy)] = 1.0f;
                    }
                }
            }

            // Copy Channel 0: Normalized Stroke Mask
            Array.Copy(strokeMask64, 0, buffer2Ch, 0, Resolution * Resolution);

            // Channel 1: 2-Pass Chamfer Euclidean Distance Transform
            float maxVal = 1e6f;
            float[] dist = new float[Resolution * Resolution];
            for (int i = 0; i < dist.Length; i++)
            {
                dist[i] = strokeMask64[i] > 0.5f ? 0.0f : maxVal;
            }

            float sqrt2 = 1.41421356f;
            // Pass 1: Top-Left to Bottom-Right
            for (int x = 0; x < Resolution; x++)
            {
                for (int y = 0; y < Resolution; y++)
                {
                    int idx = x * Resolution + y;
                    if (dist[idx] > 0f)
                    {
                        float best = dist[idx];
                        if (x > 0) best = Mathf.Min(best, dist[(x - 1) * Resolution + y] + 1.0f);
                        if (y > 0) best = Mathf.Min(best, dist[x * Resolution + (y - 1)] + 1.0f);
                        if (x > 0 && y > 0) best = Mathf.Min(best, dist[(x - 1) * Resolution + (y - 1)] + sqrt2);
                        if (x < Resolution - 1 && y > 0) best = Mathf.Min(best, dist[(x + 1) * Resolution + (y - 1)] + sqrt2);
                        dist[idx] = best;
                    }
                }
            }

            // Pass 2: Bottom-Right to Top-Left
            for (int x = Resolution - 1; x >= 0; x--)
            {
                for (int y = Resolution - 1; y >= 0; y--)
                {
                    int idx = x * Resolution + y;
                    if (dist[idx] > 0f)
                    {
                        float best = dist[idx];
                        if (x < Resolution - 1) best = Mathf.Min(best, dist[(x + 1) * Resolution + y] + 1.0f);
                        if (y < Resolution - 1) best = Mathf.Min(best, dist[x * Resolution + (y + 1)] + 1.0f);
                        if (x < Resolution - 1 && y < Resolution - 1) best = Mathf.Min(best, dist[(x + 1) * Resolution + (y + 1)] + sqrt2);
                        if (x > 0 && y < Resolution - 1) best = Mathf.Min(best, dist[(x - 1) * Resolution + (y + 1)] + sqrt2);
                        dist[idx] = best;
                    }
                }
            }

            // Pack Channel 1: Truncated SDF (normalized by dMax = 16.0)
            float dMax = 16.0f;
            int offsetCh1 = Resolution * Resolution;
            for (int i = 0; i < Resolution * Resolution; i++)
            {
                buffer2Ch[offsetCh1 + i] = Mathf.Clamp01(dist[i] / dMax);
            }

            return buffer2Ch;
        }
    }
}
