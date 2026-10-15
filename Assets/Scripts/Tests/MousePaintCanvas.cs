using System.Collections;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif
using VRDraw2Voxel.Inference;
using VRDraw2Voxel.VoxelMesh;

namespace VRDraw2Voxel.Tests
{
    /// <summary>
    /// Interactive drawing canvas & showcase for the Draw2Voxel deep learning model.
    /// Supports multi-category reconstruction: Weapons (melee & firearms), Toys (creatures/figures),
    /// and Electronics (gadgets & retro tech) with continuous 360° PS2 trophy achievement display,
    /// real-time stroke feasibility feedback, and a player capability disclaimer guide.
    /// </summary>
    public enum GamePlayMode
    {
        Sandbox50Class = 0,
        Studio12Class = 1
    }

    public class MousePaintCanvas : MonoBehaviour
    {
        [Header("Engine References")]
        [SerializeField] private GamePlayMode activeMode = GamePlayMode.Sandbox50Class;
        [SerializeField] private UniversalSandboxEngine sandboxEngine;
        [SerializeField] private SentisMultiTaskEngine multiTaskEngine;
        [SerializeField] private ColoredVoxelMesher coloredVoxelMesher;
        [SerializeField] private SentisVoxelEngine voxelEngine;
        [SerializeField] private GreedyVoxelMesher voxelMesher;
        [SerializeField] private PS2ShowcaseRotator showcaseRotator;

        [Header("Canvas Settings")]
        [SerializeField] private int canvasSize = 256; // Display resolution (downsampled to 64x64)
        [SerializeField] private int brushRadius = 4;
        [SerializeField] private bool autoGenerateOnRelease = true;

        [Header("Showcase Pedestal / Camera")]
        [SerializeField] private bool autoPositionCamera = true;

        [Header("Preview Settings")]
        [Tooltip("Render an instant extruded wireframe preview while inference runs.")]
        [SerializeField] private bool showHolographicWireframePreview = false;

        private Texture2D _drawTexture;
        private Color32[] _texturePixels;
        private Vector2? _lastMousePixel;
        private bool _isDrawing = false;
        private float _lastInferenceTimeMs = 0f;
        private int _lastSolidVoxelCount = 0;
        private string _statusMessage = "Draw with mouse or pick a preset!";
        private string _lastGeneratedItemName = "Airplane";
        private string _lastCategoryName = "VEHICLES";

        // Multi-Task Tri-View State
        private ReconstructedCategory _detectedCategory = ReconstructedCategory.Unknown;
        private float _categoryConfidence = 0f;
        private PerspectiveView _detectedView = PerspectiveView.Front_XY;
        private float _viewConfidence = 0f;
        private bool _isOutOfDistribution = false;

        // Sandbox Mode State
        private SandboxInferenceResult _lastSandboxResult;
        private int _selectedSandboxTab = 0; // 0: Vehicles, 1: Food, 2: Animals, 3: Tools, 4: Weapons

        // Multi-Category Tabs & Disclaimer Guide State
        private int _selectedCategoryTab = 0; // 0: Weapons, 1: Toys, 2: Electronics
        private bool _showDisclaimerModal = false;

        public const int ModelResolution = 64;
        private Rect _canvasRect = new Rect(20, 68, 256, 256);

        private void Awake()
        {
            InitializeTexture();

            if (sandboxEngine == null)
            {
                sandboxEngine = FindFirstObjectByType<UniversalSandboxEngine>();
                if (sandboxEngine == null)
                {
                    sandboxEngine = gameObject.AddComponent<UniversalSandboxEngine>();
                    Debug.Log("[MousePaintCanvas] Auto-attached UniversalSandboxEngine to GameObject.");
                }
            }

            if (multiTaskEngine == null)
            {
                multiTaskEngine = FindFirstObjectByType<SentisMultiTaskEngine>();
                if (multiTaskEngine == null)
                {
                    multiTaskEngine = gameObject.AddComponent<SentisMultiTaskEngine>();
                    Debug.Log("[MousePaintCanvas] Auto-attached SentisMultiTaskEngine to GameObject.");
                }
            }

            if (coloredVoxelMesher == null)
            {
                coloredVoxelMesher = FindFirstObjectByType<ColoredVoxelMesher>();
                if (coloredVoxelMesher == null)
                {
                    coloredVoxelMesher = gameObject.AddComponent<ColoredVoxelMesher>();
                    Debug.Log("[MousePaintCanvas] Auto-attached ColoredVoxelMesher to GameObject.");
                }
            }

            if (voxelEngine == null) voxelEngine = FindFirstObjectByType<SentisVoxelEngine>();
            if (voxelMesher == null) voxelMesher = FindFirstObjectByType<GreedyVoxelMesher>();
            if (showcaseRotator == null) showcaseRotator = FindFirstObjectByType<PS2ShowcaseRotator>();

            if (autoPositionCamera)
            {
                SetupCameraAndLighting();
            }
        }

        private IEnumerator Start()
        {
            // Give one frame for all inference engines to awaken cleanly
            yield return null;
            if (activeMode == GamePlayMode.Sandbox50Class)
            {
                LoadQuickDrawPreset("Airplane");
            }
            else
            {
                LoadPreset("Sword");
            }
        }

        private void InitializeTexture()
        {
            _drawTexture = new Texture2D(canvasSize, canvasSize, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };

            _texturePixels = new Color32[canvasSize * canvasSize];
            ClearCanvas(false);
        }

        private void Update()
        {
            HandleMouseInput();
        }

        private void HandleMouseInput()
        {
            Vector2 mousePos = Vector2.zero;
            bool leftDown = false;
            bool leftHeld = false;
            bool leftUp = false;
            bool rightDown = false;

#if ENABLE_INPUT_SYSTEM
            var mouse = Mouse.current;
            if (mouse != null)
            {
                mousePos = mouse.position.ReadValue();
                leftDown = mouse.leftButton.wasPressedThisFrame;
                leftHeld = mouse.leftButton.isPressed;
                leftUp = mouse.leftButton.wasReleasedThisFrame;
                rightDown = mouse.rightButton.wasPressedThisFrame;
            }
#else
            mousePos = Input.mousePosition;
            leftDown = Input.GetMouseButtonDown(0);
            leftHeld = Input.GetMouseButton(0);
            leftUp = Input.GetMouseButtonUp(0);
            rightDown = Input.GetMouseButtonDown(1);
#endif

            // Invert Y for GUI coordinates
            mousePos.y = Screen.height - mousePos.y;

            if (_canvasRect.Contains(mousePos))
            {
                if (leftDown)
                {
                    _isDrawing = true;
                    Vector2 pixel = GetPixelCoord(mousePos);
                    DrawBrushStamp((int)pixel.x, (int)pixel.y);
                    _lastMousePixel = pixel;
                    _drawTexture.Apply();
                }
                else if (leftHeld && _isDrawing)
                {
                    Vector2 pixel = GetPixelCoord(mousePos);
                    if (_lastMousePixel.HasValue)
                    {
                        DrawLine((int)_lastMousePixel.Value.x, (int)_lastMousePixel.Value.y, (int)pixel.x, (int)pixel.y);
                    }
                    _lastMousePixel = pixel;
                    _drawTexture.Apply();
                }
            }

            if (leftUp && _isDrawing)
            {
                _isDrawing = false;
                _lastMousePixel = null;
                if (autoGenerateOnRelease)
                {
                    _lastGeneratedItemName = "Custom Sketch Creation";
                    _lastCategoryName = _selectedCategoryTab == 0 ? "WEAPONS" : (_selectedCategoryTab == 1 ? "TOYS" : "ELECTRONICS");
                    Generate3DVoxels();
                }
            }

            // Right click clears canvas
            if (rightDown && _canvasRect.Contains(mousePos))
            {
                ClearCanvas();
            }
        }

        private Vector2 GetPixelCoord(Vector2 mousePos)
        {
            float normX = (mousePos.x - _canvasRect.x) / _canvasRect.width;
            float normY = 1.0f - ((mousePos.y - _canvasRect.y) / _canvasRect.height);
            int px = Mathf.Clamp((int)(normX * canvasSize), 0, canvasSize - 1);
            int py = Mathf.Clamp((int)(normY * canvasSize), 0, canvasSize - 1);
            return new Vector2(px, py);
        }

        private void DrawBrushStamp(int cx, int cy)
        {
            int rSq = brushRadius * brushRadius;
            for (int dy = -brushRadius; dy <= brushRadius; dy++)
            {
                for (int dx = -brushRadius; dx <= brushRadius; dx++)
                {
                    if (dx * dx + dy * dy <= rSq)
                    {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < canvasSize && py >= 0 && py < canvasSize)
                        {
                            _texturePixels[py * canvasSize + px] = new Color32(255, 255, 255, 255);
                        }
                    }
                }
            }
            _drawTexture.SetPixels32(_texturePixels);
        }

        private void DrawLine(int x0, int y0, int x1, int y1)
        {
            int dx = Mathf.Abs(x1 - x0);
            int dy = Mathf.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;

            while (true)
            {
                DrawBrushStamp(x0, y0);
                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 > -dy) { err -= dy; x0 += sx; }
                if (e2 < dx) { err += dx; y0 += sy; }
            }
        }

        // ==============================================================================
        // 2D DRAWING PRIMITIVES FOR PRESETS
        // ==============================================================================

        private void FillRect(int x0, int y0, int x1, int y1)
        {
            int minX = Mathf.Clamp(Mathf.Min(x0, x1), 0, canvasSize - 1);
            int maxX = Mathf.Clamp(Mathf.Max(x0, x1), 0, canvasSize - 1);
            int minY = Mathf.Clamp(Mathf.Min(y0, y1), 0, canvasSize - 1);
            int maxY = Mathf.Clamp(Mathf.Max(y0, y1), 0, canvasSize - 1);

            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    _texturePixels[y * canvasSize + x] = new Color32(255, 255, 255, 255);
                }
            }
        }

        private void FillCircle(int cx, int cy, int radius)
        {
            int rSq = radius * radius;
            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {
                    if (dx * dx + dy * dy <= rSq)
                    {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < canvasSize && py >= 0 && py < canvasSize)
                        {
                            _texturePixels[py * canvasSize + px] = new Color32(255, 255, 255, 255);
                        }
                    }
                }
            }
        }

        private void FillEllipse(int cx, int cy, int rx, int ry)
        {
            for (int dy = -ry; dy <= ry; dy++)
            {
                for (int dx = -rx; dx <= rx; dx++)
                {
                    if (((float)(dx * dx) / (rx * rx)) + ((float)(dy * dy) / (ry * ry)) <= 1.0f)
                    {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < canvasSize && py >= 0 && py < canvasSize)
                        {
                            _texturePixels[py * canvasSize + px] = new Color32(255, 255, 255, 255);
                        }
                    }
                }
            }
        }

        public void ClearCanvas(bool apply = true)
        {
            Color32 black = new Color32(18, 18, 22, 255);
            for (int i = 0; i < _texturePixels.Length; i++)
            {
                _texturePixels[i] = black;
            }
            _drawTexture.SetPixels32(_texturePixels);
            if (apply) _drawTexture.Apply();
            _statusMessage = "Canvas cleared. Sketch something or pick a preset!";
        }

        /// <summary>
        /// Loads an authentic human Quick, Draw! preset sketch and runs inference.
        /// </summary>
        public void LoadQuickDrawPreset(string presetName)
        {
            if (sandboxEngine == null) return;
            byte[] mask = sandboxEngine.GetPresetMask(presetName);
            if (mask == null || mask.Length != ModelResolution * ModelResolution)
            {
                Debug.LogWarning($"[MousePaintCanvas] Preset '{presetName}' not found in catalog, fallback to clearing.");
                ClearCanvas(true);
                return;
            }

            ClearCanvas(false);
            int scale = canvasSize / ModelResolution;

            for (int x = 0; x < ModelResolution; x++)
            {
                for (int y = 0; y < ModelResolution; y++)
                {
                    byte b = mask[x * ModelResolution + y];
                    if (b > 80)
                    {
                        FillRect(x * scale, y * scale, (x + 1) * scale, (y + 1) * scale);
                    }
                }
            }

            _drawTexture.SetPixels32(_texturePixels);
            _drawTexture.Apply();
            Generate3DVoxels();
        }

        /// <summary>
        /// Transforms canvas into normalized 2-channel representation [Channel 0: Stroke, Channel 1: SDF],
        /// displays instant holographic wireframe extrusion preview, and dispatches to Sentis.
        /// </summary>
        public void Generate3DVoxels()
        {
            float[] strokeMask;
            float aspectRatio, fillDensity;
            float[] inputBuffer2Ch = PreprocessToTwoChannel64x64(out strokeMask, out aspectRatio, out fillDensity);

            // Optional instant holographic wireframe preview (<0.5ms Burst mesh generation)
            if (showHolographicWireframePreview)
            {
                ShowHolographicWireframePreview(strokeMask);
            }

            if (activeMode == GamePlayMode.Sandbox50Class && sandboxEngine != null)
            {
                _statusMessage = "Inferring Universal Sandbox Prop (50+ Classes)...";
                sandboxEngine.PredictSandboxAsync(inputBuffer2Ch, (result) =>
                {
                    _lastSandboxResult = result;
                    _lastInferenceTimeMs = result.latencyMs;
                    _lastCategoryName = result.family.ToUpper();
                    _lastGeneratedItemName = result.label;

                    int count = 0;
                    if (result.occupancyGrid != null)
                    {
                        for (int i = 0; i < result.occupancyGrid.Length; i++)
                        {
                            if (result.occupancyGrid[i] > 0) count++;
                        }
                    }
                    _lastSolidVoxelCount = count;

                    if (result.isNoise)
                    {
                        _statusMessage = $"<color=#ff6666>[NOISE REJECTED ({result.confidence * 100f:F0}%)]</color> Input is unstructured scribble!";
                    }
                    else
                    {
                        _statusMessage = $"Sandbox: <b>{result.label}</b> ({result.confidence * 100f:F0}%) | Voxels: {count} | {result.latencyMs:F1}ms";
                        if (coloredVoxelMesher != null)
                        {
                            coloredVoxelMesher.GenerateColoredMesh(result.occupancyGrid, result.voxelColors);
                        }
                    }
                });
                return;
            }

            if (multiTaskEngine != null)
            {
                _statusMessage = "Inferring Tri-View Multi-Task 3D Voxels...";
                multiTaskEngine.PredictMultiTaskAsync(inputBuffer2Ch, (result) =>
                {
                    _lastInferenceTimeMs = result.latencyMs;
                    _detectedCategory = result.category;
                    _categoryConfidence = result.categoryConfidence;
                    _detectedView = result.detectedView;
                    _viewConfidence = result.viewConfidence;
                    _isOutOfDistribution = result.isOutOfDistribution;

                    int count = 0;
                    for (int i = 0; i < result.occupancyGrid.Length; i++)
                    {
                        if (result.occupancyGrid[i] > 0) count++;
                    }
                    _lastSolidVoxelCount = count;

                    if (_isOutOfDistribution)
                    {
                        _statusMessage = $"<color=#ff6666>[OOD REJECTED ({_categoryConfidence * 100f:F0}%)]</color> Input is unstructured noise/scribble!";
                    }
                    else if (result.isHybridSketch)
                    {
                        _lastCategoryName = $"{_detectedCategory.ToString().ToUpper()} (HYBRID)";
                        _lastGeneratedItemName = $"Hybrid {_detectedCategory}-{result.secondaryCategory}";
                        _statusMessage = $"<color=#66ddff>[HYBRID {result.family.ToString().ToUpper()}]</color> {_detectedCategory} ({result.categoryConfidence * 100f:F0}%) + {result.secondaryCategory} ({result.secondaryConfidence * 100f:F0}%) | View: {_detectedView} | Voxels: {count} | {result.latencyMs:F1}ms";

                        if (coloredVoxelMesher != null)
                        {
                            coloredVoxelMesher.GenerateFromInferenceResult(result);
                        }
                    }
                    else
                    {
                        _lastCategoryName = _detectedCategory.ToString().ToUpper();
                        _lastGeneratedItemName = GetCanonicalItemName(_detectedCategory);
                        _statusMessage = $"Reconstructed {_detectedCategory} ({_categoryConfidence * 100f:F0}%) | View: {_detectedView} ({_viewConfidence * 100f:F0}%) | Voxels: {count} | {result.latencyMs:F1}ms";

                        // Swap holographic preview to final colored voxel mesh
                        if (coloredVoxelMesher != null)
                        {
                            coloredVoxelMesher.GenerateFromInferenceResult(result);
                        }
                    }

                    if (coloredVoxelMesher == null && voxelMesher != null)
                    {
                        voxelMesher.GenerateMeshFromOccupancy(result.occupancyGrid);
                    }
                });
                return;
            }

            if (voxelEngine == null)
            {
                _statusMessage = "Error: SentisVoxelEngine or SentisMultiTaskEngine not found in scene!";
                return;
            }

            _statusMessage = "Inferring 3D Voxel Mesh...";
            float startTime = Time.realtimeSinceStartup;
            voxelEngine.PredictVoxelsAsync(strokeMask, (voxels) =>
            {
                _lastInferenceTimeMs = (Time.realtimeSinceStartup - startTime) * 1000f;
                int count = 0;
                for (int i = 0; i < voxels.Length; i++)
                {
                    if (voxels[i] > 0) count++;
                }
                _lastSolidVoxelCount = count;
                _statusMessage = $"Generated! Solid Voxels: {count} | Latency: {_lastInferenceTimeMs:F1} ms";

                if (voxelMesher != null)
                {
                    voxelMesher.GenerateMeshFromOccupancy(voxels);
                }
            });
        }

        /// <summary>
        /// Transforms current drawing into a 2-channel normalized representation:
        /// - Channel 0: Tight AABB crop rescaled to fit within 52x52 box preserving aspect ratio, centered in 64x64.
        /// - Channel 1: Normalized Truncated Euclidean Distance Field (0.0 on stroke, 1.0 at distance >= 16px).
        /// </summary>
        private float[] PreprocessToTwoChannel64x64(out float[] strokeMask64, out float aspectRatio, out float fillDensity)
        {
            strokeMask64 = new float[ModelResolution * ModelResolution];
            float[] buffer2Ch = new float[2 * ModelResolution * ModelResolution];

            int minX = canvasSize, maxX = -1, minY = canvasSize, maxY = -1;
            int activeCount = 0;

            for (int y = 0; y < canvasSize; y++)
            {
                for (int x = 0; x < canvasSize; x++)
                {
                    if (_texturePixels[y * canvasSize + x].r > 100)
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
                for (int i = 0; i < ModelResolution * ModelResolution; i++)
                {
                    buffer2Ch[ModelResolution * ModelResolution + i] = 1.0f;
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

            int ox = (ModelResolution - newW) / 2;
            int oy = (ModelResolution - newH) / 2;

            for (int dx = 0; dx < newW; dx++)
            {
                for (int dy = 0; dy < newH; dy++)
                {
                    int srcX = minX + Mathf.Clamp(Mathf.RoundToInt(dx * ((float)w / newW)), 0, w - 1);
                    int srcY = minY + Mathf.Clamp(Mathf.RoundToInt(dy * ((float)h / newH)), 0, h - 1);

                    Color32 c = _texturePixels[srcY * canvasSize + srcX];
                    if (c.r > 100)
                    {
                        strokeMask64[(ox + dx) * ModelResolution + (oy + dy)] = 1.0f;
                    }
                }
            }

            // Pack Channel 0: Normalized Stroke Mask
            System.Array.Copy(strokeMask64, 0, buffer2Ch, 0, ModelResolution * ModelResolution);

            // Compute Channel 1: 2-Pass Chamfer Euclidean Distance Transform
            float maxVal = 1e6f;
            float[] dist = new float[ModelResolution * ModelResolution];
            for (int i = 0; i < dist.Length; i++)
            {
                dist[i] = strokeMask64[i] > 0.5f ? 0.0f : maxVal;
            }

            float sqrt2 = 1.41421356f;
            for (int x = 0; x < ModelResolution; x++)
            {
                for (int y = 0; y < ModelResolution; y++)
                {
                    int idx = x * ModelResolution + y;
                    if (dist[idx] > 0f)
                    {
                        float best = dist[idx];
                        if (x > 0) best = Mathf.Min(best, dist[(x - 1) * ModelResolution + y] + 1.0f);
                        if (y > 0) best = Mathf.Min(best, dist[x * ModelResolution + (y - 1)] + 1.0f);
                        if (x > 0 && y > 0) best = Mathf.Min(best, dist[(x - 1) * ModelResolution + (y - 1)] + sqrt2);
                        if (x < ModelResolution - 1 && y > 0) best = Mathf.Min(best, dist[(x + 1) * ModelResolution + (y - 1)] + sqrt2);
                        dist[idx] = best;
                    }
                }
            }

            for (int x = ModelResolution - 1; x >= 0; x--)
            {
                for (int y = ModelResolution - 1; y >= 0; y--)
                {
                    int idx = x * ModelResolution + y;
                    float best = dist[idx];
                    if (x < ModelResolution - 1) best = Mathf.Min(best, dist[(x + 1) * ModelResolution + y] + 1.0f);
                    if (y < ModelResolution - 1) best = Mathf.Min(best, dist[x * ModelResolution + (y + 1)] + 1.0f);
                    if (x < ModelResolution - 1 && y < ModelResolution - 1) best = Mathf.Min(best, dist[(x + 1) * ModelResolution + (y + 1)] + sqrt2);
                    if (x > 0 && y < ModelResolution - 1) best = Mathf.Min(best, dist[(x - 1) * ModelResolution + (y + 1)] + sqrt2);
                    dist[idx] = best;
                }
            }

            // Normalize truncated distance field to [0.0, 1.0] (d_max = 16.0f)
            float dMax = 16.0f;
            int ch1Offset = ModelResolution * ModelResolution;
            for (int i = 0; i < dist.Length; i++)
            {
                buffer2Ch[ch1Offset + i] = Mathf.Clamp01(dist[i] / dMax);
            }

            return buffer2Ch;
        }

        /// <summary>
        /// Instantly renders an extruded holographic wireframe preview while neural inference runs.
        /// </summary>
        private void ShowHolographicWireframePreview(float[] strokeMask64)
        {
            if (coloredVoxelMesher == null) return;

            int totalVoxels = ModelResolution * ModelResolution * ModelResolution;
            byte[] previewOcc = new byte[totalVoxels];
            Color32[] previewColors = new Color32[totalVoxels];
            Color32 holoCyan = new Color32(0, 220, 255, 255);

            int zStart = 28;
            int zEnd = 36;

            for (int x = 0; x < ModelResolution; x++)
            {
                for (int y = 0; y < ModelResolution; y++)
                {
                    if (strokeMask64[x * ModelResolution + y] > 0.5f)
                    {
                        for (int z = zStart; z < zEnd; z++)
                        {
                            int idx = x * (ModelResolution * ModelResolution) + y * ModelResolution + z;
                            previewOcc[idx] = 1;
                            previewColors[idx] = holoCyan;
                        }
                    }
                }
            }

            coloredVoxelMesher.GenerateColoredMesh(previewOcc, previewColors);
        }

        // ==============================================================================
        // REAL-TIME FEASIBILITY EVALUATOR
        // ==============================================================================

        private void AnalyzeFeasibility(out string level, out string tip, out Color color)
        {
            int solidPixels = 0;
            int minX = canvasSize, maxX = 0, minY = canvasSize, maxY = 0;

            for (int y = 0; y < canvasSize; y++)
            {
                for (int x = 0; x < canvasSize; x++)
                {
                    if (_texturePixels[y * canvasSize + x].r > 100)
                    {
                        solidPixels++;
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }
                }
            }

            if (solidPixels < 50)
            {
                level = "EMPTY / TOO SMALL";
                tip = "Canvas is nearly empty. Draw a closed silhouette or try a preset!";
                color = new Color(0.95f, 0.4f, 0.4f);
            }
            else
            {
                int boundW = maxX - minX;
                int boundH = maxY - minY;

                if (boundW < 24 && boundH < 24)
                {
                    level = "LOW FEASIBILITY";
                    tip = "Drawing is too tiny. Scale up for full 3D depth and thickness.";
                    color = new Color(0.95f, 0.65f, 0.2f);
                }
                else if (solidPixels > 160)
                {
                    level = "EXCELLENT FEASIBILITY";
                    tip = "Solid silhouette detected! Full 3D volume reconstruction guaranteed.";
                    color = new Color(0.3f, 1.0f, 0.45f);
                }
                else
                {
                    level = "MODERATE (THIN STROKE)";
                    tip = "Outline detected. Draw slightly thicker strokes (3-4px) for best volume.";
                    color = new Color(0.95f, 0.9f, 0.25f);
                }
            }
        }

        // ==============================================================================
        // MULTI-CATEGORY PRESETS
        // ==============================================================================

        public void LoadPreset(string type)
        {
            ClearCanvas(false);
            int scale = canvasSize / ModelResolution;

            // --- CATEGORY 1: WEAPONS ---
            if (type == "Sword")
            {
                _lastGeneratedItemName = "Excalibur Broadsword";
                _lastCategoryName = "WEAPONS";
                FillRect(30 * scale, 20 * scale, 33 * scale, 58 * scale); // Blade
                FillRect(20 * scale, 18 * scale, 43 * scale, 21 * scale); // Crossguard
                FillRect(31 * scale, 6 * scale, 32 * scale, 18 * scale);  // Handle
                FillCircle(32 * scale, 6 * scale, 3 * scale);             // Pommel
            }
            else if (type == "Pistol")
            {
                _lastGeneratedItemName = "Tactical Magnum Handgun";
                _lastCategoryName = "WEAPONS";
                FillRect(34 * scale, 8 * scale, 42 * scale, 26 * scale);  // Grip
                FillRect(6 * scale, 26 * scale, 44 * scale, 36 * scale);   // Slide
                FillRect(4 * scale, 28 * scale, 8 * scale, 34 * scale);    // Muzzle tip
                FillRect(24 * scale, 20 * scale, 30 * scale, 26 * scale);  // Trigger guard notch
                FillRect(26 * scale, 16 * scale, 34 * scale, 24 * scale);  // Magazine/Receiver
            }
            else if (type == "Blaster")
            {
                _lastGeneratedItemName = "Hyperion Sci-Fi Blaster";
                _lastCategoryName = "WEAPONS";
                FillRect(36 * scale, 8 * scale, 44 * scale, 26 * scale);  // Grip
                FillRect(20 * scale, 26 * scale, 44 * scale, 36 * scale); // Body
                FillRect(4 * scale, 28 * scale, 20 * scale, 34 * scale);  // Barrel
                FillCircle(10 * scale, 31 * scale, 4 * scale);            // Ring 1
                FillCircle(16 * scale, 31 * scale, 4 * scale);            // Ring 2
                FillRect(24 * scale, 16 * scale, 32 * scale, 24 * scale); // Power cell
            }
            else if (type == "Axe")
            {
                _lastGeneratedItemName = "Viking Battleaxe";
                _lastCategoryName = "WEAPONS";
                FillRect(29 * scale, 4 * scale, 33 * scale, 58 * scale);  // Sturdy wooden shaft (width 5)
                FillRect(31 * scale, 38 * scale, 50 * scale, 52 * scale); // Main axe head
                FillRect(48 * scale, 32 * scale, 53 * scale, 58 * scale); // Flared curved cutting blade
                FillRect(24 * scale, 43 * scale, 31 * scale, 47 * scale); // Rear counter-spike
            }

            // --- CATEGORY 2: TOYS ---
            else if (type == "Teddy Bear")
            {
                _lastGeneratedItemName = "Plush Teddy Bear Trophy";
                _lastCategoryName = "TOYS";
                FillEllipse(32 * scale, 22 * scale, 13 * scale, 12 * scale); // Torso
                FillCircle(32 * scale, 42 * scale, 9 * scale);               // Head
                FillCircle(32 * scale, 39 * scale, 4 * scale);               // Snout
                FillCircle(24 * scale, 48 * scale, 4 * scale);               // Left ear
                FillCircle(40 * scale, 48 * scale, 4 * scale);               // Right ear
                FillEllipse(18 * scale, 24 * scale, 5 * scale, 8 * scale);   // Left arm
                FillEllipse(46 * scale, 24 * scale, 5 * scale, 8 * scale);   // Right arm
                FillEllipse(23 * scale, 11 * scale, 5 * scale, 6 * scale);   // Left foot
                FillEllipse(41 * scale, 11 * scale, 5 * scale, 6 * scale);   // Right foot
            }
            else if (type == "Rubber Duck")
            {
                _lastGeneratedItemName = "Classic Rubber Ducky";
                _lastCategoryName = "TOYS";
                FillEllipse(28 * scale, 24 * scale, 15 * scale, 10 * scale); // Body
                FillCircle(42 * scale, 36 * scale, 8 * scale);               // Head
                FillRect(48 * scale, 34 * scale, 57 * scale, 38 * scale);    // Bill
                FillRect(13 * scale, 28 * scale, 18 * scale, 34 * scale);    // Tail
            }
            else if (type == "Toy Robot")
            {
                _lastGeneratedItemName = "Retro Wind-Up Robot";
                _lastCategoryName = "TOYS";
                FillRect(24 * scale, 40 * scale, 40 * scale, 52 * scale); // Head
                FillRect(31 * scale, 52 * scale, 32 * scale, 58 * scale); // Antenna
                FillCircle(32 * scale, 58 * scale, 3 * scale);            // Antenna bulb
                FillRect(22 * scale, 20 * scale, 42 * scale, 38 * scale); // Torso
                FillCircle(32 * scale, 29 * scale, 4 * scale);            // Chest dial
                FillRect(14 * scale, 22 * scale, 22 * scale, 26 * scale); // Left arm
                FillRect(42 * scale, 22 * scale, 50 * scale, 26 * scale); // Right arm
                FillRect(22 * scale, 8 * scale, 30 * scale, 20 * scale);  // Left tread
                FillRect(34 * scale, 8 * scale, 42 * scale, 20 * scale);  // Right tread
            }
            else if (type == "Toy Bunny")
            {
                _lastGeneratedItemName = "Fluffy Bunny Figurine";
                _lastCategoryName = "TOYS";
                FillEllipse(32 * scale, 21 * scale, 12 * scale, 11 * scale); // Body
                FillCircle(32 * scale, 39 * scale, 8 * scale);               // Head
                FillEllipse(27 * scale, 51 * scale, 3 * scale, 9 * scale);   // Left ear
                FillEllipse(37 * scale, 51 * scale, 3 * scale, 9 * scale);   // Right ear
                FillCircle(18 * scale, 20 * scale, 4 * scale);               // Tail
                FillEllipse(26 * scale, 10 * scale, 4 * scale, 5 * scale);   // Left foot
                FillEllipse(38 * scale, 10 * scale, 4 * scale, 5 * scale);   // Right foot
            }

            // --- CATEGORY 3: ELECTRONICS ---
            else if (type == "Gamepad")
            {
                _lastGeneratedItemName = "DualGrip Pro Game Controller";
                _lastCategoryName = "ELECTRONICS";
                FillRect(20 * scale, 22 * scale, 44 * scale, 36 * scale);    // Bridge
                FillEllipse(16 * scale, 20 * scale, 6 * scale, 12 * scale);  // Left grip
                FillEllipse(48 * scale, 20 * scale, 6 * scale, 12 * scale);  // Right grip
                FillRect(22 * scale, 25 * scale, 28 * scale, 31 * scale);    // D-pad
                FillCircle(42 * scale, 30 * scale, 3 * scale);               // Buttons
                FillCircle(26 * scale, 18 * scale, 4 * scale);               // Left stick
                FillCircle(38 * scale, 18 * scale, 4 * scale);               // Right stick
            }
            else if (type == "Retro TV")
            {
                _lastGeneratedItemName = "Vintage Cathode CRT TV";
                _lastCategoryName = "ELECTRONICS";
                FillRect(16 * scale, 16 * scale, 48 * scale, 42 * scale); // Cabinet
                FillRect(19 * scale, 19 * scale, 41 * scale, 39 * scale); // Screen
                FillCircle(44 * scale, 34 * scale, 3 * scale);            // Top knob
                FillCircle(44 * scale, 24 * scale, 3 * scale);            // Bottom knob
                FillRect(20 * scale, 42 * scale, 23 * scale, 58 * scale); // Left antenna
                FillRect(41 * scale, 42 * scale, 44 * scale, 58 * scale); // Right antenna
                FillRect(20 * scale, 10 * scale, 24 * scale, 16 * scale); // Left foot
                FillRect(40 * scale, 10 * scale, 44 * scale, 16 * scale); // Right foot
            }
            else if (type == "Game Boy")
            {
                _lastGeneratedItemName = "Pocket 8-Bit Handheld Console";
                _lastCategoryName = "ELECTRONICS";
                FillRect(20 * scale, 12 * scale, 44 * scale, 52 * scale); // Body
                FillRect(23 * scale, 33 * scale, 41 * scale, 48 * scale); // Screen
                FillRect(24 * scale, 20 * scale, 30 * scale, 26 * scale); // D-pad
                FillCircle(36 * scale, 19 * scale, 3 * scale);            // A button
                FillCircle(40 * scale, 23 * scale, 3 * scale);            // B button
            }

            _drawTexture.SetPixels32(_texturePixels);
            _drawTexture.Apply();
            Generate3DVoxels();
        }

        private void SetupCameraAndLighting()
        {
            Camera cam = Camera.main;
            if (cam != null)
            {
                cam.transform.position = new Vector3(0, 0.4f, -1.8f);
                cam.transform.rotation = Quaternion.Euler(10, 0, 0);
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = new Color(0.08f, 0.09f, 0.12f);
            }
        }

        private void OnGUI()
        {
            // Dark Retro HUD Window
            GUI.color = Color.white;
            GUI.Box(new Rect(10, 10, 305, 715), "");

            // 1. Header & Engine Mode Switcher
            GUI.Label(new Rect(20, 14, 280, 20), "<b>★ DRAW-2-VOXEL AI WORKSHOP ★</b>");
            string[] modeNames = new string[] { "★ Sandbox (50+ Cls)", "Studio (12-Cls)" };
            int currentModeIdx = (activeMode == GamePlayMode.Sandbox50Class) ? 0 : 1;
            int newModeIdx = GUI.Toolbar(new Rect(20, 36, 285, 24), currentModeIdx, modeNames);
            if (newModeIdx != currentModeIdx)
            {
                activeMode = (newModeIdx == 0) ? GamePlayMode.Sandbox50Class : GamePlayMode.Studio12Class;
                if (activeMode == GamePlayMode.Sandbox50Class) LoadQuickDrawPreset("Airplane");
                else LoadPreset("Sword");
            }

            // 2. Canvas Border & Texture
            GUI.Label(new Rect(20, 64, 280, 18), "<color=#aaaaaa><size=11>Draw 2D sketch with mouse:</size></color>");
            GUI.Box(new Rect(_canvasRect.x - 2, _canvasRect.y - 2, _canvasRect.width + 4, _canvasRect.height + 4), "");
            GUI.DrawTexture(_canvasRect, _drawTexture);

            // 3. Action Controls
            float btnY = _canvasRect.yMax + 8;
            if (GUI.Button(new Rect(20, btnY, 138, 28), "<b>Reconstruct 3D</b>"))
            {
                Generate3DVoxels();
            }

            if (GUI.Button(new Rect(166, btnY, 138, 28), "Clear Paint"))
            {
                ClearCanvas();
            }

            // 4. Live Feasibility Meter
            btnY += 32;
            AnalyzeFeasibility(out string feasLevel, out string feasTip, out Color feasColor);
            GUI.color = feasColor;
            GUI.Label(new Rect(20, btnY, 280, 18), $"<b>● FEASIBILITY: {feasLevel}</b>");
            GUI.color = Color.white;
            GUI.Label(new Rect(20, btnY + 16, 280, 26), $"<size=10><color=#cccccc>{feasTip}</color></size>");

            // 5. Presets
            btnY += 42;
            if (activeMode == GamePlayMode.Sandbox50Class)
            {
                GUI.Label(new Rect(20, btnY, 280, 18), "<b>Quick, Draw! Sandbox Presets:</b>");
                btnY += 18;

                string[] sbTabs = new string[] { "Vehicles", "Food", "Animals", "Tools", "Weapons" };
                _selectedSandboxTab = GUI.Toolbar(new Rect(20, btnY, 285, 22), _selectedSandboxTab, sbTabs);

                btnY += 26;
                if (_selectedSandboxTab == 0) // Vehicles
                {
                    if (GUI.Button(new Rect(20, btnY, 68, 24), "Plane")) LoadQuickDrawPreset("Airplane");
                    if (GUI.Button(new Rect(92, btnY, 68, 24), "Car")) LoadQuickDrawPreset("Sports Car");
                    if (GUI.Button(new Rect(164, btnY, 68, 24), "Bike")) LoadQuickDrawPreset("Bicycle");
                    if (GUI.Button(new Rect(236, btnY, 68, 24), "Boat")) LoadQuickDrawPreset("Sailboat");
                }
                else if (_selectedSandboxTab == 1) // Food
                {
                    if (GUI.Button(new Rect(20, btnY, 68, 24), "Apple")) LoadQuickDrawPreset("Red Apple");
                    if (GUI.Button(new Rect(92, btnY, 68, 24), "Pizza")) LoadQuickDrawPreset("Pizza Slice");
                    if (GUI.Button(new Rect(164, btnY, 68, 24), "Donut")) LoadQuickDrawPreset("Glazed Donut");
                    if (GUI.Button(new Rect(236, btnY, 68, 24), "IceCream")) LoadQuickDrawPreset("Ice Cream");
                }
                else if (_selectedSandboxTab == 2) // Animals
                {
                    if (GUI.Button(new Rect(20, btnY, 68, 24), "Cat")) LoadQuickDrawPreset("Cat Figurine");
                    if (GUI.Button(new Rect(92, btnY, 68, 24), "Duck")) LoadQuickDrawPreset("Rubber Ducky");
                    if (GUI.Button(new Rect(164, btnY, 68, 24), "Fish")) LoadQuickDrawPreset("Tropical Fish");
                    if (GUI.Button(new Rect(236, btnY, 68, 24), "Frog")) LoadQuickDrawPreset("Tree Frog");
                }
                else if (_selectedSandboxTab == 3) // Tools
                {
                    if (GUI.Button(new Rect(20, btnY, 68, 24), "Hammer")) LoadQuickDrawPreset("War Hammer");
                    if (GUI.Button(new Rect(92, btnY, 68, 24), "Anvil")) LoadQuickDrawPreset("Blacksmith Anvil");
                    if (GUI.Button(new Rect(164, btnY, 68, 24), "Guitar")) LoadQuickDrawPreset("Acoustic Guitar");
                    if (GUI.Button(new Rect(236, btnY, 68, 24), "Tree")) LoadQuickDrawPreset("Pine Tree");
                }
                else // Weapons & Treasures
                {
                    if (GUI.Button(new Rect(20, btnY, 68, 24), "Sword")) LoadQuickDrawPreset("Excalibur Sword");
                    if (GUI.Button(new Rect(92, btnY, 68, 24), "Cannon")) LoadQuickDrawPreset("Siege Cannon");
                    if (GUI.Button(new Rect(164, btnY, 68, 24), "Crown")) LoadQuickDrawPreset("Royal Crown");
                    if (GUI.Button(new Rect(236, btnY, 68, 24), "Diamond")) LoadQuickDrawPreset("Magic Diamond");
                }

                // Sandbox Physics Spawning Controls
                btnY += 28;
                if (GUI.Button(new Rect(20, btnY, 175, 24), "<b>🎲 Drop Prop with Physics</b>"))
                {
                    if (sandboxEngine != null && _lastSandboxResult.occupancyGrid != null)
                    {
                        Vector3 dropPos = new Vector3(
                            UnityEngine.Random.Range(-0.35f, 0.35f),
                            0.5f,
                            UnityEngine.Random.Range(-0.25f, 0.25f)
                        );
                        sandboxEngine.SpawnPhysicalProp(_lastSandboxResult, dropPos);
                    }
                }

                int propCount = sandboxEngine != null ? sandboxEngine.SpawnedPropCount : 0;
                if (GUI.Button(new Rect(200, btnY, 104, 24), $"🧹 Clear ({propCount})"))
                {
                    if (sandboxEngine != null) sandboxEngine.ClearSpawnedProps();
                }
            }
            else // Studio12Class
            {
                GUI.Label(new Rect(20, btnY, 280, 18), "<b>Active Category Presets:</b>");
                btnY += 18;

                string[] tabNames = new string[] { "Weapons", "Toys", "Electronics" };
                int newTab = GUI.Toolbar(new Rect(20, btnY, 285, 24), _selectedCategoryTab, tabNames);
                if (newTab != _selectedCategoryTab)
                {
                    _selectedCategoryTab = newTab;
                }

                btnY += 28;
                if (_selectedCategoryTab == 0) // Weapons
                {
                    if (GUI.Button(new Rect(20, btnY, 68, 24), "Sword")) LoadPreset("Sword");
                    if (GUI.Button(new Rect(92, btnY, 68, 24), "Pistol")) LoadPreset("Pistol");
                    if (GUI.Button(new Rect(164, btnY, 68, 24), "Blaster")) LoadPreset("Blaster");
                    if (GUI.Button(new Rect(236, btnY, 68, 24), "Axe")) LoadPreset("Axe");
                }
                else if (_selectedCategoryTab == 1) // Toys
                {
                    if (GUI.Button(new Rect(20, btnY, 68, 24), "Teddy")) LoadPreset("Teddy Bear");
                    if (GUI.Button(new Rect(92, btnY, 68, 24), "Duck")) LoadPreset("Rubber Duck");
                    if (GUI.Button(new Rect(164, btnY, 68, 24), "Robot")) LoadPreset("Toy Robot");
                    if (GUI.Button(new Rect(236, btnY, 68, 24), "Bunny")) LoadPreset("Toy Bunny");
                }
                else // Electronics
                {
                    if (GUI.Button(new Rect(20, btnY, 90, 24), "Gamepad")) LoadPreset("Gamepad");
                    if (GUI.Button(new Rect(116, btnY, 90, 24), "Retro TV")) LoadPreset("Retro TV");
                    if (GUI.Button(new Rect(212, btnY, 92, 24), "Game Boy")) LoadPreset("Game Boy");
                }
            }

            // 6. Capability Guide Toggle
            btnY += 30;
            string guideLabel = _showDisclaimerModal ? "Hide Workshop Guide ▲" : "📖 Workshop Capability Guide ▼";
            if (GUI.Button(new Rect(20, btnY, 285, 24), guideLabel))
            {
                _showDisclaimerModal = !_showDisclaimerModal;
            }

            // 7. Status & Results
            btnY += 28;
            autoGenerateOnRelease = GUI.Toggle(new Rect(20, btnY, 280, 18), autoGenerateOnRelease, " Live Reconstruct on Mouse Up");

            btnY += 20;
            if (activeMode == GamePlayMode.Sandbox50Class)
            {
                if (!string.IsNullOrEmpty(_lastSandboxResult.label))
                {
                    string catBadge = _lastSandboxResult.isNoise
                        ? "<color=#ff5555><b>[NOISE REJECTED]</b></color>"
                        : $"<color=#55ff88><b>{_lastSandboxResult.label}</b></color> ({_lastSandboxResult.confidence * 100f:F0}%) [Family: {_lastSandboxResult.family}]";
                    GUI.Label(new Rect(20, btnY, 280, 18), $"<size=11>AI Class: {catBadge}</size>");
                    btnY += 18;

                    if (_lastSandboxResult.top5 != null && _lastSandboxResult.top5.Count > 1)
                    {
                        string topText = "";
                        for (int k = 0; k < Mathf.Min(3, _lastSandboxResult.top5.Count); k++)
                        {
                            topText += $"{k + 1}. {_lastSandboxResult.top5[k].Key} ({_lastSandboxResult.top5[k].Value * 100f:F0}%) ";
                        }
                        GUI.Label(new Rect(20, btnY, 280, 16), $"<size=9><color=#88ccff>Top: {topText}</color></size>");
                        btnY += 16;
                    }
                }
            }
            else // Studio12Class
            {
                if (multiTaskEngine != null && _detectedCategory != ReconstructedCategory.Unknown)
                {
                    string catBadge = _isOutOfDistribution
                        ? "<color=#ff5555><b>[OOD REJECTED]</b></color>"
                        : $"<color=#55ff88><b>{_detectedCategory}</b></color> ({_categoryConfidence * 100f:F0}%)";
                    string viewBadge = $"<color=#55ccff><b>{_detectedView}</b></color> ({_viewConfidence * 100f:F0}%)";
                    GUI.Label(new Rect(20, btnY, 280, 18), $"AI Class: {catBadge} | View: {viewBadge}");
                    btnY += 18;
                }
            }

            GUI.Label(new Rect(20, btnY, 280, 36), $"<color=#ffdd66>★ {_lastCategoryName}: {_lastGeneratedItemName}</color>\n<size=10>{_statusMessage}</size>");

            // 8. Player Capability & Disclaimer Guide Modal
            if (_showDisclaimerModal)
            {
                DrawDisclaimerGuideWindow(new Rect(330, 10, 470, 640));
            }
        }

        private void DrawDisclaimerGuideWindow(Rect windowRect)
        {
            GUI.Box(windowRect, "");
            float y = windowRect.y + 12;

            GUI.Label(new Rect(windowRect.x + 15, y, 440, 25), "<b>★ AI WORKSHOP & CAPABILITY GUIDE ★</b>");
            y += 26;

            GUI.Label(new Rect(windowRect.x + 15, y, 440, 20), "<color=#88ffaa><b>MODE 1: UNIVERSAL SANDBOX (50+ QUICK, DRAW! CLASSES)</b></color>");
            y += 20;

            string sandboxText = 
                "• <b>Vehicles:</b> Airplane, Helicopter, Sports Car, Bicycle, Sailboat, Skateboard.\n" +
                "• <b>Food & Cuisine:</b> Red Apple, Banana, Pizza Slice, Glazed Donut, Ice Cream, Burger.\n" +
                "• <b>Animals & Creatures:</b> Rubber Ducky, Cat, Dog, Grizzly Bear, Tropical Fish, Songbird, Tree Frog.\n" +
                "• <b>Tools & Instruments:</b> War Hammer, Screwdriver, Hand Saw, Pliers, Flashlight, Umbrella, Key, Anvil, Acoustic Guitar, Wall Clock, Chair.\n" +
                "• <b>Nature & Elements:</b> Pine Tree, Flower, Cactus, Mushroom, Campfire, Mountain, Cloud, Sun, Star.\n" +
                "• <b>Weapons & Armor:</b> Broadsword, Battleaxe, Rifle, Cannon, Dagger, Boomerang, Knight Helmet, Royal Crown, Magic Diamond.\n" +
                "• <b>Interactive Physics:</b> Click <b>'🎲 Drop Prop'</b> to spawn physical rigidbodies that tumble and collide!";

            GUI.Label(new Rect(windowRect.x + 15, y, 440, 190), $"<size=10>{sandboxText}</size>");
            y += 195;

            GUI.Label(new Rect(windowRect.x + 15, y, 440, 20), "<color=#88ffaa><b>MODE 2: CURATED STUDIO (12-CLASS MULTI-TASK NET)</b></color>");
            y += 20;

            string studioText =
                "• Hand-sculpted 3D voxel models across Weapons (Sword, Shield, Blaster, Axe), Toys (Teddy Bear, Rubber Duck, Toy Robot, Bunny), and Electronics (Gamepad, Retro TV, Game Boy).\n" +
                "• <b>Tri-View Multi-Task:</b> Classifies category, predicts 3D camera viewpoint (Front/Left/Top), and blends latent codes (e.g., Sword + Axe hybrid).\n" +
                "• <b>Out-Of-Distribution Rejection:</b> Scribbles and random noise are automatically detected and filtered.";

            GUI.Label(new Rect(windowRect.x + 15, y, 440, 130), $"<size=10>{studioText}</size>");
            y += 135;

            GUI.Label(new Rect(windowRect.x + 15, y, 440, 20), "<color=#ffdd66><b>DRAWING TIPS & FEASIBILITY:</b></color>");
            y += 20;

            string tipsText =
                "✔ <b>Best Results:</b> Closed contours, distinct silhouettes, 3-4px stroke thickness.\n" +
                "⚠️ <b>Outline sketches</b> in Sandbox mode are automatically closed and given smooth spherical/tubular volume!\n" +
                "✖ <b>Avoid:</b> Unconnected scattered dots, single hairline scratches (<1px), or unclosed scribble webs.";

            GUI.Label(new Rect(windowRect.x + 15, y, 440, 90), $"<size=10>{tipsText}</size>");

            if (GUI.Button(new Rect(windowRect.x + 15, windowRect.yMax - 36, 440, 28), "<b>Got it! Close Workshop Guide</b>"))
            {
                _showDisclaimerModal = false;
            }
        }

        private string GetCanonicalItemName(ReconstructedCategory cat)
        {
            switch (cat)
            {
                case ReconstructedCategory.Sword: return "Legendary Buster Sword";
                case ReconstructedCategory.Shield: return "Paladin's Kite Shield";
                case ReconstructedCategory.Blaster: return "Sci-Fi Heavy Blaster";
                case ReconstructedCategory.Axe: return "Viking War Axe";
                case ReconstructedCategory.TeddyBear: return "Plush Teddy Bear Trophy";
                case ReconstructedCategory.RubberDuck: return "Classic Rubber Ducky";
                case ReconstructedCategory.ToyRobot: return "Retro Wind-Up Robot";
                case ReconstructedCategory.ToyBunny: return "Fluffy Bunny Figurine";
                case ReconstructedCategory.Gamepad: return "DualGrip Pro Game Controller";
                case ReconstructedCategory.RetroTV: return "Vintage Cathode CRT TV";
                case ReconstructedCategory.GameBoy: return "Pocket 8-Bit Console";
                default: return "Procedural Voxel Asset";
            }
        }
    }
}

