using System;
using System.Collections.Generic;
using UnityEngine;
using VRDraw2Voxel.Inference;

namespace VRDraw2Voxel.Drawing
{
    /// <summary>
    /// Captures 3D hand/controller stroke points in VR during gesture drawing,
    /// renders real-time visual feedback via LineRenderer, and sends normalized 
    /// strokes to the rasterizer and Sentis engine upon stroke completion.
    /// </summary>
    [RequireComponent(typeof(LineRenderer))]
    public class StrokeCapture : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private DrawingCanvasProjector canvasProjector;
        [SerializeField] private SentisVoxelEngine voxelEngine;
        [SerializeField] private Transform drawingTipTransform;
        [SerializeField] private Transform headCameraTransform;

        [Header("Stroke Tuning")]
        [SerializeField] private float minPointDistance = 0.015f; // 1.5 cm between captured points
        [SerializeField] private float maxStrokeDuration = 5.0f;

        private LineRenderer _lineRenderer;
        private readonly List<Vector3> _currentStrokePoints = new List<Vector3>();
        private bool _isDrawing = false;
        private float _strokeStartTime = 0f;

        public event Action<float[]> OnStrokeRasterized;

        private void Awake()
        {
            _lineRenderer = GetComponent<LineRenderer>();
            _lineRenderer.positionCount = 0;
            if (headCameraTransform == null && Camera.main != null)
            {
                headCameraTransform = Camera.main.transform;
            }
        }

        public void StartStroke()
        {
            _currentStrokePoints.Clear();
            _lineRenderer.positionCount = 0;
            _isDrawing = true;
            _strokeStartTime = Time.time;

            if (drawingTipTransform != null)
            {
                AddPoint(drawingTipTransform.position);
            }
        }

        public void UpdateStroke()
        {
            if (!_isDrawing || drawingTipTransform == null) return;

            Vector3 currentPos = drawingTipTransform.position;
            if (_currentStrokePoints.Count == 0 || Vector3.Distance(currentPos, _currentStrokePoints[_currentStrokePoints.Count - 1]) >= minPointDistance)
            {
                AddPoint(currentPos);
            }

            if (Time.time - _strokeStartTime > maxStrokeDuration)
            {
                EndStroke();
            }
        }

        public void EndStroke()
        {
            if (!_isDrawing) return;
            _isDrawing = false;

            if (_currentStrokePoints.Count >= 3)
            {
                ProcessAndDispatchStroke();
            }
            else
            {
                Debug.LogWarning("[StrokeCapture] Stroke discarded - too few sample points.");
                _lineRenderer.positionCount = 0;
            }
        }

        private void AddPoint(Vector3 point)
        {
            _currentStrokePoints.Add(point);
            _lineRenderer.positionCount = _currentStrokePoints.Count;
            _lineRenderer.SetPosition(_currentStrokePoints.Count - 1, point);
        }

        private void ProcessAndDispatchStroke()
        {
            if (canvasProjector == null)
            {
                Debug.LogError("[StrokeCapture] DrawingCanvasProjector is null.");
                return;
            }

            // 1. Rasterize stroke into 64x64 float buffer
            float[] rasterBuffer = canvasProjector.RasterizeStrokes(_currentStrokePoints, headCameraTransform);
            OnStrokeRasterized?.Invoke(rasterBuffer);

            // 2. Dispatch to Sentis engine
            if (voxelEngine != null)
            {
                voxelEngine.PredictVoxelsAsync(rasterBuffer);
            }
        }

        public void ClearCanvas()
        {
            _currentStrokePoints.Clear();
            if (_lineRenderer != null)
            {
                _lineRenderer.positionCount = 0;
            }
        }
    }
}
