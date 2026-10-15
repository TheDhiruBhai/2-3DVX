using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRDraw2Voxel.Drawing;
using VRDraw2Voxel.Inference;

namespace VRDraw2Voxel.Interaction
{
    public enum VRGameState
    {
        RpsMatch,           // Hand tracking mini-game active
        CanvasActive,       // Player won RPS, drawing slate unlocked
        Inferring,          // Ingesting tensor and running Sentis inference
        PropSpawned         // Voxel prop spawned with physics on tabletop
    }

    /// <summary>
    /// Milestone 5 VR Game Manager & Real-Time Engine Reconstruction Bridge.
    /// Orchestrates:
    /// 1. OpenXR Hand Gesture loop (Rock-Paper-Scissors) unlocking the Tabletop Canvas
    /// 2. Spatial pen drawing on the in-world slate quad
    /// 3. Physical tabletop 3D button press to commit drawing
    /// 4. Sentis neural inference with OOD scribble rejection (<40% confidence)
    /// 5. Burst-compiled colored voxel mesh generation in <5 ms
    /// 6. Spawning interactive props with Rigidbody and BoxCollider
    /// 7. Strict 15-prop FIFO queue management to sustain 90 FPS VR headroom.
    /// </summary>
    public class VRGameManager : MonoBehaviour
    {
        [Header("Component References")]
        [SerializeField] private VRHandGestureDetector gestureDetector;
        [SerializeField] private VRTabletopCanvas tabletopCanvas;
        [SerializeField] private UniversalSandboxEngine sandboxEngine;

        [Header("Tabletop Environment")]
        [SerializeField] private Transform tableSpawnPoint;
        [SerializeField] private Transform physicalCommitButton;
        [SerializeField] private float buttonPressDistance = 0.04f; // 4 cm press activation
        [SerializeField] private int maxActiveProps = 15;

        [Header("Confidence & Quality Filter")]
        [SerializeField] private float minConfidenceThreshold = 0.40f; // Reject <40%
        [SerializeField] private int minSolidStrokePixels = 30;

        [Header("Audio Feedback")]
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private AudioClip buttonClickClip;
        [SerializeField] private AudioClip inferringClip;
        [SerializeField] private AudioClip propSpawnedClip;
        [SerializeField] private AudioClip rejectionBuzzClip;

        // Public State
        public VRGameState CurrentState { get; private set; } = VRGameState.RpsMatch;
        public int ActivePropCount => _activeProps.Count;
        public string LastReconstructionInfo { get; private set; } = "Ready";

        // Internal Queue
        private readonly Queue<GameObject> _activeProps = new Queue<GameObject>();
        private bool _isInferring = false;
        private Vector3 _buttonDefaultPos;

        private void Awake()
        {
            if (gestureDetector == null) gestureDetector = FindFirstObjectByType<VRHandGestureDetector>();
            if (tabletopCanvas == null) tabletopCanvas = FindFirstObjectByType<VRTabletopCanvas>();
            if (sandboxEngine == null) sandboxEngine = FindFirstObjectByType<UniversalSandboxEngine>();

            if (physicalCommitButton != null)
            {
                _buttonDefaultPos = physicalCommitButton.localPosition;
            }
        }

        private void OnEnable()
        {
            if (gestureDetector != null)
            {
                gestureDetector.OnTabletopCanvasActivated += HandleCanvasActivated;
                gestureDetector.OnRoundResolved += HandleRpsRoundResolved;
            }

            if (tabletopCanvas != null)
            {
                tabletopCanvas.OnStrokeFinished += HandleStrokeFinished;
            }
        }

        private void OnDisable()
        {
            if (gestureDetector != null)
            {
                gestureDetector.OnTabletopCanvasActivated -= HandleCanvasActivated;
                gestureDetector.OnRoundResolved -= HandleRpsRoundResolved;
            }

            if (tabletopCanvas != null)
            {
                tabletopCanvas.OnStrokeFinished -= HandleStrokeFinished;
            }
        }

        private void Start()
        {
            // Initial state: Start the RPS match
            if (gestureDetector != null)
            {
                CurrentState = VRGameState.RpsMatch;
                gestureDetector.StartMatchRound();
            }
            else
            {
                // Fallback to canvas active if gesture detector is not present
                CurrentState = VRGameState.CanvasActive;
            }
        }

        private void Update()
        {
            // Monitor physical tabletop commit button press
            CheckPhysicalButtonPress();

            // Keyboard shortcut for testing in Editor: [Enter] or [Return]
            if (Input.GetKeyDown(KeyCode.Return) && CurrentState == VRGameState.CanvasActive)
            {
                CommitDrawingAndReconstruct();
            }
            if (Input.GetKeyDown(KeyCode.C) && CurrentState == VRGameState.CanvasActive)
            {
                if (tabletopCanvas != null) tabletopCanvas.ClearCanvas();
            }
        }

        private void HandleCanvasActivated()
        {
            CurrentState = VRGameState.CanvasActive;
            LastReconstructionInfo = "Canvas Active! Draw your sketch and press the Stamp button.";
            Debug.Log("<color=#66ddff>[VRGameManager] State: CANVAS_ACTIVE. Drawing slate unlocked on tabletop.</color>");
        }

        private void HandleRpsRoundResolved(RPSHandGesture player, RPSHandGesture ai)
        {
            if (gestureDetector != null && gestureDetector.MatchState == RPSMatchState.PlayerWon)
            {
                LastReconstructionInfo = $"You WON ({player} beats {ai})! Slate unlocked.";
            }
            else if (gestureDetector != null && gestureDetector.MatchState == RPSMatchState.PlayerLost)
            {
                LastReconstructionInfo = $"AI won ({ai} beats {player}). Retrying round...";
            }
            else
            {
                LastReconstructionInfo = $"Tie ({player})! Shoot again...";
            }
        }

        private void HandleStrokeFinished()
        {
            if (CurrentState == VRGameState.CanvasActive)
            {
                LastReconstructionInfo = $"Stroke drawn ({tabletopCanvas.ActiveStrokePixels} px). Press Commit Button to materialize 3D prop.";
            }
        }

        /// <summary>
        /// Checks if a VR hand or finger tip presses the in-world physical button on the table.
        /// </summary>
        private void CheckPhysicalButtonPress()
        {
            if (physicalCommitButton == null || CurrentState != VRGameState.CanvasActive || _isInferring) return;

            // In Editor, raycast or mouse click on the button
            if (Input.GetMouseButtonDown(0))
            {
                Ray ray = Camera.main != null ? Camera.main.ScreenPointToRay(Input.mousePosition) : default;
                if (Physics.Raycast(ray, out RaycastHit hit))
                {
                    if (hit.transform == physicalCommitButton || hit.transform.IsChildOf(physicalCommitButton))
                    {
                        AnimateAndTriggerButton();
                    }
                }
            }
        }

        /// <summary>
        /// External or physical collider trigger for the 3D Stamp button.
        /// </summary>
        public void OnPhysicalButtonPressedByVRHand()
        {
            if (CurrentState == VRGameState.CanvasActive && !_isInferring)
            {
                AnimateAndTriggerButton();
            }
        }

        private void AnimateAndTriggerButton()
        {
            StartCoroutine(AnimateButtonCoroutine());
            CommitDrawingAndReconstruct();
        }

        private IEnumerator AnimateButtonCoroutine()
        {
            if (physicalCommitButton != null)
            {
                physicalCommitButton.localPosition = _buttonDefaultPos - new Vector3(0, buttonPressDistance, 0);
                PlaySound(buttonClickClip);
                yield return new WaitForSeconds(0.12f);
                physicalCommitButton.localPosition = _buttonDefaultPos;
            }
        }

        /// <summary>
        /// Main Milestone 5 Reconstruction Bridge:
        /// Ingests canvas into Sentis tensor, evaluates OOD confidence filter,
        /// generates Burst colored voxel mesh, and spawns interactive 3D prop.
        /// </summary>
        public void CommitDrawingAndReconstruct()
        {
            if (_isInferring || tabletopCanvas == null || sandboxEngine == null) return;

            if (tabletopCanvas.ActiveStrokePixels < minSolidStrokePixels)
            {
                PlaySound(rejectionBuzzClip);
                LastReconstructionInfo = "<color=#ff6666>[REJECTED] Canvas is empty or stroke is too small!</color>";
                return;
            }

            _isInferring = true;
            CurrentState = VRGameState.Inferring;
            PlaySound(inferringClip);
            LastReconstructionInfo = "Materializing 3D Voxel Prop...";

            // 1. Ingest 2-Channel [Stroke, SDF] array
            float[] buffer2Ch = tabletopCanvas.PreprocessToTwoChannel64x64(out float[] strokeMask64, out float ar, out float fill);

            // 2. Dispatch to UniversalSandboxEngine on Sentis
            sandboxEngine.PredictSandboxAsync(buffer2Ch, (result) =>
            {
                _isInferring = false;

                // 3. OOD Scribble & Low Confidence Filter
                if (result.isNoise || result.confidence < minConfidenceThreshold)
                {
                    PlaySound(rejectionBuzzClip);
                    LastReconstructionInfo = $"<color=#ff6666>[REJECTED ({result.confidence * 100f:F0}%)] Unstructured scribble or low confidence.</color>";
                    CurrentState = VRGameState.CanvasActive;
                    return;
                }

                // 4. Determine Spawn Position on Tabletop
                Vector3 spawnPos = tableSpawnPoint != null
                    ? tableSpawnPoint.position + new Vector3(UnityEngine.Random.Range(-0.1f, 0.1f), 0.15f, UnityEngine.Random.Range(-0.1f, 0.1f))
                    : tabletopCanvas.transform.position + new Vector3(0, 0.25f, 0);

                // 5. Spawn Vertex-Colored 3D Prop with Rigidbody & BoxCollider
                GameObject prop = sandboxEngine.SpawnPhysicalProp(result, spawnPos);
                if (prop != null)
                {
                    // Enforce 15-prop FIFO queue to maintain 90 FPS VR headroom
                    EnqueueProp(prop);

                    PlaySound(propSpawnedClip);
                    LastReconstructionInfo = $"Spawned <b>{result.label}</b> ({result.family}, {result.confidence * 100f:F0}%) in {result.latencyMs:F1}ms!";
                    CurrentState = VRGameState.PropSpawned;

                    // Clear slate for next sketch
                    tabletopCanvas.ClearCanvas();
                }
                else
                {
                    LastReconstructionInfo = "Reconstruction failed to generate mesh.";
                    CurrentState = VRGameState.CanvasActive;
                }
            });
        }

        private void EnqueueProp(GameObject newProp)
        {
            _activeProps.Enqueue(newProp);

            // Maintain maxActiveProps (FIFO recycling)
            while (_activeProps.Count > maxActiveProps)
            {
                GameObject oldest = _activeProps.Dequeue();
                if (oldest != null)
                {
                    Destroy(oldest);
                }
            }
        }

        /// <summary>
        /// Clears all active props from the table.
        /// </summary>
        public void ClearAllProps()
        {
            while (_activeProps.Count > 0)
            {
                GameObject p = _activeProps.Dequeue();
                if (p != null) Destroy(p);
            }
            if (sandboxEngine != null) sandboxEngine.ClearSpawnedProps();
            LastReconstructionInfo = "All tabletop props cleared.";
        }

        private void PlaySound(AudioClip clip)
        {
            if (audioSource != null && clip != null)
            {
                audioSource.PlayOneShot(clip);
            }
        }

        private void OnGUI()
        {
            // Desktop HUD Overlay for VR Testing & Debugging
            GUI.Box(new Rect(Screen.width - 340, 20, 320, 210), "");
            GUI.Label(new Rect(Screen.width - 330, 26, 300, 22), "<b><size=13><color=#44ffbb>VR Tabletop & Gesture HUD</color></size></b>");

            string stateColor = CurrentState == VRGameState.CanvasActive ? "#66ff66" : (CurrentState == VRGameState.Inferring ? "#ffbb33" : "#66ccff");
            GUI.Label(new Rect(Screen.width - 330, 50, 300, 20), $"State: <color={stateColor}><b>{CurrentState}</b></color> | Props: <b>{ActivePropCount}/{maxActiveProps}</b>");

            if (gestureDetector != null)
            {
                string htStatus = gestureDetector.IsHandTrackingActive ? "<color=#33ff77>OpenXR Active</color>" : "<color=#ffaa33>Editor Simulation</color>";
                GUI.Label(new Rect(Screen.width - 330, 72, 300, 18), $"Hand Tracking: {htStatus}");
                GUI.Label(new Rect(Screen.width - 330, 90, 300, 18), $"RPS Gesture: <b>{gestureDetector.CurrentGesture}</b> (vs AI: {gestureDetector.AiOpponentGesture})");
                GUI.Label(new Rect(Screen.width - 330, 108, 300, 18), $"Fingers Curl: Idx={gestureDetector.IndexCurl:F2} Mid={gestureDetector.MiddleCurl:F2}");
            }

            GUI.Label(new Rect(Screen.width - 330, 128, 300, 34), $"<size=10><color=#eeeeee>{LastReconstructionInfo}</color></size>");

            if (GUI.Button(new Rect(Screen.width - 330, 168, 145, 26), "<b>Stamp & Spawn</b>"))
            {
                CommitDrawingAndReconstruct();
            }

            if (GUI.Button(new Rect(Screen.width - 175, 168, 145, 26), "Clear Slate & Props"))
            {
                if (tabletopCanvas != null) tabletopCanvas.ClearCanvas();
                ClearAllProps();
            }

            if (GUI.Button(new Rect(Screen.width - 330, 198, 300, 22), "<size=10>Trigger RPS Match [Space]</size>"))
            {
                if (gestureDetector != null) gestureDetector.StartMatchRound();
            }
        }
    }
}
