using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands;

namespace VRDraw2Voxel.Interaction
{
    public enum RPSHandGesture
    {
        None = 0,
        Rock = 1,
        Paper = 2,
        Scissors = 3
    }

    public enum RPSMatchState
    {
        Idle,
        Countdown3,
        Countdown2,
        Countdown1,
        Shoot,
        PlayerWon,
        PlayerLost,
        Tie
    }

    /// <summary>
    /// OpenXR VR Hand Tracking & Gesture Loop for Milestone 5.
    /// Tracks hand joints via XRHandSubsystem, computes joint flexion/curl metrics,
    /// runs a 3-2-1 Rock-Paper-Scissors match against an AI opponent,
    /// and activates VR_CANVAS_ACTIVE upon player victory.
    /// Includes seamless Editor simulation mode (keys 1/2/3/Space).
    /// </summary>
    public class VRHandGestureDetector : MonoBehaviour
    {
        [Header("Hand Tracking Settings")]
        [SerializeField] private Handedness trackedHand = Handedness.Right;
        [SerializeField] private float curlThreshold = 0.75f;    // >75% flexion is curled
        [SerializeField] private float extendThreshold = 0.25f;  // <25% flexion is extended flat
        [SerializeField] private float minConfidence = 0.5f;

        [Header("Match & Countdown Settings")]
        [SerializeField] private float countdownStepDuration = 1.0f;
        [SerializeField] private bool autoRestartOnLossOrTie = true;
        [SerializeField] private float restartDelay = 2.0f;

        [Header("Audio Feedback (Optional)")]
        [SerializeField] private AudioSource audioSource;
        [SerializeField] private AudioClip countdownBeepClip;
        [SerializeField] private AudioClip victoryClip;
        [SerializeField] private AudioClip defeatClip;

        // Public Events
        public event Action<RPSHandGesture> OnPlayerGestureDetected;
        public event Action<RPSMatchState, int> OnCountdownTick;
        public event Action<RPSHandGesture, RPSHandGesture> OnRoundResolved; // (player, ai)
        public event Action OnPlayerWon;
        public event Action OnTabletopCanvasActivated;

        // Current Live State
        public RPSHandGesture CurrentGesture { get; private set; } = RPSHandGesture.None;
        public RPSHandGesture AiOpponentGesture { get; private set; } = RPSHandGesture.None;
        public RPSMatchState MatchState { get; private set; } = RPSMatchState.Idle;
        public bool IsCanvasActive { get; private set; } = false;
        public bool IsHandTrackingActive => _handSubsystem != null && _handSubsystem.running;

        // Finger Flexion Metrics (0.0 = flat, 1.0 = fully curled)
        public float IndexCurl { get; private set; }
        public float MiddleCurl { get; private set; }
        public float RingCurl { get; private set; }
        public float LittleCurl { get; private set; }
        public float ThumbCurl { get; private set; }

        private XRHandSubsystem _handSubsystem;
        private static readonly List<XRHandSubsystem> s_Subsystems = new List<XRHandSubsystem>();
        private Coroutine _matchCoroutine;

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

            if (_handSubsystem != null && _handSubsystem.running)
            {
                PollHandJoints();
            }
            else
            {
                HandleEditorSimulationInput();
            }
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
        /// Reads joint positions from the active XRHandSubsystem and classifies Rock, Paper, or Scissors.
        /// </summary>
        private void PollHandJoints()
        {
            XRHand hand = (trackedHand == Handedness.Right) ? _handSubsystem.rightHand : _handSubsystem.leftHand;
            if (!hand.isTracked)
            {
                CurrentGesture = RPSHandGesture.None;
                return;
            }

            XRHandJoint palm = hand.GetJoint(XRHandJointID.Palm);
            if (!palm.TryGetPose(out Pose palmPose)) return;

            Vector3 palmPos = palmPose.position;

            // Calculate normalized curl (0..1) for each finger relative to palm
            IndexCurl = CalculateFingerCurl(hand, XRHandJointID.IndexTip, XRHandJointID.IndexProximal, palmPos);
            MiddleCurl = CalculateFingerCurl(hand, XRHandJointID.MiddleTip, XRHandJointID.MiddleProximal, palmPos);
            RingCurl = CalculateFingerCurl(hand, XRHandJointID.RingTip, XRHandJointID.RingProximal, palmPos);
            LittleCurl = CalculateFingerCurl(hand, XRHandJointID.LittleTip, XRHandJointID.LittleProximal, palmPos);
            ThumbCurl = CalculateFingerCurl(hand, XRHandJointID.ThumbTip, XRHandJointID.ThumbProximal, palmPos);

            bool indexCurled = IndexCurl >= curlThreshold;
            bool middleCurled = MiddleCurl >= curlThreshold;
            bool ringCurled = RingCurl >= curlThreshold;
            bool littleCurled = LittleCurl >= curlThreshold;

            bool indexExt = IndexCurl <= extendThreshold;
            bool middleExt = MiddleCurl <= extendThreshold;
            bool ringExt = RingCurl <= extendThreshold;
            bool littleExt = LittleCurl <= extendThreshold;

            RPSHandGesture gesture = RPSHandGesture.None;

            // 1. Rock: All 4 fingers curled toward palm (flexion > 75%)
            if (indexCurled && middleCurled && ringCurled && littleCurled)
            {
                gesture = RPSHandGesture.Rock;
            }
            // 2. Paper: All 5 fingers extended flat (flexion < 25%)
            else if (indexExt && middleExt && ringExt && littleExt)
            {
                gesture = RPSHandGesture.Paper;
            }
            // 3. Scissors: Index and Middle extended; Ring and Pinky curled
            else if (indexExt && middleExt && ringCurled && littleCurled)
            {
                gesture = RPSHandGesture.Scissors;
            }

            if (gesture != CurrentGesture)
            {
                CurrentGesture = gesture;
                OnPlayerGestureDetected?.Invoke(gesture);
            }
        }

        private float CalculateFingerCurl(XRHand hand, XRHandJointID tipId, XRHandJointID proxId, Vector3 palmPos)
        {
            XRHandJoint tip = hand.GetJoint(tipId);
            XRHandJoint prox = hand.GetJoint(proxId);

            if (!tip.TryGetPose(out Pose tipPose) || !prox.TryGetPose(out Pose proxPose))
            {
                return 0.5f;
            }

            float distTipPalm = Vector3.Distance(tipPose.position, palmPos);
            float distProxPalm = Vector3.Distance(proxPose.position, palmPos);
            float fingerLen = Vector3.Distance(tipPose.position, proxPose.position);

            // Distance ratio: when curled into palm, distTipPalm is minimal (< 0.04m)
            // When extended, distTipPalm is maximal (~0.08m to 0.10m)
            float curl = Mathf.Clamp01(1.0f - ((distTipPalm - 0.038f) / 0.052f));
            return curl;
        }

        /// <summary>
        /// Keyboard simulation for rapid iteration in Editor without VR headset attached.
        /// Keys: [1] Rock, [2] Paper, [3] Scissors, [Space] Start Countdown
        /// </summary>
        private void HandleEditorSimulationInput()
        {
            if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1))
            {
                CurrentGesture = RPSHandGesture.Rock;
                IndexCurl = MiddleCurl = RingCurl = LittleCurl = 0.95f;
                OnPlayerGestureDetected?.Invoke(CurrentGesture);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2))
            {
                CurrentGesture = RPSHandGesture.Paper;
                IndexCurl = MiddleCurl = RingCurl = LittleCurl = 0.05f;
                OnPlayerGestureDetected?.Invoke(CurrentGesture);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3))
            {
                CurrentGesture = RPSHandGesture.Scissors;
                IndexCurl = MiddleCurl = 0.05f;
                RingCurl = LittleCurl = 0.95f;
                OnPlayerGestureDetected?.Invoke(CurrentGesture);
            }

            if (Input.GetKeyDown(KeyCode.Space) && MatchState == RPSMatchState.Idle)
            {
                StartMatchRound();
            }
        }

        /// <summary>
        /// Starts the 3-2-1 countdown against the AI opponent hand.
        /// </summary>
        public void StartMatchRound()
        {
            if (_matchCoroutine != null) StopCoroutine(_matchCoroutine);
            _matchCoroutine = StartCoroutine(MatchLoopCoroutine());
        }

        private IEnumerator MatchLoopCoroutine()
        {
            IsCanvasActive = false;
            AiOpponentGesture = RPSHandGesture.None;

            // 3...
            MatchState = RPSMatchState.Countdown3;
            OnCountdownTick?.Invoke(MatchState, 3);
            PlaySound(countdownBeepClip);
            yield return new WaitForSeconds(countdownStepDuration);

            // 2...
            MatchState = RPSMatchState.Countdown2;
            OnCountdownTick?.Invoke(MatchState, 2);
            PlaySound(countdownBeepClip);
            yield return new WaitForSeconds(countdownStepDuration);

            // 1...
            MatchState = RPSMatchState.Countdown1;
            OnCountdownTick?.Invoke(MatchState, 1);
            PlaySound(countdownBeepClip);
            yield return new WaitForSeconds(countdownStepDuration);

            // Shoot! (Evaluate gestures)
            MatchState = RPSMatchState.Shoot;
            OnCountdownTick?.Invoke(MatchState, 0);

            // AI picks random valid move (Rock, Paper, Scissors)
            AiOpponentGesture = (RPSHandGesture)UnityEngine.Random.Range(1, 4);

            RPSHandGesture playerFinal = CurrentGesture;
            Debug.Log($"[VRHandGestureDetector] Round Shoot! Player: {playerFinal} vs AI: {AiOpponentGesture}");
            OnRoundResolved?.Invoke(playerFinal, AiOpponentGesture);

            if (playerFinal == RPSHandGesture.None)
            {
                // Player made no valid gesture in time
                MatchState = RPSMatchState.PlayerLost;
                PlaySound(defeatClip);
            }
            else if (playerFinal == AiOpponentGesture)
            {
                MatchState = RPSMatchState.Tie;
            }
            else if ((playerFinal == RPSHandGesture.Rock && AiOpponentGesture == RPSHandGesture.Scissors) ||
                     (playerFinal == RPSHandGesture.Paper && AiOpponentGesture == RPSHandGesture.Rock) ||
                     (playerFinal == RPSHandGesture.Scissors && AiOpponentGesture == RPSHandGesture.Paper))
            {
                // Player Won! Trigger victory audio and unlock VR_CANVAS_ACTIVE
                MatchState = RPSMatchState.PlayerWon;
                IsCanvasActive = true;
                PlaySound(victoryClip);
                OnPlayerWon?.Invoke();
                OnTabletopCanvasActivated?.Invoke();
                Debug.Log("<color=#33ff88><b>[VRHandGestureDetector] VICTORY! Tabletop drawing state 'VR_CANVAS_ACTIVE' unlocked!</b></color>");
                _matchCoroutine = null;
                yield break;
            }
            else
            {
                MatchState = RPSMatchState.PlayerLost;
                PlaySound(defeatClip);
            }

            if (autoRestartOnLossOrTie)
            {
                yield return new WaitForSeconds(restartDelay);
                MatchState = RPSMatchState.Idle;
                StartMatchRound();
            }
            _matchCoroutine = null;
        }

        private void PlaySound(AudioClip clip)
        {
            if (audioSource != null && clip != null)
            {
                audioSource.PlayOneShot(clip);
            }
        }

        /// <summary>
        /// Forces canvas active (e.g. for debug or direct sandbox entry).
        /// </summary>
        public void ForceActivateCanvas()
        {
            IsCanvasActive = true;
            OnTabletopCanvasActivated?.Invoke();
        }
    }
}
