using System;
using UnityEngine;

namespace VRDraw2Voxel.Interaction
{
    public enum RPSGesture
    {
        None = 0,
        Rock = 1,
        Paper = 2,
        Scissors = 3
    }

    /// <summary>
    /// Evaluates player hand tracking joint curls using lightweight heuristics (<0.5 ms)
    /// to recognize Rock, Paper, or Scissors, and triggers the drawing canvas upon winning.
    /// </summary>
    public class RPSDetector : MonoBehaviour
    {
        [Header("Hand Joint Transforms (Optional - or fed via OpenXR Hand Tracking)")]
        [SerializeField] private Transform thumbTip;
        [SerializeField] private Transform indexTip;
        [SerializeField] private Transform middleTip;
        [SerializeField] private Transform ringTip;
        [SerializeField] private Transform pinkyTip;
        [SerializeField] private Transform wrist;

        // Events
        public event Action<RPSGesture> OnPlayerGestureDetected;
        public event Action OnPlayerWonRound;
        public event Action OnPlayerLostRound;
        public event Action OnRoundTied;

        private RPSGesture _currentGesture = RPSGesture.None;
        private RPSGesture _npcGesture = RPSGesture.None;

        /// <summary>
        /// Evaluates joint curls to classify gesture.
        /// </summary>
        public RPSGesture ClassifyGesture(bool indexCurled, bool middleCurled, bool ringCurled, bool pinkyCurled)
        {
            // Rock: All 4 fingers curled
            if (indexCurled && middleCurled && ringCurled && pinkyCurled)
            {
                return RPSGesture.Rock;
            }

            // Paper: All fingers extended
            if (!indexCurled && !middleCurled && !ringCurled && !pinkyCurled)
            {
                return RPSGesture.Paper;
            }

            // Scissors: Index and middle extended, ring and pinky curled
            if (!indexCurled && !middleCurled && ringCurled && pinkyCurled)
            {
                return RPSGesture.Scissors;
            }

            return RPSGesture.None;
        }

        /// <summary>
        /// Plays a round against the NPC. If player wins, fires OnPlayerWonRound.
        /// </summary>
        public void ResolveRound(RPSGesture playerMove)
        {
            if (playerMove == RPSGesture.None) return;

            _currentGesture = playerMove;
            OnPlayerGestureDetected?.Invoke(playerMove);

            // NPC picks a random counter move (or fair move)
            _npcGesture = (RPSGesture)UnityEngine.Random.Range(1, 4);
            Debug.Log($"[RPSDetector] Player: {_currentGesture} vs NPC: {_npcGesture}");

            if (_currentGesture == _npcGesture)
            {
                OnRoundTied?.Invoke();
            }
            else if ((_currentGesture == RPSGesture.Rock && _npcGesture == RPSGesture.Scissors) ||
                     (_currentGesture == RPSGesture.Paper && _npcGesture == RPSGesture.Rock) ||
                     (_currentGesture == RPSGesture.Scissors && _npcGesture == RPSGesture.Paper))
            {
                Debug.Log("[RPSDetector] Player WON! Activating Drawing Canvas.");
                OnPlayerWonRound?.Invoke();
            }
            else
            {
                Debug.Log("[RPSDetector] Player lost.");
                OnPlayerLostRound?.Invoke();
            }
        }
    }
}
