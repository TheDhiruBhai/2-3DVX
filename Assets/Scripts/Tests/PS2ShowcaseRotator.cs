using UnityEngine;
using VRDraw2Voxel.VoxelMesh;
using VRDraw2Voxel.Inference;

namespace VRDraw2Voxel.Tests
{
    /// <summary>
    /// Rotates the reconstructed 3D voxel model slowly 360 degrees like a classic
    /// PS2 trophy / game achievement viewer, with smooth bobbing and tilt.
    /// </summary>
    public class PS2ShowcaseRotator : MonoBehaviour
    {
        [Header("Showcase Rotation")]
        [Tooltip("Continuous yaw rotation speed in degrees per second.")]
        [SerializeField] private float rotationSpeed = 35.0f;

        [Tooltip("Slight tilt angle to show off 3D depth from 2D drawings.")]
        [SerializeField] private float showcaseTiltAngle = 12.0f;

        [Header("Floating / Bobbing")]
        [SerializeField] private bool enableFloating = true;
        [SerializeField] private float floatAmplitude = 0.04f;
        [SerializeField] private float floatFrequency = 1.5f;

        [Header("References")]
        [SerializeField] private ColoredVoxelMesher coloredVoxelMesher;
        [SerializeField] private GreedyVoxelMesher voxelMesher;

        private Vector3 _basePosition;
        private float _currentYaw = 0.0f;
        private Transform _meshHolder;

        private void Awake()
        {
            _basePosition = transform.position;

            // Create a child holder for pivot centering
            GameObject holder = new GameObject("ShowcasePivotHolder");
            holder.transform.SetParent(transform, false);
            _meshHolder = holder.transform;

            if (coloredVoxelMesher == null)
            {
                coloredVoxelMesher = GetComponent<ColoredVoxelMesher>() ?? FindFirstObjectByType<ColoredVoxelMesher>();
            }

            if (coloredVoxelMesher != null)
            {
                coloredVoxelMesher.OnColoredMeshConstructed += OnNewColoredMeshConstructed;
            }

            if (voxelMesher == null)
            {
                voxelMesher = GetComponent<GreedyVoxelMesher>() ?? FindFirstObjectByType<GreedyVoxelMesher>();
            }

            if (voxelMesher != null)
            {
                voxelMesher.OnWeaponMeshConstructed += OnNewWeaponConstructed;
            }
        }

        private void OnDestroy()
        {
            if (coloredVoxelMesher != null)
            {
                coloredVoxelMesher.OnColoredMeshConstructed -= OnNewColoredMeshConstructed;
            }

            if (voxelMesher != null)
            {
                voxelMesher.OnWeaponMeshConstructed -= OnNewWeaponConstructed;
            }
        }

        private void OnNewColoredMeshConstructed(GameObject itemObj, Bounds bounds, MultiTaskInferenceResult result)
        {
            OnNewWeaponConstructed(itemObj, bounds);
        }

        private void Update()
        {
            // 1. Continuous 360 degree slow PS2 showcase rotation
            _currentYaw += rotationSpeed * Time.deltaTime;
            if (_currentYaw >= 360f) _currentYaw -= 360f;

            Quaternion tilt = Quaternion.Euler(showcaseTiltAngle, 0, 0);
            Quaternion yaw = Quaternion.Euler(0, _currentYaw, 0);
            transform.rotation = yaw * tilt;

            // 2. Subtle achievement bobbing
            if (enableFloating)
            {
                float yOffset = Mathf.Sin(Time.time * floatFrequency) * floatAmplitude;
                transform.position = _basePosition + new Vector3(0, yOffset, 0);
            }
        }

        private void OnNewWeaponConstructed(GameObject weaponObj, Bounds bounds)
        {
            // Center the weapon's center of mass at the showcase pivot
            if (weaponObj != null)
            {
                // If the mesher is on this object or a child, adjust local position so rotation is centered
                Transform weaponTransform = weaponObj.transform;
                if (weaponTransform != transform)
                {
                    weaponTransform.SetParent(_meshHolder, true);
                }
                
                // Offset mesh position so its bounding center sits exactly at the pivot
                Vector3 centerOffset = bounds.center;
                _meshHolder.localPosition = -centerOffset;
            }

            // Reset rotation slightly to celebrate new generation
            _currentYaw = 0f;
        }

        public void SetRotationSpeed(float speed)
        {
            rotationSpeed = speed;
        }
    }
}
