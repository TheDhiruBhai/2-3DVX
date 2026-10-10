using UnityEngine;
using VRDraw2Voxel.VoxelMesh;

#if ENABLE_INPUT_SYSTEM && UNITY_XR_INTERACTION_TOOLKIT
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
#endif

namespace VRDraw2Voxel.Interaction
{
    /// <summary>
    /// Analyzes the principal dimension of the generated voxel weapon, calculates 
    /// the ergonomic hilt/handle position (0% - 20% along max axis), and attaches 
    /// a Rigidbody and XR Grab component for instant NPC melee combat.
    /// </summary>
    public class AutoGrabAttacher : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private GreedyVoxelMesher voxelMesher;

        [Header("Physics Settings")]
        [SerializeField] private float weaponMass = 1.5f;

        private void Awake()
        {
            if (voxelMesher != null)
            {
                voxelMesher.OnWeaponMeshConstructed += SetupGrabAndPhysics;
            }
        }

        private void OnDestroy()
        {
            if (voxelMesher != null)
            {
                voxelMesher.OnWeaponMeshConstructed -= SetupGrabAndPhysics;
            }
        }

        public void SetupGrabAndPhysics(GameObject weaponObj, Bounds bounds)
        {
            // 1. Determine Principal Longest Axis
            Vector3 size = bounds.size;
            Vector3 handleLocalOffset = Vector3.zero;

            if (size.y >= size.x && size.y >= size.z)
            {
                // Y is longest axis (standard vertical blade/bat/hammer)
                // Handle is 10% from bottom (negative Y boundary)
                float minY = bounds.min.y;
                float handleY = minY + size.y * 0.10f;
                handleLocalOffset = new Vector3(bounds.center.x, handleY, bounds.center.z);
            }
            else if (size.x >= size.y && size.x >= size.z)
            {
                // X is longest axis
                float minX = bounds.min.x;
                float handleX = minX + size.x * 0.10f;
                handleLocalOffset = new Vector3(handleX, bounds.center.y, bounds.center.z);
            }
            else
            {
                // Z is longest axis
                float minZ = bounds.min.z;
                float handleZ = minZ + size.z * 0.10f;
                handleLocalOffset = new Vector3(bounds.center.x, bounds.center.y, handleZ);
            }

            // 2. Create Grip Anchor Transform
            Transform gripTransform = weaponObj.transform.Find("GripAnchor");
            if (gripTransform == null)
            {
                GameObject gripGo = new GameObject("GripAnchor");
                gripGo.transform.SetParent(weaponObj.transform, false);
                gripTransform = gripGo.transform;
            }
            gripTransform.localPosition = handleLocalOffset;

            // 3. Configure Rigidbody
            Rigidbody rb = weaponObj.GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = weaponObj.AddComponent<Rigidbody>();
            }
            rb.mass = weaponMass;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

            // 4. Attach XR Interaction Toolkit or standard grab mechanism
#if ENABLE_INPUT_SYSTEM && UNITY_XR_INTERACTION_TOOLKIT
            XRGrabInteractable grabInteractable = weaponObj.GetComponent<XRGrabInteractable>();
            if (grabInteractable == null)
            {
                grabInteractable = weaponObj.AddComponent<XRGrabInteractable>();
            }
            grabInteractable.attachTransform = gripTransform;
            grabInteractable.throwOnDetach = true;
#endif

            Debug.Log($"[AutoGrabAttacher] Successfully configured weapon physics & grip at local pos: {handleLocalOffset}");
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.gameObject.CompareTag("NPC"))
            {
                Debug.Log($"[AutoGrabAttacher] Hit NPC {collision.gameObject.name} with impact velocity {collision.relativeVelocity.magnitude:F2} m/s!");
            }
        }
    }
}
