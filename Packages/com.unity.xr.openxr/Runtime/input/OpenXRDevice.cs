using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.XR;
using UnityEngine.Scripting;

namespace UnityEngine.XR.OpenXR.Input
{
    /// <summary>
    /// OpenXR Input System device
    /// </summary>
    /// <seealso cref="UnityEngine.InputSystem.InputDevice"/>
    [Preserve, InputControlLayout(displayName = "OpenXR Action Map")]
    public abstract class OpenXRDevice : UnityEngine.InputSystem.InputDevice
    {
        /// <summary>
        /// See [InputControl.FinishSetup](xref:UnityEngine.InputSystem.InputControl.FinishSetup)
        /// </summary>
        protected override void FinishSetup()
        {
            base.FinishSetup();

#if ENABLE_VR || UNITY_GAMECORE // UnityEngine.InputSystem.XR.XRDeviceDescriptor.characteristics is guarded with these defines starting with com.unity.inputsystem@1.14.2
            var deviceDescriptor = XRDeviceDescriptor.FromJson(description.capabilities);
            if (deviceDescriptor != null)
            {
                if ((deviceDescriptor.characteristics & InputDeviceCharacteristics.Left) != 0)
                    InputSystem.InputSystem.SetDeviceUsage(this, InputSystem.CommonUsages.LeftHand);
                else if ((deviceDescriptor.characteristics & InputDeviceCharacteristics.Right) != 0)
                    InputSystem.InputSystem.SetDeviceUsage(this, InputSystem.CommonUsages.RightHand);
            }
#endif
        }
    }
}
