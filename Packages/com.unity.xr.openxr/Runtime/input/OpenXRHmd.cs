using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem.XR;
using UnityEngine.Scripting;

namespace UnityEngine.XR.OpenXR.Input
{
    [Preserve, InputControlLayout(displayName = "OpenXR HMD")]
    internal class OpenXRHmd : XRHMD
    {
        /// <summary>
        /// Indicates whether the user is present and interacting with the device.
        /// </summary>
        [Preserve, InputControl]
        public ButtonControl userPresence { get; protected set; }

        /// <inheritdoc/>
        protected override void FinishSetup()
        {
            base.FinishSetup();
            userPresence = GetChildControl<ButtonControl>("userPresence");
        }
    }
}
