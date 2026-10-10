using System;

using UnityEditor;

using UnityEngine;
#if LIFECYCLE_APIS_AVAILABLE
using Unity.Scripting.LifecycleManagement;
#endif

namespace UnityEditor.XR.OpenXR.Features
{
    static class CommonContent
    {
#if LIFECYCLE_APIS_AVAILABLE
        // Static UI label/icon caches; content never changes at runtime.
        [NoAutoStaticsCleanup]
#endif
        public static readonly GUIContent k_Download = new GUIContent("Download");
#if LIFECYCLE_APIS_AVAILABLE
        [NoAutoStaticsCleanup]
#endif
        public static readonly GUIContent k_WarningIcon = EditorGUIUtility.IconContent("console.warnicon.sml");
#if LIFECYCLE_APIS_AVAILABLE
        [NoAutoStaticsCleanup]
#endif
        public static readonly GUIContent k_ErrorIcon = EditorGUIUtility.IconContent("console.erroricon.sml");
#if LIFECYCLE_APIS_AVAILABLE
        [NoAutoStaticsCleanup]
#endif
        public static readonly GUIContent k_HelpIcon = EditorGUIUtility.IconContent("_Help@2x");

#if LIFECYCLE_APIS_AVAILABLE
        [NoAutoStaticsCleanup]
#endif
        public static readonly GUIContent k_Validation = new GUIContent("Your project has some settings that are incompatible with OpenXR. Click to open the project validator.");
#if LIFECYCLE_APIS_AVAILABLE
        [NoAutoStaticsCleanup]
#endif
        public static readonly GUIContent k_ValidationErrorIcon = new GUIContent("", CommonContent.k_ErrorIcon.image, k_Validation.text);
#if LIFECYCLE_APIS_AVAILABLE
        [NoAutoStaticsCleanup]
#endif
        public static readonly GUIContent k_ValidationWarningIcon = new GUIContent("", CommonContent.k_WarningIcon.image, k_Validation.text);
    }
}
