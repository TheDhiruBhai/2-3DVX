using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEditor.Build;
using UnityEditor.XR.OpenXR;

using UnityEngine;
using UnityEngine.XR.OpenXR;
#if LIFECYCLE_APIS_AVAILABLE
using Unity.Scripting.LifecycleManagement;
#endif

namespace UnityEditor.XR.OpenXR.Features
{
    internal static class KnownFeatureSets
    {
#if LIFECYCLE_APIS_AVAILABLE
        // Static lookup table populated once at declaration and never mutated.
        [NoAutoStaticsCleanup]
#endif
        internal static Dictionary<BuildTargetGroup, OpenXRFeatureSetManager.FeatureSet[]> k_KnownFeatureSets =
            new Dictionary<BuildTargetGroup, OpenXRFeatureSetManager.FeatureSet[]>() { };
    }
}
