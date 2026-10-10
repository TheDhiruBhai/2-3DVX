using System;
using UnityEngine;

using UnityEngine.XR.OpenXR.Features.OculusQuestSupport;
#if LIFECYCLE_APIS_AVAILABLE
using Unity.Scripting.LifecycleManagement;
#endif

namespace UnityEditor.XR.OpenXR.Features.OculusQuestSupport
{
    [CustomEditor(typeof(OculusQuestFeature))]
    [Obsolete("OpenXR.Features.OculusQuestSupport.OculusQuestFeatureEditor is deprecated. Please use OpenXR.Features.MetaQuestSupport.MetaQuestFeatureEditor instead.", false)]
    internal class OculusQuestFeatureEditor : Editor
    {
        private SerializedProperty targetQuest;
        private SerializedProperty targetQuest2;

#if LIFECYCLE_APIS_AVAILABLE
        // Static UI label caches; content never changes at runtime.
        [NoAutoStaticsCleanup]
#endif
        static GUIContent s_TargetQuestLabel = EditorGUIUtility.TrTextContent("Quest");
#if LIFECYCLE_APIS_AVAILABLE
        [NoAutoStaticsCleanup]
#endif
        static GUIContent s_TargetQuest2Label = EditorGUIUtility.TrTextContent("Quest 2");

        void OnEnable()
        {
            targetQuest = serializedObject.FindProperty("targetQuest");
            targetQuest2 = serializedObject.FindProperty("targetQuest2");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            EditorGUILayout.LabelField("Target Devices", EditorStyles.boldLabel);

            EditorGUILayout.PropertyField(targetQuest, s_TargetQuestLabel);
            EditorGUILayout.PropertyField(targetQuest2, s_TargetQuest2Label);

            serializedObject.ApplyModifiedProperties();
        }
    }
}
