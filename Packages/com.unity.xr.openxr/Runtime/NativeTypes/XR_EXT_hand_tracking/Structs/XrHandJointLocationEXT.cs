namespace UnityEngine.XR.OpenXR.NativeTypes
{
    /// <summary>
    /// Describes the location and radius of a hand joint. Provided by `XR_EXT_hand_tracking`.
    /// </summary>
    public readonly struct XrHandJointLocationEXT
    {
        /// <summary>
        /// Flags describing the validity and tracking state of the joint pose.
        /// </summary>
        public XrSpaceLocationFlags locationFlags { get; }

        /// <summary>
        /// The pose of the hand joint.
        /// </summary>
        public XrPosef pose { get; }

        /// <summary>
        /// The radius of the hand joint in meters.
        /// </summary>
        public float radius { get; }

        /// <summary>
        /// Construct an instance.
        /// </summary>
        /// <param name="locationFlags">Flags describing the validity and tracking state of the joint pose.</param>
        /// <param name="pose">The pose of the hand joint.</param>
        /// <param name="radius">The radius of the hand joint in meters.</param>
        public XrHandJointLocationEXT(
            XrSpaceLocationFlags locationFlags, XrPosef pose, float radius)
        {
            this.locationFlags = locationFlags;
            this.pose = pose;
            this.radius = radius;
        }
    }
}
