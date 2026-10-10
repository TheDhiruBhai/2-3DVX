using System.Runtime.InteropServices;
using NUnit.Framework;
using UnityEngine.XR.OpenXR.NativeTypes;

namespace UnityEditor.XR.OpenXR.Tests.NativeTypes
{
    class XrSpatialPersistenceDataEXTTests
    {
        [Test]
        public void SizeValidation()
        {
            Assert.AreEqual(20, Marshal.SizeOf<XrSpatialPersistenceDataEXT>());
        }
    }
}
