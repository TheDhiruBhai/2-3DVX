---
uid: openxr-composition-layers-architecture
---
# Composition layers architecture

Learn how the Unity OpenXR Plug-in implements the XR Composition Layers API and which OpenXR extensions it enables.

The [XR Composition Layers](xref:xr-layers-index) package contains a platform-agnostic API for managing composition layers in XR applications. The Unity OpenXR Plug-in implements Unity's composition layers API via OpenXR extensions. The [OpenXR Meta](xref:meta-openxr-manual) package provides additional runtime-specific layer types.

## About OpenXR

OpenXR is an open-source standard that defines an interface between XR apps and platform runtimes. The OpenXR specification contains two categories of features:

* Core features: present on every platform
* Extensions: optional and might not be implemented by some platforms.

The Unity OpenXR Plug-in integrates core features and enables the composition layer extensions outlined in [Extensions enabled via the Unity OpenXR Plug-in](#openxr-plugin-extensions).

## OpenXR composition layers extensions

The OpenXR extensions definitions can be found in the Khronos Group [OpenXR Specification](https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html).

<a id="openxr-plugin-extensions"></a>
### Extensions enabled via the Unity OpenXR Plug-in

The following OpenXR extensions are composition layer features available through the Unity OpenXR Plug-in:

<!-- Alphabetical order by extension name -->
| Extension | Usage | Description |
| :-------- | :---- | :---------- |
| [XR_KHR_android_surface_swapchain](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_KHR_android_surface_swapchain) | [Android Surface](xref:xr-layers-android-surface) | Enables Android Surface objects to be used as swapchains for composition layers. This allows hardware-decoded video and other Android Surface content to be composited efficiently without routing through Unity's rendering pipeline. |
| [XR_KHR_composition_layer_color_scale_bias](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_KHR_composition_layer_color_scale_bias) | [Color Bias and Scale](xref:xr-layers-color-bias-scale) | Enables per-layer color scale and bias adjustments to be applied to composition layers, allowing color treatments such as tinting or brightness changes. |
| [XR_KHR_composition_layer_cube](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_KHR_composition_layer_cube) | [Cube layer](xref:xr-layers-add-layer) | Enables cube map composition layers centered at the user's head position. Useful for skyboxes and 360° panoramic images where only the inside faces of the cube are visible. |
| [XR_KHR_composition_layer_cylinder](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_KHR_composition_layer_cylinder) | [Cylinder layer](xref:xr-layers-add-layer) | Enables a curved, rectangular composition layer that follows the surface of a cylinder. Only the inside face of the cylinder is visible. |
| [XR_KHR_composition_layer_equirect](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_KHR_composition_layer_equirect) | [Equirect layer](xref:xr-layers-add-layer) | Enables equirectangular composition layers that map content onto the inside surface of a sphere. This is the original version, used as a fallback when `XR_KHR_composition_layer_equirect2` is unavailable. |
| [XR_KHR_composition_layer_equirect2](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html#XR_KHR_composition_layer_equirect2) | [Equirect layer](xref:xr-layers-add-layer) | Enables improved equirectangular composition layers with better angle handling and more precise control over the visible viewport. This is the preferred version for equirectangular rendering. |
