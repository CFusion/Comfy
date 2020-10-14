# Raytracer for HL1 BSP Maps using Cuda

Runs in 60fps 720p on an MX250, preprocesses traingles, and uses a box intersection approach instead of slicing rays at bsp planes.

![comfy](https://cld.moe/f/Comfy_yavNdOvL17.png "hallway in a hl1 map")
![comfy](https://cld.moe/f/Comfy_Ysf61Zig2B.png "crossfire")

Required:
- Cuda 12
- Imgui
- EASTL
- Nvidia Cugar's Vector math headers

References:
- Realtime Ray Tracing of Dynamic Scenes on an FPGA Chip - Jörg Schmittler, Sven Woop, Daniel Wagner, Wolfgang J. Paul, and Philipp Slusallek
- Watertight Ray/Triangle Intersection - Sven Woop, Carsten Benthin, Ingo Wald
- Understanding the Efficiency of Ray Traversal on GPUs - Timo Aila, Samuli Laine
- Understanding the Efficiency of Ray Traversal on GPUs - Kepler and Fermi Addendum - Timo Aila, Samuli Laine, Tero Karras
