#pragma once
#include "shared.hpp"
#ifdef __INTELLISENSE__
#include <vector_functions.hpp>
#include <vector_types.h>
extern int3 threadIdx;
extern int3 blockIdx;
extern int3 blockDim;
extern int warpSize;
template<typename T = float4>
T tex1Dfetch(CUtexObject tex, int i) { return {}; }
template<typename T>
T tex1D(CUtexObject tex, int i) { return {}; }
template<typename T>
T tex2DLod(CUtexObject tex, float x, float y, float d) { return {}; }
template<typename T>
T tex2D(CUtexObject tex, float x, float y) { return {}; }

extern int __any_sync(int, int);
extern int __all_sync(int, int);
extern int __any(int);
extern int __all(int);
extern u64 clock64();
extern i32 __float_as_int(float);
extern float __int_as_float(i32);
extern uint __float_as_uint(float);
extern uint __popc(uint);
extern uint __ballot_sync(int, int);
extern void __syncthreads();
extern int __activemask();
#endif


#define PIXEL_X() (threadIdx.x + blockIdx.x * blockDim.x)
#define PIXEL_Y() (threadIdx.y + blockIdx.y * blockDim.y)
#define SURFACE_OFFSET() (PIXEL_X() + PIXEL_Y() * (pitch / (sizeof(vec4))))
#define INVALID -1
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define WARPSPERBLOCK (int(BLOCK_DIM) / int(WARP_SIZE))
#define ACTIVE_MASK __activemask()
//#define ACTIVE_MASK 0xFFFFFFFF


namespace comfy
{
#ifdef __INTELLISENSE__
	__device__ __inline__ int   min_min(int a, int b, int c) { return 0; }
	__device__ __inline__ int   min_max(int a, int b, int c) { return 0; }
	__device__ __inline__ int   max_min(int a, int b, int c) { return 0; }
	__device__ __inline__ int   max_max(int a, int b, int c) { return 0; }
	__device__ __inline__ float fmin_fmin(float a, float b, float c) { return 0.0f; }
	__device__ __inline__ float fmin_fmax(float a, float b, float c) { return 0.0f; }
	__device__ __inline__ float fmax_fmin(float a, float b, float c) { return 0.0f; }
	__device__ __inline__ float fmax_fmax(float a, float b, float c) { return 0.0f; }
	__device__ __inline__ int max(int a, int b) { return a; }
	__device__ __inline__ int min(int a, int b) { return a; }
#else
	__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
	__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
	__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
	__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
	__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
	__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
	__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
	__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
#endif

	HOST_DEVICE CUGAR_FORCEINLINE vec3 mix(const vec3& x, const vec3& y, const vec3& a)
	{
		return x * (vec3(1.0f) - a) + y * a;
	}
	HOST_DEVICE CUGAR_FORCEINLINE vec3 mix(const vec3& x, const vec3& y, const float a)
	{
		return x * (vec3(1.0f) - vec3(a)) + y * a;
	}
	//#define USE_VMIN_KEPLER
	__device__ __inline__ float max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
	{
		return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d));
	}
	__device__ __inline__ float min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
	{
		return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d));
	}

	template<typename T>
	DEVICE FORCEINLINE void swap(T& a, T& b)
	{
		T c(a);
		a = b;
		b = c;
	}
	DEVICE FORCEINLINE bool raabb(const vec3 p0, const vec3 p1, const vec3& rayOrigin, const vec3& invRaydir) {
		vec3 t0 = (p0 - rayOrigin) * invRaydir;
		vec3 t1 = (p1 - rayOrigin) * invRaydir;
		vec3 tmin = min(t0, t1), tmax = max(t0, t1);

		return max_comp(tmin) <= min_comp(tmax);
	}
	DEVICE FORCEINLINE bool aabbs(vec3 point, vec3 v0, vec3 v1) {
		if (point.x < v0.x || point.y < v0.y || point.z < v0.z || point.x > v1.x || point.y > v1.y || point.z > v1.z)
			return false;

		return true;
	}
	DEVICE vec3 hue(float H)
	{
		float R = fabsf(H * 6.0f - 3.0f) - 1.0f;
		float G = 2 - fabsf(H * 6.0f - 2.0f);
		float B = 2 - fabsf(H * 6.0f - 4.0f);
		return vec3(cugar::saturate(R), cugar::saturate(G), cugar::saturate(B));
	}
	DEVICE vec3 hsv(float h, float s, float v)
	{
		return vec3(((hue(h) - vec3(1.0f)) * s + vec3(1.0f)) * v);
	}
}


