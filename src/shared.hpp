#pragma once
#define NOMINMAX
#define LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#undef MIN
#undef MAX
#undef min
#undef max

#pragma warning(disable:4238)
#pragma warning(disable:4100) // unreferenced formal parameter
#pragma warning(disable:4068)
#pragma warning(disable:4505) //Comdat Folding
#pragma warning(disable:4324) // Structure was padded

#include <vector_types.h>
#include <vector_functions.h>
#include "cugar.hpp"

#ifdef __CUDA_ARCH__
#define SHARED __shared__
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#define FORCEINLINE __inline__
#define RESTRICT __restrict__
#else
#define SHARED
#define HOST_DEVICE
#define DEVICE
#define FORCEINLINE __forceinline
#define RESTRICT
#endif

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 8
#define MAX_STACK (BLOCK_WIDTH * BLOCK_HEIGHT)
#define BB_PADDING 1
#define CULL_AABB
#define SCALING_FACTOR 16384
#define SCALING_FACTORf SCALING_FACTOR.0f
#define SCALING_FACTORd SCALING_FACTOR.0
#define MAX_LIGHT_SOURCE_PER_LEAF 256
//#define CULL_EDGES
#define TO_RADIANS(x) (x / 180.0f * COMFY_PIf)



// Types
namespace comfy
{
	typedef unsigned char		byte, u8;
	typedef unsigned short		u16;
	typedef unsigned int		u32;
	typedef unsigned long long	u64;

	typedef signed char			i8;
	typedef short				i16;
	typedef int					i32;
	typedef long long			i64;

	typedef unsigned int		uint;
	typedef float				real;

	typedef unsigned long long	CUtexObject;
	typedef unsigned long long	CUHandle;
	typedef void*				CUHandle2;

	#define COMFY_U32_MAX          (0xFFFFFFFFu)
	#define COMFY_S32_MIN          (~0x7FFFFFFF)
	#define COMFY_S32_MAX          (0x7FFFFFFF)
	#define COMFY_U64_MAX          ((u64)(i64)-1)
	#define COMFY_S64_MIN          ((i64)-1 << 63)
	#define COMFY_S64_MAX          (~((i64)-1 << 63))
	#define COMFY_F32_MIN          (1.175494351e-38f)
	#define COMFY_F32_MAX          (3.402823466e+38f)
	#define COMFY_F64_MIN          (2.2250738585072014e-308)
	#define COMFY_F64_MAX          (1.7976931348623158e+308)
	#define COMFY_PIf              (3.14159265358979323846f)
	#define COMFY_PI               (3.14159265358979323846)
}

namespace comfy
{
	typedef cugar::Vector3f vec3;
	typedef cugar::Vector3u vec3u;
	typedef cugar::Vector3i vec3i;
	typedef cugar::Vector4i vec4i;
	typedef cugar::Vector4f vec4;
	typedef cugar::Vector2f vec2;
	typedef cugar::Vector2i vec2i;
	typedef cugar::Vector2i vec2i;

	typedef cugar::Vector<byte, 3> colrgb;
	typedef cugar::Vector<byte, 4> colrgba;
}

struct BSPNODE;
struct BSPLEAF;
// Internal Structures
namespace comfy
{
	typedef cugar::Vector<i16, 3> bspvec3i;
	typedef cugar::Vector<float, 3> bspvec3f;

	struct alignas(16) faceinfo {
		vec4	S;
		vec4	T;
		vec4	N;
		vec4	emissive;
		u64		cuda_texture;
		u32		width;
		u32		height;
		float	worldarea;
	};

	struct alignas(16) light {
		vec3 origin;
		vec3 color;
		vec3 normal;
		i32 faceidx;
		i32 isface;
		float area;
		float pitch;
		float cone;
		float cone2;
	};

	struct lightcollection {
		i32 nlights;
		i32 nspotlights;
		light		lights		[100];
	};

	struct alignas(16) kernelstate {
		i16 stack[MAX_STACK];
		i32 idx;
		i16 currentnode;
		i16 __pad0;
		i32 mousex;
		i32 mousey;
		i32 frameid;
	};

	struct alignas(16) loadedtexture {
		u64 cuda_array;
		u64 cuda_texture;
		colrgba * texdata[4];
		u32 width;
		u32 height;
		char name[16];
	};

	struct bsptexture {
		char name[16];
		u32 width;
		u32 height;
		u32 offsets[4];
	};

	struct bspface {
		vec4 s;
		vec4 t;
		u64 texture;
		u32 width;
		u32 height;
	};

	struct alignas(16) bspnodef
	{
		union {
			int4 __data[2];
			struct {
				union {
					struct {
						short3 min;
						short3 max;
					};
					i16 bb[6];
				};
				union {
					i16 vischildren[2];
					u32 vertidx;
				};
				float4 plane;
			};
		};
	};

	struct alignas(16) bspnode
	{
		union {
			struct {
				bspvec3i min;
				bspvec3i max;
			};
			i16 bb[6];
		};
		union {
			i16 vischildren[2];
		};
		vec4 plane; // Internal, xyz, w is dist

		union {
			u32 planeidx; 		// Node
			i32 contenttype; 	// Leaf
		};
		union {
			i16 children[2]; 	// Node
			i32 pvsidx; 		// Leaf
		};

		u16 faceidx; // Note: this is a marksurface IDX for leafs and normal faceidx for nodes
		u16 nfaces;

		// Internal Variables
		i16 parent;// Internal
		u16 padding;
		u32 visframe;// Internal
		i32 vertidx; // Internal
		i32 vertend;// Internal
		vec4 visplane;

		inline bool isleaf()
		{
			return contenttype < 0;
		}
		static bspnode fromBSPNODE(const BSPNODE& n);
		static bspnode fromBSPLEAF(const BSPLEAF& n);
	};
}
