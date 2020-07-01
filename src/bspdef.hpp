#pragma once
#include <stdint.h>
#include "cugar.hpp"

#define LUMP_ENTITIES	  0
#define LUMP_PLANES		1
#define LUMP_TEXTURES	  2
#define LUMP_VERTICES	  3
#define LUMP_VISIBILITY	4
#define LUMP_NODES		 5
#define LUMP_TEXINFO	   6
#define LUMP_FACES		 7
#define LUMP_LIGHTING	  8
#define LUMP_CLIPNODES	 9
#define LUMP_LEAVES	   10
#define LUMP_MARKSURFACES 11
#define LUMP_EDGES		12
#define LUMP_SURFEDGES	13
#define LUMP_MODELS	   14
#define HEADER_LUMPS	  15

#define MAX_MAP_HULLS		4

#define MAX_MAP_MODELS	   400
#define MAX_MAP_BRUSHES	  4096
#define MAX_MAP_ENTITIES	 1024
#define MAX_MAP_ENTSTRING	(128*1024)

#define MAX_MAP_PLANES	   32767
#define MAX_MAP_NODES		32767
#define MAX_MAP_CLIPNODES	32767
#define MAX_MAP_LEAFS		8192
#define MAX_MAP_VERTS		65535
#define MAX_MAP_FACES		65535
#define MAX_MAP_MARKSURFACES 65535
#define MAX_MAP_TEXINFO	  8192
#define MAX_MAP_EDGES		256000
#define MAX_MAP_SURFEDGES	512000
#define MAX_MAP_TEXTURES	 512
#define MAX_MAP_MIPTEX	   0x200000
#define MAX_MAP_LIGHTING	 0x200000
#define MAX_MAP_VISIBILITY   0x200000

#define MAX_MAP_PORTALS	 65536

#define MAX_KEY	 32
#define MAX_VALUE   1024

#define PLANE_X 0	 // Plane is perpendicular to given axis
#define PLANE_Y 1
#define PLANE_Z 2
#define PLANE_ANYX 3  // Non-axial plane is snapped to the nearest
#define PLANE_ANYY 4
#define PLANE_ANYZ 5

#define CONTENTS_EMPTY		-1
#define CONTENTS_SOLID		-2
#define CONTENTS_WATER		-3
#define CONTENTS_SLIME		-4
#define CONTENTS_LAVA		 -5
#define CONTENTS_SKY		  -6
#define CONTENTS_ORIGIN	   -7
#define CONTENTS_CLIP		 -8
#define CONTENTS_CURRENT_0	-9
#define CONTENTS_CURRENT_90   -10
#define CONTENTS_CURRENT_180  -11
#define CONTENTS_CURRENT_270  -12
#define CONTENTS_CURRENT_UP   -13
#define CONTENTS_CURRENT_DOWN -14
#define CONTENTS_TRANSLUCENT  -15

#ifdef __CUDA_ARCH__
typedef short3 bspvec3i;
typedef float3 bspvec3f;
#else
typedef cugar::Vector<int16_t, 3> bspvec3i;
typedef cugar::Vector<float, 3> bspvec3f;
#endif


typedef struct _BSPLUMP
{
	int32_t nOffset; // File offset to data
	int32_t nLength; // Length of data
} BSPLUMP;

typedef struct _BSPHEADER
{
	int32_t version;		   // Must be 30 for a valid HL BSP file
	BSPLUMP lump[HEADER_LUMPS]; // Stores the directory of lumps
} BSPHEADER;


typedef struct _BSPPLANE
{
	bspvec3f normal; // The planes normal vector
	float dist;	  // Plane equation is: vNormal * X = fDist
	int32_t type;	// Plane type, see #defines
} BSPPLANE;

typedef struct _BSPFACE
{
	uint16_t iPlane;		  // Plane the face is parallel to
	uint16_t nPlaneSide;	  // Set if different normals orientation
	uint32_t iFirstEdge;	  // Index of the first surfedge
	uint16_t nEdges;		  // Number of consecutive surfedges
	uint16_t iTextureInfo;	// Index of the texture info structure
	uint8_t nStyles[4];	   // Specify lighting styles
	uint32_t nLightmapOffset; // Offsets into the raw lightmap data
} BSPFACE;



struct BSPLEAF
{
	int32_t nContents;						 // Contents enumeration
	int32_t nVisOffset;						// Offset into the visibility lump
	bspvec3i min;
	bspvec3i max;
	uint16_t iFirstMarkSurface, nMarkSurfaces; // Index and count into marksurfaces array
	uint8_t nAmbientLevels[4];				 // Ambient sound levels
};


struct BSPNODE
{
	uint32_t iPlane;			// Index into Planes lump
	int16_t children[2];
	bspvec3i min;
	bspvec3i max;				// Defines bounding box
	uint16_t firstFace, nFaces; // Index and count into Faces
};


typedef struct _BSPEDGE
{
	uint16_t iVertex[2]; // Indices into vertex array
} BSPEDGE;
typedef int32_t BSPSURFEDGE;
typedef bspvec3f BSPVERTEX;


typedef struct _BSPTEXTUREINFO
{
	bspvec3f vS;
	float fSShift;	// Texture shift in s direction
	bspvec3f vT;
	float fTShift;	// Texture shift in t direction
	uint32_t iMiptex; // Index into textures array
	uint32_t nFlags;  // Texture flags, seem to always be 0
} BSPTEXTUREINFO;

typedef struct _BSPCLIPNODE
{
	int32_t iPlane;	   // Index into planes
	int16_t iChildren[2]; // negative numbers are contents
} BSPCLIPNODE;

#define MAX_MAP_HULLS 4

typedef struct _BSPMODEL
{
	bspvec3f min;
	bspvec3f max;
	bspvec3f vOrigin;
	int32_t iHeadnodes[MAX_MAP_HULLS];
	int32_t nVisLeafs;
	int32_t faceidx;
	int32_t nfaces;
} BSPMODEL;

typedef uint16_t BSPMARKSURFACE;

#define LUMP_COUNT(_lump_name) (bspHeader->lump[LUMP_##_lump_name##S].nLength / sizeof(BSP##_lump_name))

// Internal Structures
namespace comfy
{
	inline bspnode bspnode::fromBSPNODE(const BSPNODE& n)
	{
		bspnode internalnode{ 0 };

		internalnode.planeidx = n.iPlane;
		internalnode.children[0] = n.children[0];
		internalnode.children[1] = n.children[1];
		internalnode.min = n.min;
		internalnode.max = n.max;
		internalnode.faceidx = n.firstFace;
		internalnode.nfaces = n.nFaces;
		internalnode.vischildren[0] = internalnode.children[0];
		internalnode.vischildren[1] = internalnode.children[1];
		return internalnode;
	}
	inline bspnode bspnode::fromBSPLEAF(const BSPLEAF& n)
	{
		bspnode internalnode{ 0 };

		internalnode.min = n.min;
		internalnode.max = n.max;
		internalnode.contenttype = n.nContents;
		internalnode.pvsidx = n.nVisOffset;
		internalnode.faceidx = n.iFirstMarkSurface;
		internalnode.nfaces = n.nMarkSurfaces;

		return internalnode;
	}
}



