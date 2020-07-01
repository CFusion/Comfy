#include "comfy.cuh"
#include "cugar.hpp"
#include "cugar_matrix.hpp"
#include "bspdef.hpp"

using namespace comfy;


//#define METRICS
#ifdef METRICS
#define PRINT_METRIC(x) printf("%s %d\r\n", #x, x)
#define METRICINC(x) x++
#else
#define PRINT_METRIC(x)
#define METRICINC(x)
#endif

constexpr float MAX_DIST = 16384.0f;
//constexpr int MAX_STACK = 64;
constexpr int MAX_DRAWS = 3;
constexpr int BLOCK_DIM = BLOCK_WIDTH * BLOCK_HEIGHT;

struct camera
{
	vec3 origin;
	mat4 invtransform;
};

struct bspobject : BSPHEADER {
	DEVICE FORCEINLINE BSPHEADER* Header() const
	{
		return (BSPHEADER*)this;
	}

	DEVICE FORCEINLINE i32 Lumpsize(i32 lumpidx) const
	{
		return Header()->lump[lumpidx].nLength;
	}

	DEVICE FORCEINLINE void* Lump(i32 lumpidx) const
	{
		return ((byte*)this) + Header()->lump[lumpidx].nOffset;
	}
};

DEVICE void preprocessstack(i16* stack, int& idx, i16& currentnode, const CUtexObject nodes) {
	do {
		//const bspnodef& node = nodes[currentnode];
		bspnodef node;
		node.__data = tex1D<int4>((const CUtexObject)nodes, currentnode);

		i16 child0 = node.vischildren[0];
		i16 child1 = node.vischildren[1];

		if (child0 != -1) {
			if (child1 != -1) {
				stack[idx++] = child1;
				currentnode = child0;
			}
			else {
				currentnode = child0;
			}
		}
		else {
			currentnode = child1;
		}
	} while (currentnode >= 0);
}

DEVICE vec3 invert(const vec3& rd)
{
	vec3 invrd;
	float ooeps = exp2f(-80.0f); // Avoid div by zero.
	invrd.x = 1.0f / (fabsf(rd.x) > ooeps ? rd.x : copysignf(ooeps, rd.x));
	invrd.y = 1.0f / (fabsf(rd.y) > ooeps ? rd.y : copysignf(ooeps, rd.y));
	invrd.z = 1.0f / (fabsf(rd.z) > ooeps ? rd.z : copysignf(ooeps, rd.z));

	return invrd;
}

extern "C" __global__ void pathtrace(
	colrgba * __restrict__ surface,
	const bspobject * bspData,
	const CUtexObject nodes,
	const CUtexObject leafs,
	const CUtexObject tristex,
	const faceinfo * faceinfos,
	const camera c,
	const int width,
	const int height,
	const int pitch,
	const kernelstate kstate)
{
#define PIXEL_X() (threadIdx.x + blockIdx.x * blockDim.x)
#define PIXEL_Y() (threadIdx.y + blockIdx.y * blockDim.y)
#define SURFACE_OFFSET() (PIXEL_X() + PIXEL_Y() * (pitch / (sizeof(colrgba))))
#define INVALID -1
#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define WARPSPERBLOCK (int(BLOCK_DIM) / int(WARP_SIZE))
	//#define ACTIVE_MASK __activemask()
#define ACTIVE_MASK 0xFFFFFFFF

	i32 BLOCKIDX = (threadIdx.x + threadIdx.y * blockDim.x);
	i32 WARPIDX = BLOCKIDX >> LOG_WARP_SIZE;
	i32 LANEIDX = BLOCKIDX & (WARP_SIZE - 1);

#ifdef METRICS
	bool picker = (kstate.mousex == PIXEL_X()) && (kstate.mousey == PIXEL_Y());
	struct {
		int AABBDisregardedLastStage = 0;
		int NodesTraversed = 0;
		int LeafsChecked = 0;
		int TriChecked = 0;
		int TriSkips = 0;
		i64 starttime = 0;
	} metrics;
	metrics.starttime = clock64();
#endif
	struct drawcommand {
		u32 tristart;
		float tmax;
		float tmin;
	};

	float	dist = COMFY_F32_MAX;

	SHARED drawcommand drawqueue[MAX_DRAWS][BLOCK_DIM];

	i32 idx;
	u32 faceidx = COMFY_U32_MAX;
	i16 currentnode;

#if 1
	i32 stackentry;
	SHARED i16 stack[WARPSPERBLOCK * MAX_STACK];
	{
		const i32 stacktop = (WARPIDX * MAX_STACK);
		stack[stacktop + LANEIDX + 0] = 0;
		stack[stacktop + LANEIDX + 32] = 0;
		stack[stacktop + LANEIDX + 0] = kstate.stack[LANEIDX + 0];
		stack[stacktop + LANEIDX + 32] = kstate.stack[LANEIDX + 32];
		__syncthreads();

		stackentry = stacktop + 1;
		idx = stacktop + kstate.idx;
		currentnode = kstate.currentnode;
	}
#elif 0
	i16* stack;
	constexpr auto stackentry = 1;
	{
		SHARED i16 _stack[WARPSPERBLOCK * MAX_STACK];
		stack = _stack + (WARPIDX * MAX_STACK);

		const i32 stacktop = 0;// (WARPIDX* MAX_STACK);
		stack[stacktop + LANEIDX + 0] = 0;
		stack[stacktop + LANEIDX + 32] = 0;
		stack[stacktop + LANEIDX + 0] = kstate.stack[LANEIDX + 0];
		stack[stacktop + LANEIDX + 32] = kstate.stack[LANEIDX + 32];

		idx = stacktop + kstate.idx;
		currentnode = kstate.currentnode;
	}
#else
	constexpr auto stackentry = 1;
	i16 stack[MAX_STACK] = { 0 };
	idx = 1;
	currentnode = 0;

	preprocessstack(stack, idx, currentnode, nodes);
#endif

	vec3	ro;
	vec3	rd;
	mat3	raytrans;

	{
		const vec2 pixel_center = vec2(PIXEL_X(), PIXEL_Y()) + vec2(0.5f);
		const vec2 screen_dimensions = vec2(width, height);

		vec2 uv = pixel_center / screen_dimensions;
		uv.y = 1.0f - uv.y;

		// Sc in clip space
		vec2 sc = uv * 2.0f - vec2(1.0f);

		const vec4 rd4 = c.invtransform * vec4(sc.x, sc.y, 1.0f, 1.0f);

		rd = normalize((rd4 / rd4.w).xyz());
		ro = c.origin;

	}

	{
		vec3 invrd = invert(rd);
		int kx, ky, kz;
		vec3 S;
		vec3 rd2 = abs(rd);
		if (rd2.y > rd2.x) {
			if (rd2.z > rd2.y)
				kz = 2;
			else
				kz = 1;
		}
		else
			kz = 0;

		kx = kz + 1; if (kx == 3) kx = 0;
		ky = kx + 1; if (ky == 3) ky = 0;

		if (rd[kz] < 0.0f)
			swap(kx, ky);

		S.x = rd[kx] * invrd[kz];
		S.y = rd[ky] * invrd[kz];
		S.z = 1.0f * invrd[kz];

		vec3 rtscale[3] = {
			{ 0.0f, 0.0f, 0.0f },
			{ 0.0f, 0.0f, 0.0f },
			{ 0.0f, 0.0f, 0.0f }
		};

		rtscale[0][kx] = 1.0f;
		rtscale[1][ky] = 1.0f;
		rtscale[2][kz] = 1.0f;

		vec3 rtshear[3] = {
			{ 1.0f, 0.0f, -rd[kx] * invrd[kz] },
			{ 0.0f, 1.0f, -rd[ky] * invrd[kz] },
			{ 0.0f, 0.0f, 1.0f * invrd[kz] }
		};

		mat3 scale_transform(rtscale);
		mat3 shear_transform(rtshear);

		raytrans = shear_transform * scale_transform;
	}


	do {
		i32 drawqueueend = 0;
		{
			// Disabled because of register preasure
#ifdef FASTER_TRAVERSAL
			vec3 invrd = invert(rd);
			vec3 ood = ro * invrd;
#endif
			do
			{
				bspnodef node;
				if (currentnode < 0) {
					node.__data = tex1D<int4>((const CUtexObject)leafs, ~currentnode);
				}
				else {
					node.__data = tex1D<int4>((const CUtexObject)nodes, currentnode);
				}
				i16 child0 = node.vischildren[0];
				i16 child1 = node.vischildren[1];


				vec3 bmin;
				bmin.x = node.bb[0];
				bmin.y = node.bb[1];
				bmin.z = node.bb[2];

				vec3 bmax;
				bmax.x = node.bb[3];
				bmax.y = node.bb[4];
				bmax.z = node.bb[5];

#ifdef FASTER_TRAVERSAL
				vec3 t0 = bmin * invrd - ood;
				vec3 t1 = bmax * invrd - ood;
#else							
				vec3 t0 = (bmin - ro) / rd;
				vec3 t1 = (bmax - ro) / rd;
#endif		
				const float tminbox = max7(t0.x, t1.x, t0.y, t1.y, t0.z, t1.z, 0.0f);
				const float tmaxbox = min7(t0.x, t1.x, t0.y, t1.y, t0.z, t1.z, MAX_DIST);

				if ((tminbox <= tmaxbox) && (currentnode < 0)) { // Current node is a leaf, draw and then POP
					drawcommand& dc = drawqueue[drawqueueend][BLOCKIDX];
					dc.tristart = node.vertidx;
					dc.tmax = tmaxbox;
					dc.tmin = tminbox;

					drawqueueend++;
				}

				bool goChild0 = child0 != -1;
				bool goChild1 = child1 != -1;

				if (tminbox > tmaxbox || currentnode < 0) {
					goChild0 = false;
					goChild1 = false;
				}

				goChild0 = __popc(__ballot_sync(ACTIVE_MASK, goChild0));
				goChild1 = __popc(__ballot_sync(ACTIVE_MASK, goChild1));

				if (!goChild0) {
					if (!goChild1) {
						currentnode = stack[--idx]; // Don't need to sync threads here they'll sync multiple times in this loop
					}
					else {
						currentnode = child1;
					}
				}
				else {
					if (goChild1) {
						stack[idx++] = child1;
						currentnode = child0;
					}
					else {
						currentnode = child0;
					}
				}

				if (__all_sync(ACTIVE_MASK, drawqueueend >= 1))
					break;
				if (__any_sync(ACTIVE_MASK, drawqueueend >= MAX_DRAWS))
					break;

				METRICINC(metrics.NodesTraversed);
			} while (idx >= stackentry);
		}

		for (i32 i = 0; i < drawqueueend && dist > MAX_DIST; i++)
		{
			u32 vertidx = drawqueue[i][BLOCKIDX].tristart;
			float tmax = drawqueue[i][BLOCKIDX].tmax;
			float tmin = drawqueue[i][BLOCKIDX].tmin;

			do {
				const float4 v00 = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 0);
				//assert(__float_as_uint(v00.x) == 0x80000000);
				u32 nextface = __float_as_uint(v00.y);
				u32 bspfaceidx = __float_as_uint(v00.z);

				if (nextface == 0)
					break;

				vertidx++;

				do {
					vec3 v00 = ((const vec4)tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 0)).xyz();
					vec3 v11 = ((const vec4)tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 1)).xyz();
					v00 = raytrans * (v00 - ro);
					v11 = raytrans * (v11 - ro);

					float W = v11.x * v00.y - v11.y * v00.x;
					if (W >= 0.0f) {
						vec3 v22 = ((const vec4)tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 2)).xyz();
						v22 = raytrans * (v22 - ro);
						float U = v22.x * v11.y - v22.y * v11.x;
						if (U >= 0.0f) {
							float V = v00.x * v22.y - v00.y * v22.x;
							if (V >= 0.0f) {
								float T = U * v00.z + V * v11.z + W * v22.z;
								T = T * (1.0f / (U + V + W));

								if (T > tmax || T < tmin) {
									vertidx = nextface;
									break;
								}

								if (T <= dist) {
									dist = T;
									faceidx = bspfaceidx;
								}
							}
						}
					}

					vertidx += 3;
#if 0
					const float4 v00 = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 0);
					const float4 v11 = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 1);
					const float4 v22 = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 2);

					float Oz = v00.w - ro.x * v00.x - ro.y * v00.y - ro.z * v00.z;
					float d = (rd.x * v00.x + rd.y * v00.y + rd.z * v00.z);
					float invDz = 1.0f / d;
					float t = Oz * invDz;


					if (t > tmax || t < tmin) {
						vertidx = nextface;
						break;
					}
#define ESP 0.00001f
#define ESP1 1.00001f
					if (t > tmin && t < dist && t < tmax && d <= 0.0f) {
						float Ox = v11.w + ro.x * v11.x + ro.y * v11.y + ro.z * v11.z;
						float Dx = rd.x * v11.x + rd.y * v11.y + rd.z * v11.z;
						float u = Ox + t * Dx;

						if (u >= 0.0f && u <= 1.0f) {

							float Oy = v22.w + ro.x * v22.x + ro.y * v22.y + ro.z * v22.z;
							float Dy = rd.x * v22.x + rd.y * v22.y + rd.z * v22.z;
							float v = Oy + t * Dy;

							if (v >= -ESP && u + v <= ESP1) { // EPSILON
								dist = t;
								trisidx = vertidx;
								faceidx = bspfaceidx;
							}
						}
					}
					vertidx += 3;
#endif
				} while (vertidx < nextface);
			} while (dist > MAX_DIST);
		} // End Triangle Intersects
	} while (dist > MAX_DIST && idx >= stackentry);

	if (faceidx == COMFY_U32_MAX) {
		surface[SURFACE_OFFSET()] = { 255, 255, 255, 255 };
		return;
	}

	vec3 outcol;
	//auto bsp_faces = (const BSPFACE*)bspData->Lump(LUMP_FACES);
	//auto bsp_texinfos = (const BSPTEXTUREINFO*)bspData->Lump(LUMP_TEXINFO);
	//const BSPFACE& face = bsp_faces[faceidx];
	//const BSPTEXTUREINFO& texinfo = bsp_texinfos[face.iTextureInfo];
	//const loadedtexture tex = bsptex[texinfo.iMiptex];

	faceinfo& fi = faceinfos[faceidx];

	vec3 hitpos = ro + rd * dist;

	vec2 uv;
	uv.x = dot(hitpos, vec3(texinfo.vS)) + texinfo.fSShift;
	uv.y = dot(hitpos, vec3(texinfo.vT)) + texinfo.fTShift;

	uv.x = fmodf(uv.x / float(tex.width), 1.0f);
	uv.y = fmodf(uv.y / float(tex.height), 1.0f);

	vec4 texsample = (vec4)tex2D<float4>((const CUtexObject)tex.cuda_texture, uv.x, uv.y);
	outcol = texsample.xyz();

#ifdef METRICS
	if (picker) {
		PRINT_METRIC(metrics.AABBDisregardedLastStage);
		PRINT_METRIC(metrics.NodesTraversed);
		PRINT_METRIC(metrics.LeafsChecked);
		PRINT_METRIC(metrics.TriChecked);
		PRINT_METRIC(metrics.TriSkips);

		printf("Time %llu \r\n", clock64() - metrics.starttime);

		printf("Dist %f \r\n", dist);
	}
	u64 time = clock64() - metrics.starttime;
	outcol = vec3(float(time) / 1000000.0f);

	if (picker)
		outcol = vec3(1.0f);

#endif

	//outcol = hsv(fmodf(float(faceidx) / 23.0f, 1.0f), 0.5f, 0.5f);

	outcol = max(min(outcol, vec3(1.0f)), vec3(0.0f));

	if ((PIXEL_X() >= width) || (PIXEL_Y() >= height))
		return;

	surface[SURFACE_OFFSET()].x = outcol.z * 255.0f;
	surface[SURFACE_OFFSET()].y = outcol.y * 255.0f;
	surface[SURFACE_OFFSET()].z = outcol.x * 255.0f;
	surface[SURFACE_OFFSET()].w = 255;
}

