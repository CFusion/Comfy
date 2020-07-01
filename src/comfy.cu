#include "comfy.cuh"
#include "cugar.hpp"
#include "cugar_matrix.hpp"
#include "bspdef.hpp"

using namespace comfy;

#define METRICS 0
#if METRICS
#define PRINT_METRIC(x) printf("%s %d\r\n", #x, x)
#define METRICINC(x) x++
#else
#define PRINT_METRIC(x)
#define METRICINC(x)
#endif

#define SHARED_STACK 0
#define MAX_DIST 16384.0f
#define RENDER_CUTOFF  MAX_DIST
#define MAX_DRAWS 3
#define BLOCK_DIM (BLOCK_WIDTH * BLOCK_HEIGHT)
#define ESP 0.00001f
#define ESP1 1.00001f

struct camera
{
	vec3 origin;
	mat4 invtransform;
};

struct drawcommand {
	u32 tristart;
	float tmax;
	float tmin;
	i16 node;
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



DEVICE void preprocessstack(i16 * stack, int& idx, i16& currentnode, const CUtexObject nodes) {
	do {
		//const bspnodef& node = nodes[currentnode];
		bspnodef node;
		node.__data[0] = tex1D<int4>((const CUtexObject)nodes, currentnode * 2);

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

DEVICE float clamp(float x, float mi, float ma) {
	return max(min(x, ma), mi);
}
DEVICE float smoothstep(float f0, float f1, float x) {
	float t = clamp((x - f0) / (f1 - f0), 0.0f, 1.0f);
	return t * t * (3.0 - 2.0 * t);
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

DEVICE FORCEINLINE float randfloat(u32 i, u32 p)
{
	i ^= p;
	i ^= i >> 17;
	i ^= i >> 10; i *= 0xb36534e5;
	i ^= i >> 12;
	i ^= i >> 21; i *= 0x93fc4795;
	i ^= 0xdf6e307f;
	i ^= i >> 17; i *= 1 | p >> 18;
	return i * (1.0f / 4294967808.0f);
}

struct Random {
	u32 offset = 0;
	u32 seed;
	DEVICE FORCEINLINE Random(u32 s) : seed(s) {}
	DEVICE FORCEINLINE float next() { return ::randfloat(offset++, seed); }
};

DEVICE FORCEINLINE real linearRand(real min, real max, Random& randState)
{
	return randState.next() * (max - min) + min;
}

DEVICE FORCEINLINE vec3 sphereRand(Random& randState)
{
	real theta = linearRand(0.0f, 6.283185307179586476925286766559f, randState);
	real phi = acosf(linearRand(-1.0f, 1.0f, randState));

	real x = sinf(phi) * cosf(theta);
	real y = sinf(phi) * sinf(theta);
	real z = cosf(phi);

	return vec3(x, y, z);
}

struct kernel {
	struct ray {
		vec3 ro;
		vec3 rd;
	};
	// Constants
	const CUtexObject nodes;
	const CUtexObject leafs;
	const CUtexObject tristex;
	const bspobject* bspobject;
	const camera cam;
	const faceinfo* RESTRICT faceinfos;
	const i16* RESTRICT lightvismatrix;
	const light* RESTRICT lights;
	const int width;
	const int height;
	const int pitch;

	// Volatiles
	Random rnd;
	bool picker;

#if METRICS
	struct {
		int AABBDisregardedLastStage = 0;
		int NodesTraversed = 0;
		int LeafsChecked = 0;
		int TriChecked = 0;
		int TriSkips = 0;
		i64 starttime = 0;
	} metrics;
#endif

	DEVICE FORCEINLINE i32 blockidx() {
		return (threadIdx.x + threadIdx.y * blockDim.x);
	}
	DEVICE FORCEINLINE i32 warpidx() {
		return blockidx() >> LOG_WARP_SIZE;
	}
	DEVICE FORCEINLINE i32 laneidx() {
		return blockidx() & (WARP_SIZE - 1);
	}

	DEVICE void preprocessstackforpos(vec3 pos, i16* stack, int& idx, i16& currentnode, i32 stacktop) {
		idx = stacktop + 1;
		currentnode = 0;
		__syncthreads();
		if (laneidx() == 0) {
			do {
				//const bspnodef& node = nodes[currentnode];
				bspnodef node;
				node.__data[0] = tex1D<int4>((const CUtexObject)nodes, currentnode * 2);

				i16 child0 = node.vischildren[0];
				i16 child1 = node.vischildren[1];

				if (child0 != -1) {
					if (child1 != -1) {

						node.__data[1] = tex1D<int4>((const CUtexObject)nodes, currentnode * 2 + 1);
						float d = dot(vec4(node.plane).xyz(), pos) - node.plane.w;

						if (d > 0.0f) {
							stack[idx++] = child0;
							currentnode = child1;
						}
						else {
							stack[idx++] = child1;
							currentnode = child0;
						}
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

		currentnode = __shfl_sync(__activemask(), currentnode, 0);
		idx = __shfl_sync(__activemask(), idx, 0);
	}

	DEVICE ray raysfromcam() {
		vec2 pixel_center = vec2(PIXEL_X(), PIXEL_Y()) + vec2(0.5f);
		vec2 screen_dimensions = vec2(width, height);

		vec2 uv = pixel_center / screen_dimensions;
		uv.y = 1.0f - uv.y;

		// Sc in clip space
		vec2 sc = uv * 2.0f - vec2(1.0f);

		vec4 rd4 = cam.invtransform * vec4(sc.x, sc.y, 1.0f, 1.0f);

		vec3 camrd = normalize(rd4.xyz() / rd4.w);

		for (int i = 0; i < 3; i++) {
			if (camrd[i] == 0.0f)
				camrd[i] = 0.0000001f;
		}
		vec3 camro = cam.origin / SCALING_FACTORf;

		return { camro, camrd };
	}

	DEVICE vec3 trace(const kernelstate& kstate) {
		SHARED drawcommand drawqueue[MAX_DRAWS][BLOCK_DIM];
		

		// Copy stack out of kernel arguments
		// First Traversal shares a stack amongst all warps
#if SHARED_STACK
		SHARED i16 stack[WARPSPERBLOCK * MAX_STACK];
		const i32 stacktop = warpidx() * MAX_STACK; // Warp stack idx
		stack[stacktop + laneidx() + 0] = 0;
		stack[stacktop + laneidx() + 32] = 0;
		stack[stacktop + laneidx() + 0] = kstate.stack[laneidx() + 0];
		stack[stacktop + laneidx() + 32] = kstate.stack[laneidx() + 32];
		 // Faster then replacing stack with a i16 * ??????
		i16 currentnode = kstate.currentnode;
		i32 idx = stacktop + kstate.idx;
#else
		constexpr i32 stacktop = 0;
		i16 stack[MAX_STACK];
		for (int i = 0; i < MAX_STACK; i++) {
			stack[i] = kstate.stack[i];
		}
		i16 currentnode = kstate.currentnode;
		i32 idx = stacktop + kstate.idx;
#endif
		i32 stackentry = stacktop + 1;

		__syncthreads();
		
		ray r = raysfromcam();
		vec3 ro = r.ro;
		vec3 rd = r.rd;

		vec3 outcol(1.0f);
		vec3 light(0.0f);

		int HITS = 0;
		do {
			i16 drawnodeidx = 0;
			u32 faceidx = COMFY_U32_MAX;
			float dist = COMFY_F32_MAX;
			do {
				i32 drawqueueend = 0;
				{
					vec3 invrd = invert(rd);
					vec3 ood = (ro * SCALING_FACTORf) * invrd;
					do
					{
						bspnodef node;
						if (currentnode < 0) {
							node.__data[0] = tex1D<int4>((const CUtexObject)leafs, ~currentnode * 2);
						}
						else {
							node.__data[0] = tex1D<int4>((const CUtexObject)nodes, currentnode * 2);
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

						vec3 t0 = bmin * invrd - ood;
						vec3 t1 = bmax * invrd - ood;
						const float tminbox = max7(t0.x, t1.x, t0.y, t1.y, t0.z, t1.z, 0.0f);
						const float tmaxbox = min7(t0.x, t1.x, t0.y, t1.y, t0.z, t1.z, MAX_DIST);

						if (tminbox <= tmaxbox && currentnode < 0) { // Current node is a leaf, draw and then POP
							drawcommand& dc = drawqueue[drawqueueend][blockidx()];
							dc.tristart = node.vertidx;
							dc.tmin = tminbox;
							dc.tmax = tmaxbox;
							dc.node = currentnode;

							drawqueueend++;
						}
						bool goChild0 = child0 != INVALID;
						bool goChild1 = child1 != INVALID;

						if (tminbox > tmaxbox || currentnode < 0) {
							goChild0 = false;
							goChild1 = false;
						}

						// Vote what nodes to go to amongst warp
#if 0
						goChild0 = __popc(__ballot_sync(ACTIVE_MASK, goChild0));
						goChild1 = __popc(__ballot_sync(ACTIVE_MASK, goChild1));
#endif

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

						// If all warps have 1 or more draw commands scheduled
						if (__all_sync(ACTIVE_MASK, drawqueueend >= 1))
							break;

						// Or if any warp is out of stack space
						if (__any_sync(ACTIVE_MASK, drawqueueend >= MAX_DRAWS))
							break;

						METRICINC(metrics.NodesTraversed);
					} while (idx >= stackentry);
				}

				for (i32 i = 0; i < drawqueueend && dist > RENDER_CUTOFF; i++)
				{
					u32 vertidx = drawqueue[i][blockidx()].tristart;
					float tmax = drawqueue[i][blockidx()].tmax / SCALING_FACTORf;
					float tmin = drawqueue[i][blockidx()].tmin / SCALING_FACTORf;

					do {
						float4 vControls = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 0);
						//assert(__float_as_uint(v00.x) == 0x80000000);
						u32 nextface = __float_as_uint(vControls.y);
						u32 bspfaceidx = __float_as_uint(vControls.z);

						if (faceidx == bspfaceidx) {
							vertidx = nextface;
							continue;
						}
							

						if (nextface == 0)
							break;

						vertidx -= 2;

						while ((vertidx += 3) < nextface) {
							METRICINC(metrics.TriChecked);
							const float4 v00 = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 0);
							float Oz = v00.w - ro.x * v00.x - ro.y * v00.y - ro.z * v00.z;
							float d = (rd.x * v00.x + rd.y * v00.y + rd.z * v00.z);

							if (d != 0.0f) {
								float invDz = 1.0f / d;
								float t = Oz * invDz;


								//@TODO Solidify this
								if (invDz >= 0.00005f) {
#if METRICS
									metrics.TriSkips += (nextface - vertidx) / 3;
									metrics.TriChecked += (nextface - vertidx) / 3;
#endif
									vertidx = nextface;
									break;
								}

								if (t > tmax || t < tmin || t > dist) {
									if (invDz < -0.0005f) {
#if METRICS
										metrics.TriSkips += (nextface - vertidx) / 3;
										metrics.TriChecked += (nextface - vertidx) / 3;
#endif
										vertidx = nextface;
										break;
									}
									else {
										continue;
									}
								}

								const float4 v11 = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 1);

								float Ox = v11.w + ro.x * v11.x + ro.y * v11.y + ro.z * v11.z;
								float Dx = rd.x * v11.x + rd.y * v11.y + rd.z * v11.z;
								float u = Ox + t * Dx;

								if (u >= -0.00001f && u <= 1.0f) {
									const float4 v22 = tex1Dfetch<float4>((const CUtexObject)tristex, vertidx + 2);

									float Oy = v22.w + ro.x * v22.x + ro.y * v22.y + ro.z * v22.z;
									float Dy = rd.x * v22.x + rd.y * v22.y + rd.z * v22.z;
									float v = Oy + t * Dy;

									if (v >= -0.00001f && u + v <= 1.00001f) { // @EPS
										dist = t;
										faceidx = bspfaceidx;
										drawnodeidx = drawqueue[i][blockidx()].node;
									}
								}
							}
						}
					} while (dist > RENDER_CUTOFF);
				} // End Triangle Intersects
			} while (dist > RENDER_CUTOFF && idx >= stackentry);

			if (faceidx != COMFY_U32_MAX) {
				const faceinfo& fi = faceinfos[faceidx];
				vec3 hitpos = ro + rd * dist;
				vec2 uv;
				uv.x = dot(hitpos, vec3(fi.S.xyz())) * SCALING_FACTORf + fi.S.w;
				uv.y = dot(hitpos, vec3(fi.T.xyz())) * SCALING_FACTORf + fi.T.w;

				uv.x = fmodf(uv.x / float(fi.width), 1.0f);
				uv.y = fmodf(uv.y / float(fi.height), 1.0f);

				vec4 texsample = (vec4)tex2D<float4>((const CUtexObject)fi.cuda_texture, uv.x, uv.y);
				outcol *= texsample.xyz();// / powf(2.0f, float(HITS));

				vec3 normal = fi.N.xyz();
				vec3 lightcol(0.0f);


				//if (fi.emissive.w > 0.0f)
				//	lightcol += fi.emissive.xyz() / 255.0f * fi.emissive.w;

				const i16* RESTRICT lightidx = lightvismatrix + ((~drawnodeidx) * MAX_LIGHT_SOURCE_PER_LEAF);

				while (*lightidx++ != 0) {
					const comfy::light& l = lights[*lightidx];

					const vec3 f_light_origin = vec3(l.origin);
					const vec3 f_light_color = vec3(l.color.xyz());

					if (l.faceidx == faceidx) {
						lightcol += f_light_color;
						continue;
					}

					vec3 delta = hitpos * SCALING_FACTORf - f_light_origin;
					float distance = length(delta);
					delta /= distance;

					float d = -dot(normal, delta);
					
					if (d < 0.0f)
						continue;

					if (distance < 1.0f)
						distance = 1.0f;

					if (l.normal != vec3(0.0f)) {
						float dspot = dot(l.normal, delta);
						float inner = l.cone;
						float outer = l.cone2;

						if (dspot < outer)
							continue;

						if (dspot > 0.0f) {
							d = dspot * d;

							if (dspot < inner) {
								d *= (dspot - outer) / (inner - outer);
							}
						}
						else {
							d = 0.0000001f;
						}						
					}

					lightcol += f_light_color * (d / (distance * distance)) * sqrt(l.area);
				}
				//return hsv(fmodf(float(~drawnodeidx) / 23.0f, 1.0f), 0.5f, 0.5f);
				return outcol * (lightcol * 24.0f);
			}

			{
				faceidx = COMFY_U32_MAX;
				dist = COMFY_F32_MAX;
				// Is this even needed?, the stack is currently in the leaf we hit something in, unwinding from there should result in nearest to furthest
				// This is just made more complex by the drawque mechanism
				// Consider switching to a mechanis where we progress multiple rays per lane.
				preprocessstackforpos(ro, stack, idx, currentnode, stacktop); 
				__syncthreads();

				HITS++;
			}
		} while (HITS < 1);

#if METRICS
		if (picker) {
			PRINT_METRIC(metrics.AABBDisregardedLastStage);
			PRINT_METRIC(metrics.NodesTraversed);
			PRINT_METRIC(metrics.LeafsChecked);
			PRINT_METRIC(metrics.TriChecked);
			PRINT_METRIC(metrics.TriSkips);
			//
			printf("Time %llu \r\n", clock64() - metrics.starttime);
			//
		}
		u64 time = clock64() - metrics.starttime;


		outcol = vec3(float(metrics.TriSkips) / float(metrics.TriChecked));

		if (picker)
			outcol = vec3(1.0f);
#endif


		/*
		if (fi.emissive.w > 0.0f)  
		{
			outcol = outcol * (fi.emissive.xyz() * fi.emissive.w * 0.01f);// (GatherLight(hitpos, normal) * 0.01f);
		}
		else {
			outcol = outcol *(GatherLight(hitpos, normal) * 0.01f);
		}
		*/

		//if (fi.emissive.w > 0.0f)
		//	outcol = vec3(1.0f, 0.0f, 1.0f);

		return outcol;

	}
};


extern "C" __global__ void pathtrace(
	vec4 * RESTRICT surface,
	const bspobject * bspData,
	const CUtexObject nodes,
	const CUtexObject leafs,
	const CUtexObject tristex,
	const faceinfo * RESTRICT faceinfos,
	const i16 * RESTRICT lightvismatrix,
	const light * RESTRICT lights,
	const camera c,
	const int width,
	const int height,
	const int pitch,
	const kernelstate kstate)
{
	Random r(SURFACE_OFFSET() * kstate.frameid + 123686);
	kernel k{ nodes, leafs, tristex,bspData, c, faceinfos, lightvismatrix, lights, width, height, pitch, r };
	k.picker = (kstate.mousex == PIXEL_X()) && (kstate.mousey == PIXEL_Y());

#if METRICS
	k.metrics.starttime = clock64();
#endif

	vec3 outcol = k.trace(kstate);

	//if (fi.emissive.w > 0.0f)
	//	outcol = vec3(1.0f, 0.0f, 1.0f);


	/*
	vec3 lightsampling(0.0f);
	for (int i = 0; i < lc.nlights; i++) {
		const light& l = lc.lights[i];

		vec3 f_light_origin = vec3(l.origin);
		vec3 f_light_color = vec3(l.color.xyz()) / 255.0f * float(l.color.w);

		if (picker)
			printf("l.color.w %f \r\n", float(l.color.w));

		vec3 delta = hitpos_bsp_space - f_light_origin;
		float distance = length(delta);
		delta /= distance;

		float d = dot(normal, delta);
		if (d < 0.0f)
			continue;

		if (distance < 1.0f)
			distance = 1.0f;

		lightsampling += f_light_color * (d / (distance * distance));
	}

	for (int i = 0; i < lc.nspotlights; i++) {
		const spotlight& l = lc.spotlights[i];

		vec3 f_light_origin = vec3(l.origin);
		vec3 f_light_color = vec3(l.color.xyz()) / 255.0f * float(l.color.w);

		vec3 delta = hitpos_bsp_space - f_light_origin;
		float distance = length(delta);
		delta /= distance;

		float d = dot(normal, delta);

		if (d < 0.00f)
			continue;


		vec3 lightangle(0.0f);
		lightangle.z = cugar::sinf(TO_RADIANS(l.pitch));

		float dspot = dot(lightangle, delta);
		float inner = cugar::cosf(TO_RADIANS(l.cone));
		float outer = cugar::cosf(TO_RADIANS(l.cone2));

		float ratio = 0.0000000000001f;
		if (dspot > 0.0f) {
			ratio = dspot * d;

			if (dspot < inner) {
				ratio *= (dspot - outer) / (inner - outer);
			}
		}

		if (dspot < outer)
			continue;

		if (distance < 1.0f)
			distance = 1.0f;

		lightsampling += f_light_color * (ratio / (distance * distance));
	}
	outcol = outcol * (lightsampling * 255.0f);

	
	*/
#if METRICS
	
#endif
	if ((PIXEL_X() >= width) || (PIXEL_Y() >= height))
		return;

	outcol.x = powf(outcol.x, 0.5f);
	outcol.y = powf(outcol.y, 0.5f);
	outcol.z = powf(outcol.z, 0.5f);

#if 1
	outcol = max(min(outcol, vec3(1.0f)), vec3(0.0f));
	surface[SURFACE_OFFSET()] = vec4(outcol, 1.0f);

#else
	int frameidx = kstate.frameid;
	frameidx = 1;
	vec3 out = outcol / float(frameidx);
	vec3 existing = surface[SURFACE_OFFSET()].xyz() / float(frameidx) * float(frameidx - 1);
	surface[SURFACE_OFFSET()] = vec4(out + existing, 1.0f);
#endif
	
	/*
	surface[SURFACE_OFFSET()].x = outcol.z * 255.0f;
	surface[SURFACE_OFFSET()].y = outcol.y * 255.0f;
	surface[SURFACE_OFFSET()].z = outcol.x * 255.0f;
	surface[SURFACE_OFFSET()].w = 255;
	*/
}

