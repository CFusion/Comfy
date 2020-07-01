#include "shared.hpp"
#include "host.hpp"
#include "platform_utlity.hpp"
#include "bsp.hpp"
#include "bspdef.hpp"
#include "cugar_matrix.hpp"

#include <Windows.h>
#include <sstream>
#include <EASTL/map.h>
#include <EASTL/string_map.h>


namespace comfy {
	eastl::map<string, vec4> emissives = {
		/*
		{ "+0~GENERIC65" 	,vec4(255.0f, 255.0f, 255.0f, 750.0f) },
		{ "+0~GENERIC85"	,vec4(110.0f, 140.0f, 235.0f, 20000.0f) },
		{ "+0~GENERIC86"	,vec4(255.0f, 230.0f, 125.0f, 10000.0f) },
		{ "+0~GENERIC86B"	,vec4(60.0f, 220.0f, 170.0f, 20000.0f) },
		{ "+0~GENERIC86R"	,vec4(128.0f, 0.0f, 0.0f, 60000.0f) },
		{ "GENERIC87A"		,vec4(100.0f,  255.0f, 100.0f, 1000.0f) },
		{ "GENERIC88A"		,vec4(255.0f, 100.0f, 100.0f, 1000.0f) },
		{ "GENERIC89A"		,vec4(40.0f, 40.0f, 130.0f, 1000.0f) },
		{ "GENERIC90A"		,vec4(200.0f, 255.0f, 200.0f, 1000.0f) },
		{ "GENERIC105"		,vec4(255.0f, 100.0f, 100.0f, 1000.0f) },
		{ "GENERIC106"		,vec4(120.0f, 120.0f, 100.0f, 1000.0f) },
		{ "GENERIC107"		,vec4(180.0f, 50.0f, 180.0f, 1000.0f) },
		{ "GEN_VEND1"		,vec4(50.0f, 180.0f, 50.0f, 1000.0f) },
		{ "EMERGLIGHT"		,vec4(255.0f, 200.0f, 100.0f, 50000.0f) },
		{ "+0~FIFTS_LGHT01" ,vec4(160.0f, 170.0f, 220.0f, 4000.0f) },
		{ "+0~FIFTIES_LGT2" ,vec4(160.0f, 170.0f, 220.0f, 5000.0f) },
		{ "+0~FIFTS_LGHT4"	,vec4(160.0f, 170.0f, 220.0f, 4000.0f) },
		{ "+0~LIGHT1"		,vec4(40.0f, 60.0f, 150.0f, 3000.0f) },
		{ "+0~LIGHT3A"		,vec4(180.0f, 180.0f, 230.0f, 10000.0f) },
		{ "+0~LIGHT4A"		,vec4(200.0f, 190.0f, 130.0f, 11000.0f) },
		{ "+0~LIGHT5A"		,vec4(80.0f, 150.0f, 200.0f, 10000.0f) },
		{ "+0~LIGHT6A"		,vec4(150.0f, 5.0f, 5.0f, 25000.0f) },
		{ "+0~TNNL_LGT1"	,vec4(240.0f, 230.0f, 100.0f, 10000.0f) },
		{ "+0~TNNL_LGT2"	,vec4(190.0f, 255.0f, 255.0f, 12000.0f) },
		{ "+0~TNNL_LGT3"	,vec4(150.0f, 150.0f, 210.0f, 17000.0f) },
		{ "+0~TNNL_LGT4"	,vec4(170.0f, 90.0f, 40.0f, 10000.0f) },
		{ "+0LAB1_W6D"		,vec4(165.0f, 230.0f, 255.0f, 4000.0f) },
		{ "+0LAB1_W6"		,vec4(150.0f, 160.0f, 210.0f, 8800.0f) },
		{ "+0LAB1_W7"		,vec4(245.0f, 240.0f, 210.0f, 4000.0f) },
		{ "SKKYLITE"		,vec4(165.0f, 230.0f, 255.0f, 1000.0f) },
		{ "+0~DRKMTLS1"		,vec4(205.0f, 0.0f, 0.0f, 6000.0f) },
		{ "+0~DRKMTLGT1"	,vec4(200.0f, 200.0f, 180.0f, 6000.0f) },
		{ "+0~DRKMTLS2"		,vec4(150.0f, 120.0f, 20.0f, 30000.0f) },
		{ "+0~DRKMTLS2C"	,vec4(255.0f, 200.0f, 100.0f, 50000.0f) },
		{ "+0DRKMTL_SCRN"	,vec4(60.0f, 80.0f, 255.0f, 10000.0f) },
		{ "~LAB_CRT9A"		,vec4(225.0f, 150.0f, 150.0f, 100.0f) },
		{ "~LAB_CRT9B"		,vec4(100.0f, 100.0f, 255.0f, 100.0f) },
		{ "~LAB_CRT9C"		,vec4(100.0f, 200.0f, 150.0f, 100.0f) },
		{ "~LIGHT3A"		,vec4(190.0f, 20.0f, 20.0f, 3000.0f) },
		{ "~LIGHT3B"		,vec4(155.0f, 155.0f, 235.0f, 2000.0f) },
		{ "~LIGHT3C"		,vec4(220.0f, 210.0f, 150.0f, 2500.0f) },
		{ "~LIGHT3E"		,vec4(90.0f, 190.0f, 140.0f, 6000.0f) },
		{ "C1A3C_MAP"		,vec4(100.0f, 100.0f, 255.0f, 100.0f) },
		{ "FIFTIES_MON1B"	,vec4(100.0f, 100.0f, 180.0f, 30.0f) },
		{ "+0~LAB_CRT8"		,vec4(50.0f, 50.0f, 255.0f, 100.0f) },
		{ "ELEV2_CIEL"		,vec4(255.0f, 200.0f, 100.0f, 800.0f) },
		{ "YELLOW"			,vec4(255.0f, 200.0f, 100.0f, 2000.0f) },
		{ "RED"				,vec4(255.0f, 0.0f, 0.0f,1000.0f) }
		*/
		{"+0~WHITE",					vec4(255.0f, 255.0f, 255.0f, 100.0f)},
		{"+0~GENERIC65",			 	vec4(255.0f ,255.0f ,255.0f ,750.0f)},
		{"+0~GENERIC85",				vec4(110.0f ,140.0f ,235.0f ,20000.0f)},
		{"+0~GENERIC86",				vec4(255.0f ,230.0f ,125.0f ,10000.0f)},
		{"+0~GENERIC86B",				vec4(60.0f ,220.0f ,170.0f ,20000.0f)},
		{"+0~GENERIC86R",				vec4(128.0f ,0.0f ,0.0f ,60000.0f)},
		{"GENERIC87A",					vec4(100.0f  ,255.0f ,100.0f ,1000.0f)},
		{"GENERIC88A",					vec4(255.0f ,100.0f ,100.0f ,1000.0f)},
		{"GENERIC89A",					vec4(40.0f ,40.0f ,130.0f ,1000.0f)},
		{"GENERIC90A",					vec4(200.0f ,255.0f ,200.0f ,1000.0f)},
		{"GENERIC105",					vec4(255.0f ,100.0f ,100.0f ,1000.0f)},
		{"GENERIC106",					vec4(120.0f ,120.0f ,100.0f ,1000.0f)},
		{"GENERIC107",					vec4(180.0f ,50.0f ,180.0f ,1000.0f)},
		{"GEN_VEND1",					vec4(50.0f ,180.0f ,50.0f ,1000.0f)},
		{"EMERGLIGHT",					vec4(255.0f ,200.0f ,100.0f ,50000.0f)},
		{"+0~FIFTS_LGHT01",			 	vec4(160.0f ,170.0f ,220.0f ,4000.0f)},
		{"+0~FIFTIES_LGT2",			 	vec4(160.0f ,170.0f ,220.0f ,5000.0f)},
		{"+0~FIFTS_LGHT4",				vec4(160.0f ,170.0f ,220.0f ,4000.0f)},
		{"+0~FIFTS_LGHT06",				vec4(245.0f ,250.0f ,255.0f ,1000.0f)},
		{"+0~LIGHT1",					vec4(40.0f ,60.0f ,150.0f ,3000.0f)},
		{"+0~LIGHT2A",					vec4(200.0f ,200.0f ,170.0f ,10000.0f)},
		{"+0~LIGHT3A",					vec4(180.0f ,180.0f ,230.0f ,10000.0f)},
		{"+0~LIGHT4A",					vec4(200.0f ,190.0f ,130.0f ,11000.0f)},
		{"+0~LIGHT5A",					vec4(80.0f ,150.0f ,200.0f ,10000.0f)},
		{"+0~LIGHT6A",					vec4(150.0f ,5.0f ,5.0f ,25000.0f)},
		{"+0~TNNL_LGT1",				vec4(240.0f ,230.0f ,100.0f ,10000.0f)},
		{"+0~TNNL_LGT2",				vec4(190.0f ,255.0f ,255.0f ,12000.0f)},
		{"+0~TNNL_LGT3",				vec4(150.0f ,150.0f ,210.0f ,17000.0f)},
		{"+0~TNNL_LGT4",				vec4(170.0f ,90.0f ,40.0f ,10000.0f)},
		{"+0LAB1_W6D",					vec4(165.0f ,230.0f ,255.0f ,4000.0f)},
		{"+0LAB1_W6",					vec4(150.0f ,160.0f ,210.0f ,8800.0f)},
		{"+0LAB1_W7",					vec4(245.0f ,240.0f ,210.0f ,5500.0f)},
		{"SKKYLITE",					vec4(165.0f ,230.0f ,255.0f ,10000.0f)},
		{"+0~DRKMTLS1",					vec4(205.0f ,0.0f ,0.0f ,6000.0f)},
		{"+0~DRKMTLGT1",				vec4(200.0f ,200.0f ,180.0f ,6000.0f)},
		{"+0~DRKMTLS2",					vec4(150.0f ,120.0f ,20.0f ,30000.0f)},
		{"+0~DRKMTLS2C",				vec4(255.0f ,200.0f ,100.0f ,50000.0f)},
		{"+0DRKMTL_SCRN",				vec4(60.0f ,80.0f ,255.0f ,10000.0f)},
		{"~LAB_CRT9A",					vec4(225.0f ,150.0f ,150.0f ,100.0f)},
		{"~LAB_CRT9B",					vec4(100.0f ,100.0f ,255.0f ,100.0f)},
		{"~LAB_CRT9C",					vec4(100.0f ,200.0f ,150.0f ,100.0f)},
		{"~LIGHT3A",					vec4(190.0f ,20.0f ,20.0f ,3000.0f)},
		{"~LIGHT3B",					vec4(155.0f ,155.0f ,235.0f ,2000.0f)},
		{"~LIGHT3C",					vec4(220.0f ,210.0f ,150.0f ,2500.0f)},
		{"~LIGHT3E",					vec4(90.0f ,190.0f ,140.0f ,6000.0f)},
		{"C1A3C_MAP",					vec4(100.0f ,100.0f ,255.0f ,100.0f)},
		{"FIFTIES_MON1B",				vec4(100.0f ,100.0f ,180.0f ,30.0f)},
		{"+0~LAB_CRT8",					vec4(50.0f ,50.0f ,255.0f ,100.0f)},
		{"ELEV2_CIEL",					vec4(255.0f ,200.0f ,100.0f ,800.0f)},
		{"YELLOW",						vec4(255.0f ,200.0f ,100.0f ,2000.0f)},
		{"RED",							vec4(255.0f ,0.0f ,0.0f ,1000.0f)},
		{"~TRN_LT1",					vec4(255.0f ,255.0f ,225.0f ,1700.0f)},
		{"+AEXIT",						vec4(0.0f ,255.0f ,35.0f ,1900.0f)},
		{"~SPOTRED",					vec4(255.0f ,0.0f ,0.0f ,800.0f)},
		{"!FLUID2B",					vec4(255.0f ,180.0f ,20.0f ,9000.0f)},
		{"~LIGHT5A",					vec4(65.0f ,90.0f ,150.0f ,3500.0f)},
		{"+A~FIFTS_LGHT06",				vec4(255.0f ,0.0f ,0.0f ,1000.0f)},
	};
}

namespace comfy
{
	void* cuda_allocator::allocate(size_t n, int flags)
	{
		void* cumem;
		cuMemAllocHost(&cumem, n);
		return cumem;
	}
	void* cuda_allocator::allocate(size_t n, size_t alignment, size_t offset, int flags)
	{
		return allocate(n, flags);
	}
	void cuda_allocator::deallocate(void* d, size_t)
	{
		cuMemFreeHost(d);
	}


	struct bsptextureEx : bsptexture
	{
		/*
		char name[16];
		u32 width;
		u32 height;
		u32 offsets[4];
		*/
		inline vec2i dimensions(int mipmaplevel) const
		{
			return vec2i(width >> mipmaplevel, height >> mipmaplevel);
		}

		inline const byte* data(int mipmaplevel) const
		{
			return ((byte*)this) + offsets[mipmaplevel];
		}
		inline const colrgb* colortable() const
		{
			return (colrgb*)(data(3) + dimensions(3).x * dimensions(3).y + 2);
		}
	};
	struct WAD {
		struct WADHEADER
		{
			char magic[4];		// should be WAD2/WAD3
			i32 ndirs;			// number of directory entries
			i32 diroffset;		// offset into directory
		};
		struct WADDIRENTRY
		{
			i32 offset;
			i32 store_size;
			i32 size;
			byte type;
			byte iscompressed;
			i16 unk0;
			char name[16];
		};

		byte* data = NULL;
		u32 size = 0;
		eastl::map<string, int> entryidx;
		WADHEADER* header;
		WADDIRENTRY* entries;

		void Init()
		{
			header = (WADHEADER*)data;
			entries = (WADDIRENTRY*)(data + header->diroffset);

			entryidx["INVALID"] = 0;

			for (int i = 0; i < header->ndirs; i++)
			{
				entryidx[entries[i].name] = i;
			}
		}

		const bsptextureEx* GetTexture(const char * _name) const
		{
			char name[16];
			memcpy(name, _name, 16);

			WADDIRENTRY entry = entries[entryidx.at(_strupr(name))];

			Assert(entry.iscompressed == false);// Not supported

			return (const bsptextureEx*)(data + entry.offset);
		}
	};

	class BSP : public IBSP
	{
	private:
		
		eastl::multimap<string, eastl::map<string, string>> entities;
		vector<vec4> tris;
		vector<loadedtexture> loadedtextures;
		CUarray cuarr_nodes, cuarr_leafs;
		CUtexObject tex_nodefs, tex_leafsf;
		// Quake code
		size_t DecompressPVS(const byte* in, byte* visdata_out, int visbytes)
		{
			byte* out = visdata_out;

			int	c;
			do
			{
				if (*in)
				{
					*out++ = *in++;
					continue;
				}
				c = in[1];
				in += 2;
				while (c)
				{
					*out++ = 0;
					c--;
				}
			} while (out - visdata_out < visbytes);
			return int(out - visdata_out);
		}



		array<vec4, 3>& Woopify(vec3 _v0, vec3 _v1, vec3 _v2) {
			typedef cugar::Matrix<double, 4, 4> mat4d;
			typedef cugar::Vector4d vec4d;
			typedef cugar::Vector3d vec3d;
			vec3d v0 = vec3d(_v0) / SCALING_FACTORd;
			vec3d v1 = vec3d(_v1) / SCALING_FACTORd;
			vec3d v2 = vec3d(_v2) / SCALING_FACTORd;

			static array<vec4, 3> out;
			static mat4d mtxin;
			static mat4d mtxinv;

			{
				vec4d col0 = vec4d(v0 - v2, 0.0);
				mtxin.r[0][0] = col0.x;
				mtxin.r[1][0] = col0.y;
				mtxin.r[2][0] = col0.z;
				mtxin.r[3][0] = col0.w;

				vec4d col1 = vec4d(v1 - v2, 0.0);
				mtxin.r[0][1] = col1.x;
				mtxin.r[1][1] = col1.y;
				mtxin.r[2][1] = col1.z;
				mtxin.r[3][1] = col1.w;

				vec4d col2 = vec4d(cross(v0 - v2, v1 - v2), 0.0);
				mtxin.r[0][2] = col2.x;
				mtxin.r[1][2] = col2.y;
				mtxin.r[2][2] = col2.z;
				mtxin.r[3][2] = col2.w;

				vec4d col3 = vec4d(v2, 1.0f);
				mtxin.r[0][3] = col3.x;
				mtxin.r[1][3] = col3.y;
				mtxin.r[2][3] = col3.z;
				mtxin.r[3][3] = col3.w;
			}
			cugar::invert(mtxin, mtxinv);

			out[0] = vec4((float)mtxinv(2, 0), (float)mtxinv(2, 1), (float)mtxinv(2, 2), (float)-mtxinv(2, 3));
			out[1] = vec4(mtxinv.get(0));
			out[2] = vec4(mtxinv.get(1));

#if 0
			// Remove -0.0f from all triangles so we can use -0.0f for control flags in the vertex stream
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 4; j++) {
					if (out[i][j] == -0.0f)
						out[i][j] = 0.0f;
				}
			}
#endif

			return out;
		}

		// http://jcgt.org/published/0002/01/05/paper.pdf
		array<vec4, 3>& WoopifyWaterTight(vec3 v0, vec3 v1, vec3 v2) {
			static array<vec4, 3> out;
			out[0] = vec4(v0, 0.0f);
			out[1] = vec4(v1, 0.0f);
			out[2] = vec4(v2, 0.0f);

			// Remove -0.0f from all triangles so we can use -0.0f for control flags in the vertex stream
#if 0
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 4; j++) {
					if (out[i][j] == -0.0f)
						out[i][j] = 0.0f;
				}
			}
#endif
			return out;
		}



		BSPHEADER* Header()
		{
			return (BSPHEADER*)cpumem;
		}

		i32 Lumpsize(i32 lumpidx)
		{
			return Header()->lump[lumpidx].nLength;
		}

		void* Lump(i32 lumpidx)
		{
			return cpumem + Header()->lump[lumpidx].nOffset;
		}
	public:
		WAD* wad;

		i16 GetLeafIdx(vec3 pos) override {
			auto bsp_planes = (BSPPLANE*)Lump(LUMP_PLANES);

			i16 curnode = 0;
			do { // Go down the stack to find the leaf we're in
				const bspnode& node = nodes[curnode];
				const BSPPLANE& plane = bsp_planes[node.planeidx];

				if ((dot(vec3(plane.normal), pos) - plane.dist) < 0.0f)
					curnode = node.children[1];
				else
					curnode = node.children[0];
			} while ((curnode >= 0));

			return curnode;
		}

		bspvec3i ParseBSPVec3i(const string& in) const
		{
			bspvec3i out;
			sscanf(in.c_str(), "%hd %hd %hd", &out.x, &out.y, &out.z);

			return out;
		}
		vec4i ParseVec4i(const string& in) const
		{
			vec4i out;
			sscanf(in.c_str(), "%d %d %d %d", &out.x, &out.y, &out.z, &out.w);

			return out;
		}
		colrgba ParseColrgba(const string& in) const
		{
			colrgba out;
			int nparsed = sscanf(in.c_str(), "%hhu %hhu %hhu %hhu", &out.x, &out.y, &out.z, &out.w);

			if (nparsed == 3)
				out.w = 255;

			return out;
		}

		int ParseInt(const string& in) const
		{
			int out;
			int nout = sscanf(in.c_str(), "%d", &out);
			if (nout < 1)
				out = 0;
			return out;
		}

		vec3 GetSpawn() const noexcept override
		{
			auto spawn_search = entities.find("info_player_start");
			if (spawn_search != entities.end())
			{
				auto spawn = spawn_search->second;
				return vec3(ParseBSPVec3i(spawn["origin"]));
			}
			return vec3(0.0f);
		}

		bool IsVisible(i16 from_leaf, i16 to_leaf) {
			// @TODO Figure out what goes on here
			static i16 current_leafidx = 0;
			static byte buffer[2048];
			static int visbytes = 0;

			if (to_leaf > 0)
				to_leaf = to_leaf - 1;

			//if (to_leaf == 0)
			//	return false;

			if (from_leaf == to_leaf)
				return true;

			if (from_leaf != current_leafidx) {
				current_leafidx = from_leaf;
				bspnode& leaf = leafs[~current_leafidx];
				if (leaf.pvsidx >= 0) // Decompress the visiblity data for the current PVS
					visbytes = (int)DecompressPVS(leaf.pvsidx + (byte*)Lump(LUMP_VISIBILITY), buffer, 1024);
			}

			return buffer[(to_leaf) >> 3] & (1 << ((to_leaf) & 7));
		}

		void ParseEntitiesUnsafely()
		{
			byte* base = (byte*)Lump(LUMP_ENTITIES);

			eastl::map<string, string> cur;

			std::istringstream stream((char*)base);
			std::string line;
			while (std::getline(stream, line)) {
				if (line[0] == '{') {
					cur = {};
					continue;
				}
				if (line[0] == '}') {
					auto name = cur["classname"];
					entities.insert({ name, cur });
					continue;
				}
				char key[32];
				char value[1024];
				sscanf(line.c_str(), "\"%[^\"]\" \"%[^\"]\"", key, value); // @DANGER
				cur[key] = value;
			}
		}

		void LoadTextures()
		{
			i32 * texoffsets = (i32*)Lump(LUMP_TEXTURES);
			i32 ntex = texoffsets[0]; // First element is array size

			texoffsets++;

			for (i32 i = 0; i < ntex; i++)
			{
				loadedtexture lt{};

				byte* location = (byte*)Lump(LUMP_TEXTURES) + texoffsets[i];
				const bsptextureEx* tex = (const bsptextureEx*)location;

				if (tex->offsets[0] + tex->offsets[1] + tex->offsets[2] + tex->offsets[3] == 0) {
					tex = wad->GetTexture(tex->name);
				}

				CUDA_ARRAY3D_DESCRIPTOR desc{};

				desc.Width = tex->dimensions(0).x;
				desc.Height = tex->dimensions(0).y;
				desc.Depth = 0;
				desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
				desc.NumChannels = 4;

				CUSuccess(cuMipmappedArrayCreate((CUmipmappedArray*)&lt.cuda_array, &desc, 4));

				const colrgb* coltable = tex->colortable();

				for (int miplevel = 0; miplevel < 4; miplevel++) {
					vec2i dim = tex->dimensions(miplevel);

					const byte * data = tex->data(miplevel);

					CUSuccess(cuMemAllocHost((void**)&lt.texdata[miplevel], dim.x* dim.y * sizeof(colrgba)));
					for (int idx = 0; idx < (dim.x * dim.y); idx++)
					{
						lt.texdata[miplevel][idx] = colrgba(coltable[data[idx]], 0);
					}

					CUarray texarr;
					CUSuccess(cuMipmappedArrayGetLevel(&texarr, (CUmipmappedArray)lt.cuda_array, miplevel));


					CUDA_MEMCPY2D copyParam{ 0 };
					copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
					copyParam.srcHost = lt.texdata[miplevel];
					copyParam.srcPitch = 0;
					copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
					copyParam.dstArray = texarr;
					copyParam.Height = tex->dimensions(miplevel).y;
					copyParam.WidthInBytes = tex->dimensions(miplevel).x * 4; // Num channels

					CUSuccess(cuMemcpy2DAsync(&copyParam, 0));
				}

				CUDA_RESOURCE_DESC resdesc{};
				resdesc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
				resdesc.res.mipmap.hMipmappedArray = (CUmipmappedArray)lt.cuda_array;

				CUDA_TEXTURE_DESC texdesc{};
				texdesc.minMipmapLevelClamp = 0;
				texdesc.maxMipmapLevelClamp = 3;
				texdesc.maxAnisotropy = 16;
				texdesc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
				texdesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
				texdesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
				texdesc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
				texdesc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
				texdesc.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;

				CUDA_RESOURCE_VIEW_DESC rvdesc{};
				rvdesc.format = CU_RES_VIEW_FORMAT_UINT_4X8;
				rvdesc.width = desc.Width;
				rvdesc.height = desc.Height;
				rvdesc.depth = desc.Depth;
				rvdesc.firstMipmapLevel = 0;
				rvdesc.lastMipmapLevel = 3;

				CUSuccess(cuTexObjectCreate(&lt.cuda_texture, &resdesc, &texdesc, &rvdesc));

				lt.width = tex->dimensions(0).x;
				lt.height = tex->dimensions(0).y;
				memcpy(lt.name, tex->name, 16);

				loadedtextures.push_back(lt);
			}
		}

		void LightSourceVisiblityMatrix()
		{
			
			const int nleaves = Lumpsize(LUMP_LEAVES) / sizeof(BSPLEAF);
			lights.push_back();

			lightvismatrix.resize(nleaves * MAX_LIGHT_SOURCE_PER_LEAF);
			eastl::fill(lightvismatrix.begin(), lightvismatrix.end(), 0);		

			// Collect all surfaces and entities the lighting pass cares about
			auto bsp_faces = (BSPFACE*)Lump(LUMP_FACES);
			auto bsp_surfedges = (const BSPSURFEDGE*)Lump(LUMP_SURFEDGES);
			auto bsp_edges = (const BSPEDGE*)Lump(LUMP_EDGES);
			auto bsp_verts = (const BSPVERTEX*)Lump(LUMP_VERTICES);

			for (int i = 0; i < faces.size(); i++) {
				faceinfo& f = faces[i];
				if (faces[i].emissive.w > 0.0f) {
					
					comfy::light l{};
					l.origin = faces[i].N.xyz() * (faces[i].N.w);

					auto v4c = vec4(faces[i].emissive);
					l.color = v4c.xyz() / 255.0f * v4c.w;
					l.normal = faces[i].N.xyz();
					l.faceidx = i;
					l.isface = i;

					auto bf = bsp_faces[l.faceidx];

					l.origin = vec3(0.0f);
					for (u32 edgeidx = bf.iFirstEdge; edgeidx < bf.iFirstEdge + bf.nEdges; edgeidx++) {
						auto e = bsp_edges[abs(bsp_surfedges[edgeidx])];

						vec3 v0 = bsp_verts[e.iVertex[0]];
						vec3 v1 = bsp_verts[e.iVertex[1]];

						l.origin += v0;
						l.origin += v1;
					}

					l.area = f.worldarea;// (smax - smin)* (tmax - tmin);
					l.origin /= (bf.nEdges * 2.0f); // Estimate

					l.cone = cugar::cosf(TO_RADIANS(175.0f));
					l.cone2 = cugar::cosf(TO_RADIANS(120.0f));

					lights.push_back(l);
				}
			}
			
			for (auto it = entities.begin(); it != entities.end(); ++it) {
				if (it->first == "light" || it->first == "light_spot") {

					eastl::map<string, string>& ent = it->second;

					comfy::light l{};
					l.origin = vec3(ParseBSPVec3i(ent["origin"]));
					auto v4c = vec4(ParseColrgba(ent["_light"]));
					l.color = v4c.xyz() / 255.0f * v4c.w;

					l.pitch = TO_RADIANS((float)ParseInt(ent["pitch"]));
					l.cone = cugar::cosf(TO_RADIANS((float)ParseInt(ent["_cone"])));
					l.cone2 = cugar::cosf(TO_RADIANS((float)ParseInt(ent["_cone2"])));

					if (l.pitch != 0.0f) {
						l.normal = vec3(0.0f, 0.0f, cugar::sinf(l.pitch));
					}

					l.isface = 0;
					l.area = 1.0f;

					lights.push_back(l);
				}
			}
			
 			for (i16 i = 1; i < nleaves; i++) { // Leaf 0 is empty .
				const bspnode& lf = leafs[i];
				int size = 0;
				for (i16 j = 1; j < lights.size() && size < MAX_LIGHT_SOURCE_PER_LEAF; j++) { // Light 0 also unused
					const light& l = lights[j];

					vec3 leaforigin = (vec3(lf.max) + vec3(lf.min)) * 0.5f;
					vec3 leafextend = (vec3(lf.max) - vec3(lf.min)) * 0.5f;
					float sphereradius = cugar::length(leafextend);

					if (l.isface) {
						if (l.area <= 16.0f)
							continue;
						if (length(leaforigin - l.origin) > sphereradius + length(l.color * l.area))
							continue;
					}
					

					// Too far away 
					

					bool is_light_visible = false;
					if (l.isface) {
						auto bf = bsp_faces[l.isface];

						// Check all verts of the face
						for (u32 edgeidx = bf.iFirstEdge; edgeidx < bf.iFirstEdge + bf.nEdges; edgeidx++) {
							auto e = bsp_edges[abs(bsp_surfedges[edgeidx])];

							i16 lidx0 = GetLeafIdx(bsp_verts[e.iVertex[0]]);
							i16 lidx1 = GetLeafIdx(bsp_verts[e.iVertex[1]]);
							if (IsVisible(~i, ~lidx0) || IsVisible(~i, ~lidx1)) {
								is_light_visible = true;
								break;
							}
						}
					}
					else {
						i16 lidx = GetLeafIdx(l.origin);

						// Just check the origin
						if (IsVisible(~i, ~lidx)) {
							is_light_visible = true;
						}
					}

					if(is_light_visible)
						lightvismatrix[MAX_LIGHT_SOURCE_PER_LEAF * i + size++] = j;
				}
			}
		}

		void Triangulate() {
			//auto bsp_leaves = (BSPLEAF*)Lump(LUMP_LEAVES);
			auto bsp_faces = (BSPFACE*)Lump(LUMP_FACES);
			auto bsp_marksurfs = (BSPMARKSURFACE*)Lump(LUMP_MARKSURFACES);
			auto bsp_verts = (BSPVERTEX*)Lump(LUMP_VERTICES);
			auto bsp_surfedges = (BSPSURFEDGE*)Lump(LUMP_SURFEDGES);
			auto bsp_edges = (BSPEDGE*)Lump(LUMP_EDGES);

			int cullededges = 0;
			for(int i = 0; i < leafs.size();i++) {
				bspnode& leaf = leafs[i];

				// Skip Empty Faces
				if (leaf.nfaces == 0) {
					leaf.vertidx = 0;
					leaf.vertend = 0;
					continue;
				}

				leaf.vertidx = cugar::max(i32(tris.size()), 0);
				for (int j = leaf.faceidx; j < leaf.faceidx + leaf.nfaces; j++) {

					BSPFACE& face = bsp_faces[bsp_marksurfs[j]];

					vector<array<vec3, 3>> optedges;

					for (u32 y = 0; y < face.nEdges; y += 2) {
						int edgeindex = bsp_surfedges[face.iFirstEdge + y];
						BSPEDGE bsp_edge = bsp_edges[abs(edgeindex)];

						vec3 v0 = bsp_verts[bsp_edge.iVertex[0]];
						vec3 v1 = bsp_verts[bsp_edge.iVertex[1]];

						if (edgeindex < 0)
							std::swap(v0, v1);

						vec3 edge = v1 - v0;

						#ifdef CULL_EDGES
						if (y > 0) { // Check previous edge
							array<vec3, 3>& adjacent = optedges[optedges.size() - 1];
							if (dot(normalize(adjacent[2]), normalize(edge)) > 0.999f) {
								cullededges++;
								adjacent[1] = v1;
								continue;
							}
						}

						if ((y + 2) >= face.nEdges) { // if this is the last edge we check the first edge
							array<vec3, 3>& adjacent = optedges[0];
							if (dot(normalize(adjacent[2]), normalize(edge)) > 0.999f) {
								if (adjacent[0] == v1) {
									cullededges++;
									adjacent[0] = v0;
									continue;
								}
							}
						}
						#endif

						optedges.push_back({ v0, v1, edge });
					}

					vector<vec3> faceverts;
					for (int y = 0; y < optedges.size(); y++) {
						faceverts.push_back(optedges[y][0]);
						faceverts.push_back(optedges[y][1]);
					}

					// Create Space for control flag
					tris.push_back(vec4(0.0f));
					u32 StartFlagIdx = u32(tris.size() - 1);

					vec3 v0 = faceverts[0];
					vec3 prev = faceverts[1];
					for (int y = 2; y < faceverts.size(); y++) {
						vec3 v1 = faceverts[y];
						vec3 v2 = prev;

						//auto woopified = WoopifyWaterTight(v0, v1, v2);
						auto woopified = Woopify(v0, v1, v2);

						tris.push_back(woopified[0]);
						tris.push_back(woopified[1]);
						tris.push_back(woopified[2]);

						prev = faceverts[y];
					}

					// Write Control x: Controlbit, Y: nextface position, Z Surface
					uint xyzw[4] = { 0x80000000, uint(tris.size()), uint(bsp_marksurfs[j]) , 0 };
					memcpy(&tris[StartFlagIdx], xyzw, sizeof(vec4));

				} // For faces
				uint xyzw[4] = { 0x80000000, 0, 0, 0 };
				tris.push_back(*(vec4*)&xyzw);

				leaf.vertend = i32(tris.size() - 1);
			} // for leafs

			comfy::DebugPrintF("Triangle Count %d culled edges %d", tris.size() / 3, cullededges);
		}

		void PrepareMap() override
		{
			/*
			BSP:
			Leaf->MarksurfaceIdx
				Marksurfaces->FaceIdx
					Faces->SurfEdgesIdx : Normal Inverse Flag
						SurfEdges->Edge : Edge Inverse Flag
							Edge->VertexAidx, VertexBidx
								VertexA
								VertexB
					Faces->PlaneIdx
						Plane->Normal
					Faces->TextureInfoIdx
						TextureInfo
							Tangent
							Bitangent
							TextureIdx
								Width
								Height
								Data

			Internal:
			Leaf->TrisIdx : nTris
				[v0, v1, v2]

			TrisIdx : FaceIdx

			Loaded texture array matches miptexarray idx's

			Triangles are woopified, so every 'vertex' can be processed by the GPU individually without needing to load the whole triangle
			*/

			auto bsp_leaves = (BSPLEAF*)Lump(LUMP_LEAVES);
			auto bsp_nodes = (BSPNODE*)Lump(LUMP_NODES);
			auto bsp_planes = (BSPPLANE*)Lump(LUMP_PLANES);

			for (int i = 0; i < Lumpsize(LUMP_NODES) / sizeof(BSPNODE); i++)
				nodes.push_back(bspnode::fromBSPNODE(bsp_nodes[i]));

			for (int i = 0; i < Lumpsize(LUMP_LEAVES) / sizeof(BSPLEAF); i++)
				leafs.push_back(bspnode::fromBSPLEAF(bsp_leaves[i]));

			// Make the tree double linked, instead of single
			for (u16 i = 0; i < nodes.size(); i++)
			{
				for (int ci = 0; ci < 2; ci++)
				{
					if (nodes[i].children[ci] != -1)
						if (nodes[i].children[ci] > 0)
							nodes[nodes[i].children[ci]].parent = i;
						else
							leafs[~nodes[i].children[ci]].parent = i;
				}

				nodes[i].plane = vec4(bsp_planes[nodes[i].planeidx].normal, bsp_planes[nodes[i].planeidx].dist);
			}


			// Upsize BB's
			for (int i = 0; i < nodes.size(); i++) {
				nodes[i].min.x -= BB_PADDING;
				nodes[i].min.y -= BB_PADDING;
				nodes[i].min.z -= BB_PADDING;

				nodes[i].max.x += BB_PADDING;
				nodes[i].max.y += BB_PADDING;
				nodes[i].max.z += BB_PADDING;
			}
			for (int i = 0; i < leafs.size(); i++) {
				leafs[i].min.x -= BB_PADDING;
				leafs[i].min.y -= BB_PADDING;
				leafs[i].min.z -= BB_PADDING;

				leafs[i].max.x += BB_PADDING;
				leafs[i].max.y += BB_PADDING;
				leafs[i].max.z += BB_PADDING;
			}


			Triangulate();
			LoadTextures();
			ParseEntitiesUnsafely();

			auto bsp_faces = (const BSPFACE*)Lump(LUMP_FACES);
			const BSPTEXTUREINFO* bsp_texinfos = (const BSPTEXTUREINFO * )Lump(LUMP_TEXINFO);
			//const BSPPLANE* bsp_planes = (const BSPPLANE*)Lump(LUMP_TEXINFO);

			
			auto bsp_surfedges = (const BSPSURFEDGE*)Lump(LUMP_SURFEDGES);
			auto bsp_edges = (const BSPEDGE*)Lump(LUMP_EDGES);
			auto bsp_verts = (const BSPVERTEX*)Lump(LUMP_VERTICES);

			for (int i = 0; i < Lumpsize(LUMP_FACES) / sizeof(BSPFACE);i++) {
				const BSPFACE& face = bsp_faces[i];
				const BSPTEXTUREINFO& texinfo = bsp_texinfos[face.iTextureInfo];
				const loadedtexture tex = loadedtextures[texinfo.iMiptex];

				auto bp = bsp_planes[face.iPlane];				
				vec4 n = vec4(bp.normal, bp.dist);

				if (face.nPlaneSide != 0)
					n = -n;

				faceinfo fi{};
				fi.N = n;
				fi.S = vec4(texinfo.vS, texinfo.fSShift);
				fi.T = vec4(texinfo.vT, texinfo.fTShift);
				fi.width = tex.width;
				fi.height = tex.height;
				fi.cuda_texture = tex.cuda_texture;

				vec3 s = normalize(fi.S.xyz());
				vec3 t = normalize(fi.T.xyz());

				s = normalize(cross(t, fi.N.xyz())); // Bitangent

				float smin = 3200000.0f;
				float tmin = 3200000.0f;

				float smax = -3200000.0f;
				float tmax = -3200000.0f;

				for (u32 edgeidx = face.iFirstEdge; edgeidx < face.iFirstEdge + face.nEdges; edgeidx++) {
					auto e = bsp_edges[abs(bsp_surfedges[edgeidx])];

					vec3 v0 = bsp_verts[e.iVertex[0]];
					vec3 v1 = bsp_verts[e.iVertex[1]];

					smin = cugar::min(smin, cugar::dot(v0, s));
					smin = cugar::min(smin, cugar::dot(v1, s));
					tmin = cugar::min(tmin, cugar::dot(v0, t));
					tmin = cugar::min(tmin, cugar::dot(v1, t));

					smax = cugar::max(smax, cugar::dot(v0, s));
					smax = cugar::max(smax, cugar::dot(v1, s));
					tmax = cugar::max(tmax, cugar::dot(v0, t));
					tmax = cugar::max(tmax, cugar::dot(v1, t));
				}

				fi.worldarea = (smax - smin) * (tmax - tmin);
				
				string sn = string(tex.name);
				sn.make_upper();

				if (emissives.count(sn)) {
					auto e = emissives[sn];
					fi.emissive = vec4(e.xyz() / 255.0f, e.w);
				} 
				else {
					fi.emissive = vec4(0.0f);
				}

				faces.push_back(fi);
			}

			LightSourceVisiblityMatrix();
		}

		void PrepareFrame(vec3 campos, array<vec3, 5> camerafrustrums, uint framenumber) override
		{

			const int nleaves = Lumpsize(LUMP_LEAVES) / sizeof(BSPLEAF);

			i16 camera_node_idx = GetLeafIdx(campos);

			/*
			int curnode = 0;
			do { // Go down the stack to find the leaf we're in
				const bspnode& node = nodes[curnode];
				const BSPPLANE& plane = bsp_planes[node.planeidx];

				if ((dot(vec3(plane.normal), campos) - plane.dist) > 0.0f)
					curnode = node.children[0];
				else
					curnode = node.children[1];
			} while ((curnode >= 0));

			int visbytes = 0;
			byte buffer[2048];
			bspnode& leaf = leafs[~curnode];
			if (leaf.pvsidx >= 0) // Decompress the visiblity data for the current PVS
				visbytes = (int)DecompressPVS(leaf.pvsidx + bsp_vis, buffer, 1024);
				*/

			for (i16 i = 0; i < nleaves; i++) {
				if(IsVisible(camera_node_idx, i)) {
					bspnode& pvleaf = leafs[i];
					auto center = (vec3(pvleaf.max + pvleaf.min) * 0.5f);
					auto extent = vec3(pvleaf.max - pvleaf.min) * 0.5f;

					int boxvis = 0;
					for (int fidx = 0; fidx < 5; fidx++) {
						vec3 plane = camerafrustrums[fidx];
						vec3 absplane = vec3(fabsf(plane.x), fabsf(plane.y), fabsf(plane.z));

						float d = dot(center - campos, plane);
						float r = dot(extent, absplane);

						if (d + r > 0.0f) // partially inside
							boxvis++;
						else if (d - r >= 0.0f) // fully inside
							boxvis++;
						else
							break;
					}

					if (boxvis != 5) {
						continue;
					}

					leafs[i].visframe = framenumber;

					i16 nodeidx = leafs[i].parent;
					while ((nodeidx != 0) && (nodeidx >= 0))
					{
						bspnode& node = nodes[nodeidx];

						if (node.visframe == framenumber)
							break;

						node.visframe = framenumber;
						nodeidx = node.parent;
					}
				}
			}

			// Static cull
			for (i16 i = 0; i < nodes.size(); i++) {
				nodes[i].vischildren[0] = nodes[i].children[0]; // Reset the visiblity
				nodes[i].vischildren[1] = nodes[i].children[1];

#ifdef CULL_AABB
				for (int ci = 0; ci < 2; ci++) // For each Child
				{
					if (nodes[i].children[ci] != -1) { // If the Child is not Invalid (-1)
						if (nodes[i].children[ci] > 0) { // If its a node
							if (nodes[nodes[i].children[ci]].visframe != framenumber) { // If its not visible this frame
								nodes[i].vischildren[ci] = -1; // Kill it
							}
						}
						else { // If its a leaf
							if (leafs[~nodes[i].children[ci]].visframe != framenumber) {
								nodes[i].vischildren[ci] = -1;
							}
						}
					}
				}
#endif
			}


			// Sort all childs front to back before sending it off to the GPU.
			
			for (i16 i = 0; i < nodes.size(); i++) {
				bspnode * node = &nodes[i];

				i16 child0 = node->vischildren[0];
				i16 child1 = node->vischildren[1];

				bool goChild0 = child0 != -1;
				bool goChild1 = child1 != -1;


				if (goChild0 && goChild1) {
					vec4 normplane = node->plane;
					float d = (dot(normplane.xyz(), campos) - normplane.w);

					if (d < 0.0f) {
						// Swap
						i16 c = node->vischildren[0];
						node->vischildren[0] = node->vischildren[1];
						node->vischildren[1] = c;

						// Invert the plane so dot(plane,pos) returns the right result.
						node->visplane = -node->plane;
					}
				}
			}
			
			nodesf.clear();
			leafsf.clear();

			for (i32 i = 0; i < nodes.size(); i++) {
				nodesf.push_back(*(bspnodef*)&nodes[i]);
				nodesf[i].plane = nodes[i].visplane;
			}

			for (i32 i = 0; i < leafs.size(); i++) {
				leafsf.push_back(*(bspnodef*)&leafs[i]);
				leafsf[i].vertidx = leafs[i].vertidx;
			}

			CUSuccess(cuMemcpyHtoAAsync(cuarr_nodes,0,nodesf.data(), nodesf.size() * sizeof(bspnodef), 0));
			CUSuccess(cuMemcpyHtoAAsync(cuarr_leafs,0,leafsf.data(), leafsf.size() * sizeof(bspnodef), 0));

			gpu_nodes = (u64)tex_nodefs;
			gpu_leafs = (u64)tex_leafsf;
		}


		void GPUInit(ID3DCudaDevice * device) override
		{
			CUSuccess(cuMemAlloc(&gpu_bsp, bspsize));
			CUSuccess(cuMemAlloc(&gpu_leafs, leafs.size() * sizeof(bspnode)));
			CUSuccess(cuMemAlloc(&gpu_nodes, nodes.size() * sizeof(bspnode)));
			CUSuccess(cuMemAlloc(&gpu_tris, tris.size() * sizeof(vec4)));
			CUSuccess(cuMemAlloc(&gpu_leafs, leafs.size() * sizeof(bspnode)));

			CUSuccess(cuMemAlloc(&gpu_faceinfo, faces.size() * sizeof(faceinfo)));
			CUSuccess(cuMemAlloc(&gpu_lights, lights.size() * sizeof(light)));
			CUSuccess(cuMemAlloc(&gpu_lightvismatrix, lightvismatrix.size() * sizeof(i16)));

			{
				CUDA_RESOURCE_DESC resdesc{};
				resdesc.resType = CU_RESOURCE_TYPE_LINEAR;
				resdesc.res.linear.devPtr = gpu_tris;
				resdesc.res.linear.format = CU_AD_FORMAT_FLOAT;
				resdesc.res.linear.numChannels = 4;
				resdesc.res.linear.sizeInBytes = tris.size() * sizeof(vec4);
				CUDA_TEXTURE_DESC texdesc{};

				CUSuccess(cuTexObjectCreate(&gpu_tex_tris, &resdesc, &texdesc, NULL));
			}

			{
				CUDA_ARRAY_DESCRIPTOR desc{ 0 };
				desc.Format = CU_AD_FORMAT_SIGNED_INT32;
				desc.NumChannels = 4;
				desc.Width = 8192*2;
				desc.Height = 0;
				CUSuccess(cuArrayCreate(&cuarr_nodes, &desc));
				CUSuccess(cuArrayCreate(&cuarr_leafs, &desc));

				// set texture parameters
				CUDA_RESOURCE_DESC resdesc{};

				resdesc.resType = CU_RESOURCE_TYPE_ARRAY;
				resdesc.res.array.hArray = cuarr_nodes;

				CUDA_TEXTURE_DESC texdesc{};
				texdesc.flags = CU_TRSF_READ_AS_INTEGER;

				CUDA_RESOURCE_VIEW_DESC format;
				memset(&format, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));
				format.format = CU_RES_VIEW_FORMAT_SINT_4X32;
				format.width = 8192/2;
				format.height = 0;
				format.depth = 0;

				CUSuccess(cuTexObjectCreate(&tex_nodefs, &resdesc, &texdesc, NULL));
				resdesc.res.array.hArray = cuarr_leafs;
				CUSuccess(cuTexObjectCreate(&tex_leafsf, &resdesc, &texdesc, NULL));
			}

			SyncToGPU(device);
		}

		void SyncToGPU(ID3DCudaDevice* device) override
		{
			Assert(nodes.size() > 0);
			Assert(leafs.size() > 0);

			CUSuccess(cuMemcpyHtoD(gpu_bsp, cpumem, bspsize));
			//CUSuccess(cuMemcpyHtoD(gpu_leafs, leafs.data(), leafs.size() * sizeof(bspnode)));
			//CUSuccess(cuMemcpyHtoD(gpu_nodes, nodes.data(), nodes.size() * sizeof(bspnode)));
			CUSuccess(cuMemcpyHtoD(gpu_tris, tris.data(), tris.size() * sizeof(vec4)));
			CUSuccess(cuMemcpyHtoD(gpu_faceinfo, faces.data(), faces.size() * sizeof(faceinfo)));
			CUSuccess(cuMemcpyHtoD(gpu_lights, lights.data(), lights.size() * sizeof(light)));
			CUSuccess(cuMemcpyHtoD(gpu_lightvismatrix, lightvismatrix.data(), lightvismatrix.size() * sizeof(i16)));
		}
	};

	IBSP* IBSP::FromFile(string filename) noexcept {
		BSP * _out = new BSP;
		BSP& out = *_out;
		static WAD wad;

		{
			// Load into memory
			HANDLE h = CreateFile(
				"halflife.wad", GENERIC_READ, FILE_SHARE_READ,
				NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

			if (h == INVALID_HANDLE_VALUE)
				format_winerr_break("CreateFile");

			wad.size = GetFileSize(h, NULL);
			wad.data = (byte*)malloc(wad.size);

			DWORD bytesRead = 0;
			if (!ReadFile(h, wad.data, wad.size, &bytesRead, NULL))
				format_winerr_break("ReadFile");
			CloseHandle(h);

			Assert(bytesRead == wad.size);

			wad.Init();
		}

		{
			// Load into memory
			HANDLE h = CreateFile(
				filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
				NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

			if (h == INVALID_HANDLE_VALUE)
				format_winerr_break("CreateFile");

			out.bspsize = GetFileSize(h, NULL);
			out.cpumem = (byte*)malloc(out.bspsize);

			DWORD bytesRead = 0;
			if (!ReadFile(h, out.cpumem, DWORD(out.bspsize), &bytesRead, NULL))
				format_winerr_break("ReadFile");
			CloseHandle(h);

			Assert(bytesRead == out.bspsize);
		}

		out.wad = &wad;

		return &out;
	}
}

#undef _CRT_SECURE_NO_WARNINGS