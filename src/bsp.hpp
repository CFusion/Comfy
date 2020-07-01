// http://hlbsp.sourceforge.net/index.php?content=bspdef
#pragma once
#include "host.hpp"

namespace comfy
{
	class EASTL_API cuda_allocator
	{
	public:
		EASTL_ALLOCATOR_EXPLICIT cuda_allocator(const char* pName = EASTL_NAME_VAL("COMFY")) { }
		cuda_allocator(const cuda_allocator&) { }
		cuda_allocator(const cuda_allocator&, const char*) { }

        void* allocate(size_t n, int flags = 0);
        void* allocate(size_t n, size_t alignment, size_t offset, int flags = 0);
        void  deallocate(void* d, size_t);

		cuda_allocator& operator=(const cuda_allocator&) { return *this; }
		const char* get_name() const { return ""; }
		void        set_name(const char*) { }
	};

	inline bool operator==(const cuda_allocator&, const cuda_allocator&) { return true; }
	inline bool operator!=(const cuda_allocator&, const cuda_allocator&) { return false; }

    class IBSP {
    public:
        u64 bspsize;
        u64 gpu_bsp, gpu_nodes, gpu_leafs, gpu_tris, gpu_tex_tris, gpu_tristosurf, gpu_faceinfo, gpu_lightvismatrix, gpu_lights;
        byte * cpumem;
        vector<bspnode> nodes;
        vector<bspnode> leafs;

        vector<faceinfo, cuda_allocator> faces;
        vector<bspnodef, cuda_allocator> nodesf;
        vector<bspnodef, cuda_allocator> leafsf;
        vector<light, cuda_allocator> lights;
        vector<i16, cuda_allocator> lightvismatrix;

        virtual vec3 GetSpawn() const noexcept = 0;
        static IBSP * FromFile(string filename) noexcept;
        virtual void SyncToGPU(ID3DCudaDevice* device) = 0;
        virtual void PrepareMap() = 0;
        virtual void GPUInit(ID3DCudaDevice* device) = 0;
        virtual void PrepareFrame(vec3 campos, array<vec3, 5> camerafrustrums, uint framenumber) = 0;
        virtual i16 GetLeafIdx(vec3 pos) = 0;
    };
}