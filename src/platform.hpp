#pragma once
#ifdef COMFY_IMPLEMENTATION
#define COMFY_EXTERN
#else
#define COMFY_EXTERN extern
#endif
#include "host.hpp"

namespace comfy {
	namespace state {
		enum KEYSTATE : i32 {
			NONE = 0,
			DOWN = 1,
			PRESS = 4,
			RELEASE = 8
		};

		COMFY_EXTERN float frametime;
		COMFY_EXTERN volatile i32 keyframe[0xFF];
		COMFY_EXTERN volatile i32 key[0xFF];
		COMFY_EXTERN vec2 lastmousepos;
		COMFY_EXTERN volatile vec2 mousemove;
		COMFY_EXTERN i32 isfocus;
		COMFY_EXTERN void* windowhandle;
		COMFY_EXTERN hash_map<string, i32> launchparams;

		inline bool KeyUp(int vkey) {
			return (keyframe[vkey] & RELEASE) == RELEASE;
		}
		inline bool KeyPressed(int vkey) {
			return (keyframe[vkey] & PRESS) == PRESS;
		}
		inline bool KeyDown(int vkey) {
			return (keyframe[vkey] & DOWN) == DOWN;
		}
		inline void KeyFrameStart() {
			for (int i = 0; i < 0xFF; i++) {
				keyframe[i] = key[i];

				key[i] &=  ~RELEASE;
				key[i] &=  ~PRESS;
			}
		}
	}

	class ICudaKernel
	{
	public:
		static ICudaKernel* Create(const string name);
		virtual CUHandle2 Get() = 0;
	};

	class ID3D9CudaTexture
	{
	public:
		uint height, width;
		size_t pitch;
		u64 cuda_device_mem;

		virtual void Release() = 0;
		virtual void CopyToDirectXAsync() = 0;
		virtual void Init() = 0;
		virtual void Bind() = 0;
		virtual void Unbind() = 0;
		virtual void Draw() = 0;
	};

	class ID3DCudaDevice
	{
	public:
		std::recursive_timed_mutex mutex;

		static const int frames_saved = 30;

		uint cuda_completed_frame_count = 0;
		float cuda_frametimes[frames_saved];
		ID3D9CudaTexture* cuda_render_texture = NULL;

		inline float CudaFrametime() {
			if (cuda_completed_frame_count < 1)
				return 0.0f;
			return cuda_frametimes[(cuda_completed_frame_count - 1) % frames_saved];
		}

		virtual void StartFrame() = 0;
		virtual void EndFrame() = 0;
		virtual bool TryResize(uint width, uint height) = 0;
		virtual void CudaRun(const string function, void* args[], ICudaKernel* cudakernel, int shared_memorysize = 0) = 0;
		virtual u64 CudaMalloc(size_t size) = 0;
	};

}