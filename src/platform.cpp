#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define NO_MIN_MAX
#define COMFY_IMPLEMENTATION
#include <Windows.h>
#include <d3d9.h>
#include <tchar.h>

#include <nvrtc.h>
#include <cuda.h>
#include <cudaD3D9.h>

#include <thread>
#include <new.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_dx9.h"
#include "imgui/imgui_impl_win32.h"

#include "platform.hpp"
#include "host.hpp"
#include "platform_utlity.hpp"

using namespace comfy;

extern "C"
{
	__declspec(dllexport) DWORD NvOptimusEnablement = 0x01;
	__declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x01;
}

void* operator new[](size_t size, const char* /*name*/, int /*flags*/, unsigned /*debugFlags*/,
	const char* /*file*/, int /*line*/);
void* operator new[](size_t size, size_t alignment, size_t /*alignmentOffset*/,
	const char* /*name*/, int /*flags*/, unsigned /*debugFlags*/, const char* /*file*/,
	int /*line*/);

void* __cdecl operator new[](
	size_t size, const char* name, int flags, unsigned int debugFlags, const char* file, int line)
{
	return new byte[size];
}
void* __cdecl operator new[](size_t size, size_t alignment, size_t, const char* name, int flags,
	unsigned debugFlags, const char* file, int line)
{
	return new byte[size];
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
void ThreadProc();

static bool g_running = true;

namespace comfy {
	class CudaKernel : public ICudaKernel {
	public:
		u64 last_resolved_time = 0;
		CUHandle2 pModule = NULL;
		string srcpath;
		string name;

		CudaKernel(const string inname) {
			name = inname;
			srcpath.sprintf("src/%s.cu", &name[0]);
		}

		CUHandle2 Get() {
			comfy::u64 last_source_write = GetWriteTime(srcpath);

			// Skip if src file was not updated
			if (last_source_write <= last_resolved_time && !comfy::state::KeyPressed(0x74)) {
				return pModule;
			}

			static comfy::string name_with_time(""); // Expected location of cubin if it's been allready compiled
			name_with_time.sprintf("data/%s.cu.cubin.%I64u", &name[0], last_source_write);

			// Skip if output file allready exists
			if (FileExists(name_with_time) && !comfy::state::KeyPressed(0x74)) {
				GetCubin(name_with_time, (CUmodule *)&pModule);
				last_resolved_time = last_source_write;
				return pModule;
			}

			// File changed or F5 was held down
			RunAndPipeToDebugPrint("compile_shader.bat");
			CopyFile("data\\kernel.cubin", name_with_time.c_str(), 0);

			GetCubin(name_with_time, (CUmodule * )&pModule);

			return pModule;
		}
	};

	ICudaKernel * ICudaKernel::Create(const string name) {
		return new CudaKernel(name);
	}
}

class D3D9CudaTexture : public comfy::ID3D9CudaTexture
{
	typedef float TexType;
	size_t CHANNELS = 4;
	//D3DFORMAT D3FORMAT = D3DFMT_A8R8G8B8;
	D3DFORMAT D3FORMAT = D3DFMT_A32B32G32R32F;

public:
	IDirect3DTexture9* d3d9_texture{ 0 };
	CUgraphicsResource cuda_resource{ 0 };
	CUarray cuda_array{ 0 };
	IDirect3DDevice9* d3dd{ 0 };
	D3DPRESENT_PARAMETERS* d3dpp{ 0 };

	D3D9CudaTexture(IDirect3DDevice9* d, D3DPRESENT_PARAMETERS* p) : d3dd(d), d3dpp(p) {}

	void Bind() override
	{
		CUSuccess(cuGraphicsMapResources(1, &cuda_resource, 0));
		CUSuccess(cuGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0));
	}
	void Unbind() override
	{
		CUSuccess(cuGraphicsUnmapResources(1, &cuda_resource, 0));
	}

	void CopyToDirectXAsync() override {
		CUDA_MEMCPY2D copyParam{ 0 };

		copyParam.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		copyParam.srcDevice = cuda_device_mem;
		copyParam.srcPitch = pitch;
		copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		copyParam.dstArray = cuda_array;
		copyParam.Height = height;
		copyParam.WidthInBytes = width* CHANNELS * sizeof(TexType);

		CUSuccess(cuMemcpy2DAsync(&copyParam, 0));
	}
	void Init() override {
		width = d3dpp->BackBufferWidth;
		height = d3dpp->BackBufferHeight;

		Assert(d3dd->CreateTexture(width, height, 1, 0,
			D3FORMAT, D3DPOOL_DEFAULT, &d3d9_texture, NULL) == 0);

		CUSuccess(cuGraphicsD3D9RegisterResource(&cuda_resource, d3d9_texture, 0));
		CUSuccess(cuMemAllocPitch(&cuda_device_mem, &pitch, uint(size_t(width) * sizeof(TexType) * CHANNELS), uint(size_t(height)),uint(sizeof(TexType) * CHANNELS)));
		//CUSuccess(cuMemsetD2D8(cuda_device_mem, pitch, 0, width, height));// cuMemsetD8(cuda_device_mem, 0, pitch * height));
	}
	void Release() override {
		if (cuda_resource != NULL)
			cuGraphicsUnregisterResource(cuda_resource);
		if (cuda_device_mem != NULL)
			cuMemFree(cuda_device_mem);
		if (d3d9_texture != NULL)
			d3d9_texture->Release();

		d3d9_texture = NULL;
		cuda_device_mem = NULL;
		cuda_resource = NULL;
	}

	void SetGenericState()
	{
		D3DVIEWPORT9 vp;
		vp.X = vp.Y = 0;
		vp.Width = (DWORD)d3dpp->BackBufferWidth;
		vp.Height = (DWORD)d3dpp->BackBufferHeight;
		vp.MinZ = 0.0f;
		vp.MaxZ = 1.0f;
		d3dd->SetViewport(&vp);

		d3dd->SetPixelShader(NULL);
		d3dd->SetVertexShader(NULL);
		d3dd->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
		d3dd->SetRenderState(D3DRS_LIGHTING, FALSE);
		d3dd->SetRenderState(D3DRS_ZENABLE, FALSE);
		d3dd->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
		d3dd->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
		d3dd->SetRenderState(D3DRS_BLENDOP, D3DBLENDOP_ADD);
		d3dd->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
		d3dd->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
		d3dd->SetRenderState(D3DRS_SCISSORTESTENABLE, TRUE);
		d3dd->SetRenderState(D3DRS_SHADEMODE, D3DSHADE_GOURAUD);
		d3dd->SetRenderState(D3DRS_FOGENABLE, FALSE);
		d3dd->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
		d3dd->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);

		// Setup orthographic projection matrix stolen from imgui :0
		{
			float L = 0.5f;
			float R = d3dpp->BackBufferWidth + 0.5f;
			float T = 0.5f;
			float B = d3dpp->BackBufferHeight + 0.5f;
			D3DMATRIX mat_identity = { { { 1.0f, 0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f } } };
			D3DMATRIX mat_projection =
			{ { {
				2.0f / (R - L),   0.0f,         0.0f,  0.0f,
				0.0f,         2.0f / (T - B),   0.0f,  0.0f,
				0.0f,         0.0f,         0.5f,  0.0f,
				(L + R) / (L - R),  (T + B) / (B - T),  0.5f,  1.0f
			} } };
			d3dd->SetTransform(D3DTS_WORLD, &mat_identity);
			d3dd->SetTransform(D3DTS_VIEW, &mat_identity);
			d3dd->SetTransform(D3DTS_PROJECTION, &mat_projection);
		}
	}

	void Draw() override {
		unsigned int IB[6] =
		{
			0,1,2,
			0,2,3,
		};
		struct VertexStruct
		{
			float position[3];
			float texture[3];
		};
		VertexStruct VB[4] =
		{
			{{0,0,0.0},							{0,0,0,},},
			{{float(width),0,0.0},				{1,0,0,},},
			{{float(width), float(height),0.0},	{1,1,0,},},
			{{0,float(height),0.0},				{0,1,0,},},
		};

		SetGenericState();

		d3dd->SetFVF(D3DFVF_XYZ | D3DFVF_TEX1 | D3DFVF_TEXCOORDSIZE2(0));
		d3dd->SetTexture(0, d3d9_texture);
		d3dd->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB, D3DFMT_INDEX32, VB, sizeof(VertexStruct));
	}
};


class D3DCudaDevice : comfy::ID3DCudaDevice
{
public:
	CUevent start, stop, prevframe_start, prevframe_end, startblocking = NULL;
	LPDIRECT3DDEVICE9EX		d3dd = NULL;
	CUcontext cuda_ctx;
private:
	CUdevice cuda_device = -1;
	LPDIRECT3D9EX			d3d = NULL;
	D3DPRESENT_PARAMETERS	d3dpp{ 0 };
	CUdeviceptr gpumem;

	unsigned int EventType = CU_EVENT_BLOCKING_SYNC;
public:

	bool TryResize(uint width, uint height) override
	{
		using Ms = std::chrono::milliseconds;
		if (mutex.try_lock_for(Ms(200))) {
			CUSuccess(cuCtxSetCurrent(cuda_ctx));

			if (d3dd != NULL) {
				d3dpp.BackBufferWidth = width;
				d3dpp.BackBufferHeight = height;

				Reset();
			}
			mutex.unlock();

			return true;
		}
		else
		{
			comfy::DebugPrintF("Failed to get Mutex for D3DCudaDevice from thread %u after 200ms", comfy::GetThreadID());
			return false;
		}
	}
	void StartFrame() override
	{
		// ------------------------
		mutex.lock();
		// ------------------------
	}

	void EndFrame() override
	{
		d3dd->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, NULL, 1.0f, 0);
		if (d3dd->BeginScene() >= 0)
		{
			cuda_render_texture->Draw();
			ImGui::Render();
			ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
			d3dd->EndScene();
		}

		ImGuiIO& io = ImGui::GetIO();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}

		// ------------------
		mutex.unlock();
		// ------------------

		// We don't include d3dd->Present in the mutex because thats where dx9 blocks
		// d3d9 in multithreaded mode is caked with mutexes so this is probably fine
		HRESULT result = d3dd->Present(NULL, NULL, NULL, NULL);
		if (result == D3DERR_DEVICELOST && d3dd->TestCooperativeLevel() == D3DERR_DEVICENOTRESET)
			Reset();
	}

	void InitResources()
	{
		CUSuccess(cuD3D9CtxCreateOnDevice(&cuda_ctx, CU_CTX_SCHED_BLOCKING_SYNC, d3dd, cuda_device));
		CUSuccess(cuCtxSetCurrent(cuda_ctx));

		cuda_render_texture = new D3D9CudaTexture(d3dd, &d3dpp);
		cuda_render_texture->Init();
		cuda_render_texture->Bind();

		cuEventCreate(&start, EventType);
		cuEventCreate(&stop, EventType);
		cuEventCreate(&startblocking, CU_EVENT_BLOCKING_SYNC);

	}

	void Create(HWND hWnd)
	{
		if (!SUCCEEDED(Direct3DCreate9Ex(D3D_SDK_VERSION, &d3d)))
			COMFY_FAIL();

		UINT adapter = 0;

		for (adapter = 0; adapter < d3d->GetAdapterCount(); adapter++) {
			D3DADAPTER_IDENTIFIER9 adapterId;
			HRESULT hr = d3d->GetAdapterIdentifier(adapter, 0, &adapterId);

			if (FAILED(hr))
				continue;

			auto cuStatus = cuD3D9GetDevice(&cuda_device, adapterId.DeviceName);
			if (cuStatus == CUDA_SUCCESS) {
				comfy::DebugPrintF("cuD3D9GetDevice for %s resulted %s", adapterId.DeviceName, "CUDA_SUCCESS");
				break;
			}
			else {
				const char* msg;
				cuGetErrorName(cuStatus, &msg);
				comfy::DebugPrintF("cuD3D9GetDevice for %s resulted %s", adapterId.DeviceName, msg);
			}
		}

		if (cuda_device == -1)
			COMFY_FAIL();

		// Create the D3DDevice
		ZeroMemory(&d3dpp, sizeof(d3dpp));
		d3dpp.Windowed = TRUE;
		d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
		d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;
		d3dpp.EnableAutoDepthStencil = TRUE;
		d3dpp.AutoDepthStencilFormat = D3DFMT_D16;
		d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
		d3dpp.hDeviceWindow = hWnd;
		if (!SUCCEEDED(d3d->CreateDeviceEx(adapter, D3DDEVTYPE_HAL, hWnd,
			D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED, &d3dpp, NULL, &d3dd)))
			COMFY_FAIL();

		return;
	}
	void Cleanup()
	{
		mutex.lock();
		if (d3dd) { d3dd->Release(); d3dd = NULL; }
		if (d3d) { d3d->Release(); d3d = NULL; }
		mutex.unlock();
	}
	u64 CudaMalloc(size_t size) override
	{
		CUdeviceptr cudev = NULL;
		CUSuccess(cuMemAlloc(&cudev, size));
		return cudev;
	}
	void CudaRun(const string function, void* args[], ICudaKernel * cudakernel, int shared_memorysize = 0) override
	{
		static int frame_number = 1;

		if(state::KeyDown(0x71))
			CUSuccess(cuCtxSynchronize());

		// Return if the GPU is yet to start on the current frame
		if (cuEventQuery(start) != CUDA_SUCCESS) {
			if (comfy::state::frametime < (1.0f / 30.0f)) { // If FPS is high we spinlock
				cuEventSynchronize(startblocking);
			}
			else {
				return;
			}
		}

		// If the GPU stream started on the current frame, the previous frame must have finished
		if ((prevframe_end != NULL) && (prevframe_start != NULL)) {
			if ((cuEventQuery(prevframe_start) == CUDA_SUCCESS) && (cuEventQuery(prevframe_end) == CUDA_SUCCESS))
			{
				auto e = cuEventElapsedTime(&(cuda_frametimes[cuda_completed_frame_count % 30]), prevframe_start, prevframe_end);

				if (e == CUDA_SUCCESS)
					cuda_completed_frame_count++;

				CUSuccess(cuEventDestroy(prevframe_start));
				CUSuccess(cuEventDestroy(prevframe_end));
			}
		}

		// Start a new frame by saving the old frame, and creating new timers
		prevframe_start = start;
		prevframe_end = stop;

		CUSuccess(cuEventCreate(&start, EventType));
		CUSuccess(cuEventCreate(&stop, EventType));
		CUSuccess(cuEventCreate(&startblocking, CU_EVENT_BLOCKING_SYNC));

		CUmodule cuda_kernel = (CUmodule)cudakernel->Get();//cuda_compiler.Load(kernel, kernelpath);
		if (cuda_kernel) {
			CUfunction pKernel = NULL;
			CUSuccess(cuModuleGetFunction(&pKernel, cuda_kernel, function.c_str()));

			//CUSuccess(cuFuncSetSharedMemConfig(pKernel, CUsharedconfig::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

			vec3u block = vec3u(BLOCK_WIDTH, BLOCK_HEIGHT, 1); // Threads per block
			vec3u grid = vec3u((cuda_render_texture->width + block.x - 1) / block.x,
				(cuda_render_texture->height + block.y - 1) / block.y, 1); // Block Count*/


			CUSuccess(cuCtxSynchronize());

			CUSuccess(cuEventRecord(start, 0));
			CUSuccess(cuEventRecord(startblocking, 0));
			CUSuccess(cuLaunchKernel(pKernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_memorysize, NULL, args, 0));
			CUSuccess(cuEventRecord(stop, 0));

			CUSuccess(cuCtxSynchronize());

			cuda_render_texture->CopyToDirectXAsync();
		}
		frame_number++;
	}

	inline float CudaFrametime()
	{
		if (cuda_completed_frame_count < 1)
			return 0.0f;

		return cuda_frametimes[(cuda_completed_frame_count - 1) % 30];
	}

	void Reset()
	{
		CUSuccess(cuCtxSynchronize());
		if (cuda_render_texture) {
			cuda_render_texture->Unbind();
			cuda_render_texture->Release();
		}

		ImGui_ImplDX9_InvalidateDeviceObjects();
		HRESULT hr = d3dd->Reset(&d3dpp);
		if (hr == D3DERR_INVALIDCALL)
			IM_ASSERT(0);

		if (cuda_render_texture) {
			cuda_render_texture->Init();
			cuda_render_texture->Bind();
		}

		ImGui_ImplDX9_CreateDeviceObjects();
	}
};
D3DCudaDevice device;


static i64 GlobalPerfCountFrequency;
static inline LARGE_INTEGER Win32GetWallClock(void)
{
	LARGE_INTEGER Result;
	QueryPerformanceCounter(&Result);
	return(Result);
}

static inline comfy::real Win32GetSecondsElapsed(LARGE_INTEGER Start, LARGE_INTEGER End)
{
	comfy::real Result = ((comfy::real)(End.QuadPart - Start.QuadPart) /
		(comfy::real)GlobalPerfCountFrequency);
	return(Result);
}

void ThreadProc()
{
	// Initialite CUDA Api so we can ask about the graphics adapter compatbility
	CUSuccess(cuInit(0));

	::device.Create((HWND)state::windowhandle);
	::device.InitResources();

	::ShowWindowAsync((HWND)state::windowhandle, SW_SHOWDEFAULT);
	::UpdateWindow((HWND)state::windowhandle);

	// Setup Dear ImGui context
	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io; // ???
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;	   // Enable Keyboard Controls
		//io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;		   // Enable Docking
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;		 // Enable Multi-Viewport / Platform Windows
		io.ConfigViewportsNoTaskBarIcon = true;

		ImGui::StyleColorsDark();

		// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
		ImGuiStyle& style = ImGui::GetStyle();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			style.WindowRounding = 0.0f;
			style.Colors[ImGuiCol_WindowBg].w = 1.0f;
		}
		ImGui_ImplWin32_Init((HWND)state::windowhandle);
		ImGui_ImplDX9_Init(::device.d3dd);
	}

	comfy::Comfy((ID3DCudaDevice *)&::device);

	LARGE_INTEGER LastCounter = Win32GetWallClock();

	LARGE_INTEGER PerfCountFrequencyResult;
	QueryPerformanceFrequency(&PerfCountFrequencyResult);
	GlobalPerfCountFrequency = PerfCountFrequencyResult.QuadPart;

	MSG msg;
	ZeroMemory(&msg, sizeof(msg));
	while (g_running) {
		if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
			if (msg.message == WM_SIZE)
				::device.TryResize(LOWORD(msg.lParam), HIWORD(msg.lParam));
			else { // Imgui
				::TranslateMessage(&msg);
				::DispatchMessage(&msg);
			}
			continue;
		}

		comfy::Update();
		//comfy::state::KeyFrame();

		LARGE_INTEGER EndCounter = Win32GetWallClock();
		comfy::state::frametime = Win32GetSecondsElapsed(LastCounter, EndCounter);
		LastCounter = EndCounter;
	}

	ImGui_ImplDX9_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();

	::device.Cleanup();
}

static std::thread* grender_thread;
/* @TODO Fix PrintF from GPU and Output pipe from compiler
int WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     lpCmdLine,
	int       nShowCmd)
{

	string cl = lpCmdLine;
	for (size_t i = 0; i < cl.size(); i++)
	{
		if (cl[i] == '-') {
			size_t startpos = i + 1;

			do
			{
				i++;
			} while (i < cl.size() && IsCharAlphaNumericA(cl[i]));

			string key = cl.substr(startpos, i - startpos);
			int value = 1;
			if (cl[i] == '=') {
				startpos = i + 1;
				do
				{
					i++;
				} while (i < cl.size() && IsCharAlphaNumericA(cl[i]));
				sscanf(&cl[startpos], "%d", &value);
			}

			state::launchparams[key] = value;
		}
	}*/

int main() {

	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL),
		NULL, NULL, NULL, NULL, _T("Comfy Rays"), NULL };
	::RegisterClassEx(&wc);

	state::windowhandle = ::CreateWindow(wc.lpszClassName, _T("Comfy"), WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, 1280, 720, NULL, NULL, wc.hInstance, NULL);

	RAWINPUTDEVICE Rid[2];
	Rid[0].usUsagePage = 0x01; // HID_USAGE_PAGE_GENERIC;
	Rid[0].usUsage = 0x02; // HID_USAGE_GENERIC_MOUSE;
	Rid[0].dwFlags = RIDEV_INPUTSINK;
	Rid[0].hwndTarget = (HWND)state::windowhandle;

	Rid[1].usUsagePage = 0x01; // HID_USAGE_PAGE_GENERIC;
	Rid[1].usUsage = 0x06; // HID_USAGE_GENERIC_KEYBOARD;
	Rid[1].dwFlags = RIDEV_INPUTSINK;
	Rid[1].hwndTarget = (HWND)state::windowhandle;

	RegisterRawInputDevices(Rid, 2, sizeof(RAWINPUTDEVICE));

	std::thread render_thread(ThreadProc);
	grender_thread = &render_thread;

	MSG msg;
	ZeroMemory(&msg, sizeof(msg));
	while (g_running)
	{
		if (::PeekMessage(&msg, (HWND)state::windowhandle, 0U, 0U, PM_REMOVE)) {
			switch (msg.message)
			{
				case WM_QUIT:
				{
					g_running = false;
				} break;
			}

			::TranslateMessage(&msg);
			::DispatchMessage(&msg);
			continue;
		}
	}

	DebugPrint("Joining with Main Thread");
	render_thread.join();

	/*
	ImGui_ImplDX9_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();

	device.Cleanup();*/
	::DestroyWindow((HWND)state::windowhandle);
	::UnregisterClass(wc.lpszClassName, wc.hInstance);

	return 0;
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
		return true;

	switch (msg)
	{
		case WM_INPUT:
		{
			static RAWINPUT raw;
			UINT dwSize = sizeof(RAWINPUT);

			GetRawInputData((HRAWINPUT)lParam, RID_INPUT,
				&raw, &dwSize, sizeof(RAWINPUTHEADER));

			if (raw.header.dwType == RIM_TYPEMOUSE) {
				auto mm = raw.data.mouse;
				if ((mm.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN) == RI_MOUSE_LEFT_BUTTON_DOWN) {
					state::key[0x1] |= state::PRESS | state::DOWN;
				}
				if ((mm.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN) == RI_MOUSE_RIGHT_BUTTON_DOWN) {
					state::key[0x2] |= state::PRESS | state::DOWN;
				}
				if ((mm.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP) == RI_MOUSE_LEFT_BUTTON_UP) {
					state::key[0x1] = state::RELEASE;
				}
				if ((mm.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP) == RI_MOUSE_RIGHT_BUTTON_UP) {
					state::key[0x2] = state::RELEASE;
				}
				
				state::mousemove.x += float(mm.lLastX);
				state::mousemove.y += float(mm.lLastY);
			}
			else if (raw.header.dwType == RIM_TYPEKEYBOARD) {
				auto kb = raw.data.keyboard;

				if ((kb.Flags & RI_KEY_MAKE) == RI_KEY_MAKE) // Down
				{
					state::key[byte(kb.VKey)] |= state::PRESS | state::DOWN;
				}
				if ((kb.Flags & RI_KEY_BREAK) == RI_KEY_BREAK) // UP
				{
					state::key[byte(kb.VKey)] = state::RELEASE;
				}
			}
		} break;
		case WM_ACTIVATE:
		{
			if(LOWORD(wParam) == WA_INACTIVE)
				state::isfocus = 0;
			else
				state::isfocus = 1;
		} break;
		case WM_SETFOCUS:
		{
			state::isfocus = 1;
		} break;
		case WM_KILLFOCUS:
		{
			state::isfocus = 0;
		} break;
		case WM_SIZE:
		{
			// Forward to render thread
			if(wParam != SIZE_MINIMIZED)
				PostThreadMessage(*(DWORD*)&grender_thread->get_id(), WM_SIZE, wParam, lParam);

			return 0;
		} break;
		case WM_SYSCOMMAND:
		{
			if ((wParam & 0xfff0) == SC_KEYMENU)
				return 0;
		} break;
		case WM_DESTROY:
		{
			g_running = false;
			::PostQuitMessage(0);
			return 0;
		} break;
		/*case WM_DPICHANGED: {
			if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DpiEnableScaleViewports)
			{
				const RECT* suggested_rect = (RECT*)lParam;
				::SetWindowPos(hWnd, NULL, suggested_rect->left, suggested_rect->top, suggested_rect->right - suggested_rect->left,
					suggested_rect->bottom - suggested_rect->top, SWP_NOZORDER | SWP_NOACTIVATE);
			}
		} break;*/
		default: {
		} break;
	}
	return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

