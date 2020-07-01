#include <filesystem>

#include "host.hpp"
#include "cugar_matrix.hpp"

#include "platform.hpp"
#include "bsp.hpp"
#include "bspdef.hpp"

#include <imgui/imgui.h>
#include <imgui/imgui_impl_win32.h>
#include <imgui/imgui_impl_dx9.h>

namespace comfy {
	struct Camera
	{
		vec3 origin;
		mat4 invtransform;
		mat4 transform;

		float heading = 0.0f;
		float pitch = -COMFY_PIf*0.5f;
		float fov = 0.85f;

		float speed = 400.0f;
		float rotation_speed = 0.5f; // Degrees per DPI

		bool locked = true;

		int lastmove = 0;

		inline mat4 Rotation() {
			return cugar::rotation_around_X(pitch) * cugar::rotation_around_Z(heading);
		}

		inline vec3 Right() {
			return (vec4(1.0f, 0.0f, 0.0f, 1.0f) * Rotation()).xyz();
		}
		inline vec3 Up() {
			return (vec4(0.0f, 1.0f, 0.0f, 1.0f) * Rotation()).xyz();
		}
		inline vec3 Forward() {
			return (vec4(0.0f, 0.0f, -1.0f, 1.0f) * Rotation()).xyz();
		}

		array<vec3, 5>& Frustrum()
		{
			static array<vec3, 5> f;
			{
				const vec4 _rd = invtransform * vec4(1.0f, 0.0f, 1.0f, 1.0f);
				vec3 rd = (_rd / _rd.w).xyz();
				f[0] = normalize(cross(Up(), rd));
			}
			{
				const vec4 _rd = invtransform * vec4(-1.0f, 0.0f, 1.0f, 1.0f);
				vec3 rd = (_rd / _rd.w).xyz();
				f[1] = normalize(cross(rd, Up()));
			}
			{
				const vec4 _rd = invtransform * vec4(0.0f, -1.0f, 1.0f, 1.0f);
				vec3 rd = (_rd / _rd.w).xyz();
				f[2] = normalize(cross(Right(), rd));
			}
			{
				const vec4 _rd = invtransform * vec4(0.0f, 1.0f, 1.0f, 1.0f);
				vec3 rd = (_rd / _rd.w).xyz();
				f[3] = normalize(cross(rd, Right()));
			}
			f[4] = normalize(Forward());

			return f;
		}
	};
}

namespace comfy {
	namespace fs = std::filesystem;
	void LoadMap(stdstring name);
	void CameraUpdate();
	void DrawDiag(bool* p_open);
	void LoadMap(stdstring name);

	stdstring currentmap = "c0a0.bsp";
	const fs::path mapdir("C:\\Program Files (x86)\\Steam\\steamapps\\common\\Half-Life\\valve\\maps");

	Camera camera;
	ID3DCudaDevice* device;
	ICudaKernel* kernel;
	IBSP* bsp;
	bool RunCuda = true;

	void Update()
	{
		state::KeyFrameStart();

		device->StartFrame();

		ImGui_ImplDX9_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		static bool showoverlay = true;
		if (showoverlay)
			DrawDiag(&showoverlay);

		CameraUpdate();

		bsp->PrepareFrame(camera.origin, camera.Frustrum(), device->cuda_completed_frame_count + 1 - camera.lastmove);

		static kernelstate kstate;
		static lightcollection lc;
		memset(&kstate,	0,	sizeof(kernelstate));
		memset(&lc,		0,	sizeof(lightcollection));

		//bsp->GetVisibleEntities(camera.origin, lc);

		kstate.idx = 1;
		do {
			const bspnodef& node = bsp->nodesf[kstate.currentnode];

			i16 child0 = node.vischildren[0];
			i16 child1 = node.vischildren[1];

			if (child0 != -1) {
				if (child1 != -1) {
					kstate.stack[kstate.idx++] = child1;
					kstate.currentnode = child0;
				}
				else {
					kstate.currentnode = child0;
				}
			}
			else {
				kstate.currentnode = child1;
			}
		} while (kstate.currentnode >= 0);

		kstate.mousex = -1;
		kstate.mousey = -1;
		if (comfy::state::KeyDown(0x2) || comfy::state::KeyPressed(0x3)) {
			POINT p;
			GetCursorPos(&p);
			ScreenToClient((HWND)state::windowhandle, &p);

			kstate.mousex = p.x;
			kstate.mousey = p.y;
		}

		kstate.frameid = device->cuda_completed_frame_count + 1 - camera.lastmove;
		void* args[] = {
			&device->cuda_render_texture->cuda_device_mem,
			&bsp->gpu_bsp,
			&bsp->gpu_nodes,
			&bsp->gpu_leafs,
			&bsp->gpu_tex_tris,
			&bsp->gpu_faceinfo,
			&bsp->gpu_lightvismatrix,
			&bsp->gpu_lights,
			&camera,
			&device->cuda_render_texture->width,
			&device->cuda_render_texture->height,
			&device->cuda_render_texture->pitch,
			&kstate };

		if(RunCuda)
			device->CudaRun(
				"pathtrace",
				args,
				kernel,
				0);

		ImGui::EndFrame();
		device->EndFrame();

		if (state::launchparams["nvprof"] && i32(device->cuda_completed_frame_count) >= state::launchparams["nvprof"]) {
			exit(0);
		}

		if (!comfy::state::isfocus && !ImGui::IsAnyWindowFocused()) {
			if (comfy::state::frametime < 0.1f)
			{
				Sleep(DWORD((0.1f - comfy::state::frametime) * 1000.0f));
			}
		}
	}

	void Comfy(ID3DCudaDevice * dev)
	{
		device = dev;
		kernel = ICudaKernel::Create("comfy");

		LoadMap(currentmap);
	}

	void CameraUpdate()
	{
		
		if (comfy::state::isfocus && !camera.locked) {
			if (comfy::state::KeyDown(0x10))
				camera.speed *= 10.0f;

			if (comfy::state::KeyDown('X'))
				camera.origin += camera.Up() * state::frametime * camera.speed;
			if (comfy::state::KeyDown('C'))
				camera.origin -= camera.Up() * state::frametime * camera.speed;
			if (comfy::state::KeyDown('W'))
				camera.origin += camera.Forward() * state::frametime * camera.speed;
			if (comfy::state::KeyDown('S'))
				camera.origin -= camera.Forward() * state::frametime * camera.speed;
			if (comfy::state::KeyDown('D'))
				camera.origin += camera.Right() * state::frametime * camera.speed;
			if (comfy::state::KeyDown('A'))
				camera.origin -= camera.Right() * state::frametime * camera.speed;

			if (comfy::state::KeyDown(0x10))
				camera.speed /= 10.0f;


			if (comfy::state::KeyDown(0x1))
			{
				camera.heading += state::mousemove.x * (camera.rotation_speed / 360.0f);
				camera.pitch += state::mousemove.y * (camera.rotation_speed / 360.0f);
			}

			camera.lastmove = device->cuda_completed_frame_count;
		}

		state::mousemove.x = 0.0f;
		state::mousemove.y = 0.0f;

		mat4 p = cugar::perspective(camera.fov, float(device->cuda_render_texture->width) / float(device->cuda_render_texture->height), 0.1f, 4096.0f);
		camera.transform = p * camera.Rotation();
		cugar::invert(camera.transform, camera.invtransform);
	}

	void DrawDiag(bool* p_open)
	{
		static bool locked = true;
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImVec2 work_area_pos = viewport->GetWorkPos();
		ImVec2 work_area_size = viewport->GetWorkSize();
		ImVec2 window_pos = ImVec2(work_area_pos.x - 10.0f, work_area_pos.y + 10.0f);
		ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f);

		if(locked)
			ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
		else
			ImGui::SetNextWindowPos(window_pos, ImGuiCond_Appearing, window_pos_pivot);

		int flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBringToFrontOnFocus;

		if (locked)
			flags = flags | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_AlwaysAutoResize;

		//ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
		if (ImGui::Begin("o", p_open, flags))
		{
			static const uint nvalues = 60;
			static float values[nvalues] = {};
			values[device->cuda_completed_frame_count % nvalues] = device->CudaFrametime();

			static char overlay[64];
			uint slow_updated_offset = ((device->cuda_completed_frame_count % nvalues) / 20) * 20;
			sprintf_s(overlay, "%.1fms %.1f fps", values[slow_updated_offset], 1000.0f / values[slow_updated_offset]);

			ImGui::PlotLines(
				"", values, nvalues, device->cuda_completed_frame_count % nvalues, overlay, 0, 100.0f, ImVec2(0, 80));

			if (ImGui::BeginPopupContextWindow()) {
				if (p_open && ImGui::MenuItem("Close"))
					*p_open = false;

				if (ImGui::MenuItem("Lock"))
					locked = !locked;
				ImGui::EndPopup();
			}

			ImGui::Separator();

			ImGui::Text("Camera");
			ImGui::InputFloat3("Origin", &camera.origin.x);

			ImGui::SliderFloat("Heading", &camera.heading, -5.0f, 5.0f);
			ImGui::SliderFloat("Pitch", &camera.pitch, -5.0f, 5.0f);
			ImGui::SliderFloat("Fov", &camera.fov, 0.0f, 1.35f);
			ImGui::InputFloat("Speed", &camera.speed);
			ImGui::InputFloat("Rot Speed", &camera.rotation_speed);
			ImGui::Checkbox("Locked", &camera.locked);

			ImGui::Separator();

			ImGui::Checkbox("Run", &RunCuda);

			ImGui::Separator();

			ImGui::Text("Maps");
			if (ImGui::ListBoxHeader("", 32, 16))
			{
				for (auto p : fs::directory_iterator(mapdir)) {
					stdstring filename = p.path().filename().generic_u8string();
					if (p.path().extension() == ".bsp") {
						if (ImGui::Selectable(filename.c_str(), filename == currentmap)) {
							LoadMap(filename);
						}
					}
				}
				ImGui::ListBoxFooter();
			}
		}
		ImGui::End();
	}

	void LoadMap(stdstring name)
	{
		currentmap = name;
		DebugPrintF("Loading Map %s", currentmap.c_str());

		bsp = IBSP::FromFile(string((mapdir / currentmap).generic_string().c_str()));

		bsp->PrepareMap();
		bsp->GPUInit(device);

		
		if (currentmap == "c0a0.bsp") {
			camera.origin = vec3(2908.20923f, 3024.24683f, 574.364380f);
			camera.pitch = -1.445796250f;
			camera.heading = -2.26111007f;
		} else if (currentmap == "stalkyard.bsp") {
			camera.origin = vec3(-653.768f, 1105.183f, 96.928f);
			camera.pitch = -1.495f;
			camera.heading = -4.679f;
		}
		else {
			vec3 spawn = bsp->GetSpawn();
			camera = { spawn };
		}

		//camera.lastmove = device->cuda_completed_frame_count;

		CameraUpdate();

		bsp->PrepareFrame(camera.origin, camera.Frustrum(), device->cuda_completed_frame_count + 1);
		bsp->SyncToGPU(device);
	}
}

