#pragma once
#define LEAN_AND_MEAN
#include <windows.h>

#include <d3d9.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cudaD3D9.h>

#include "host.hpp"
#include "platform.hpp"

#define NVCRTSuccess(x) \
	do { \
		nvrtcResult __result = x; \
		if (__result != NVRTC_SUCCESS) { \
			comfy::DebugPrintF("error: %s failed with %s", #x, nvrtcGetErrorString(__result)); \
			COMFY_FAIL(); \
		} \
	} while (0)

#define CUSuccess(x) \
	do { \
		CUresult __result = x; \
		if (__result != CUDA_SUCCESS) { \
			const char* __msg; \
			cuGetErrorName(__result, &__msg); \
			comfy::DebugPrintF("error: %s failed with %s", #x, __msg); \
			COMFY_FAIL(); \
		} \
	} while (0)


static inline void format_winerr_break(LPCSTR msg)
{
	LPVOID lpMsgBuf;
	LPVOID lpDisplayBuf;
	DWORD dw = GetLastError();

	FormatMessage(
		FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);

	lpDisplayBuf = (LPVOID)LocalAlloc(
		LMEM_ZEROINIT, (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)msg) + 40) * sizeof(TCHAR));
	StringCchPrintf((LPTSTR)lpDisplayBuf, LocalSize(lpDisplayBuf) / sizeof(TCHAR),
		TEXT("%s failed with error %d: %s"), msg, dw, lpMsgBuf);

	OutputDebugString((LPCTSTR)lpDisplayBuf);

	LocalFree(lpMsgBuf);
	LocalFree(lpDisplayBuf);
	DebugBreak();
}

static inline comfy::u64 GetWriteTime(const comfy::string filename)
{
	FILETIME ft_write;
	HANDLE hFile = CreateFile(
			filename.c_str(), GENERIC_READ,
		FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, // Don't block the IDE, we'll recover from these events
			NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

    if (hFile == INVALID_HANDLE_VALUE)
		return 0; // // Fail silently format_winerr_break("CreateFile");

	if (GetFileTime(hFile, NULL, NULL, &ft_write))
	{
		ULARGE_INTEGER write_time;
		write_time.LowPart = ft_write.dwLowDateTime;
		write_time.HighPart = ft_write.dwHighDateTime;
		CloseHandle(hFile);
		return write_time.QuadPart;
	}

	CloseHandle(hFile);
	return 0;
}


// Buffer allocated -> malloc(fileSize+1) // Extra trailing zero
static inline char * GetFile(const comfy::string filename)
{
	char* buffer = NULL;
	{
		HANDLE h = CreateFile(
			filename.c_str(), GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
			NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

		if (h == INVALID_HANDLE_VALUE)
			return NULL;

		DWORD fileSize = GetFileSize(h, NULL);
		DWORD bytesRead = 0;
		buffer = (char*)malloc(fileSize+1);

		if (!ReadFile(h, buffer, fileSize, &bytesRead, NULL))
			format_winerr_break("ReadFile");

		buffer[fileSize] = '\0';
		CloseHandle(h);
	}

	return buffer;
}

static HANDLE g_hChildStd_OUT_Rd = NULL;
static inline void ReadFromPipe(void)
{
	static const size_t BUFSIZE = 1024;
	DWORD dwRead, dwWritten;
	CHAR chBuf[BUFSIZE] = { 0 };
	BOOL bSuccess = FALSE;
	HANDLE hParentStdOut = GetStdHandle(STD_OUTPUT_HANDLE);

	for (;;)
	{
		bSuccess = ReadFile(g_hChildStd_OUT_Rd, chBuf, BUFSIZE, &dwRead, NULL);
		
		if (!bSuccess || dwRead == 0) 
			break;

		chBuf[dwRead - 1] = '\0';
		::OutputDebugStringA(chBuf);

		bSuccess = WriteFile(hParentStdOut, chBuf,
			dwRead, &dwWritten, NULL);
		if (!bSuccess) break;
	}
}

static inline void RunAndPipeToDebugPrint(const comfy::string command)
{
	HANDLE g_hChildStd_OUT_Wr = NULL;

	SECURITY_ATTRIBUTES saAttr;
	saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	saAttr.bInheritHandle = TRUE;
	saAttr.lpSecurityDescriptor = NULL;

	if (!CreatePipe(&g_hChildStd_OUT_Rd, &g_hChildStd_OUT_Wr, &saAttr, 0))
		COMFY_FAIL();
	if (!SetHandleInformation(g_hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0))
		COMFY_FAIL();

	PROCESS_INFORMATION piProcInfo;
	ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

	STARTUPINFO siStartInfo;
	ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
	siStartInfo.cb = sizeof(STARTUPINFO);
	siStartInfo.hStdError = g_hChildStd_OUT_Wr;
	siStartInfo.hStdOutput = g_hChildStd_OUT_Wr;
	siStartInfo.hStdInput = NULL;
	siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

	BOOL bSuccess = CreateProcessA(NULL,
		(char*)command.c_str(),     // command line 
		NULL,          // process security attributes 
		NULL,          // primary thread security attributes 
		TRUE,          // handles are inherited 
		0,             // creation flags 
		NULL,          // use parent's environment 
		NULL,          // use parent's current directory 
		&siStartInfo,  // STARTUPINFO pointer 
		&piProcInfo);  // receives PROCESS_INFORMATION 

	 // If an error occurs, exit the application. 
	if (!bSuccess) {
		COMFY_FAIL();
	}
	else
	{
		CloseHandle(piProcInfo.hProcess);
		CloseHandle(piProcInfo.hThread);
		CloseHandle(g_hChildStd_OUT_Wr);
	}
	ReadFromPipe(); //@TODO

	
}

inline static bool FileExists(const comfy::string filename)
{
	HANDLE h = CreateFile(
		filename.c_str(), GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
		NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

	if (h == INVALID_HANDLE_VALUE) 
		return false;
	
	CloseHandle(h);

	return true;
}

inline int GetCubin(const comfy::string filename, CUmodule* cumodule_out)
{
	char* buffer = NULL;
	{
		HANDLE h = CreateFile(
			filename.c_str(), GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
			NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

		if (h == INVALID_HANDLE_VALUE) {
			format_winerr_break("CreateFile");
			return -1;
		}

		DWORD fileSize = GetFileSize(h, NULL);
		DWORD bytesRead = 0;
		buffer = (char*)malloc(fileSize + 1);

		if (!ReadFile(h, buffer, fileSize, &bytesRead, NULL)) {
			format_winerr_break("ReadFile");
			return -1;
		}

		buffer[fileSize] = '\0';
		CloseHandle(h);
	}
	CUSuccess(cuModuleLoadData(cumodule_out, buffer));
	comfy::DebugPrintF("Loaded Cuda Modules %s", &filename[0]);

	return 0;
}

/*
struct _cudamodule
{
	comfy::u64 last_write_time = 0;
	CUmodule pModule = NULL;

	const comfy::vector<const char *> deps = {"src/comfy.cuh", "src/cugar.hpp", "src/bsp.cuh" };

	CUmodule Load(const char* filename, const char* sourcename)
	{
		if (GetWriteTime(sourcename) <= last_write_time)
			return pModule;

		//magic("compile_shader.bat");

		char* buffer = NULL;
		{
			HANDLE h = CreateFile(
				filename, GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
				NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

			if (h == INVALID_HANDLE_VALUE) {
				format_winerr_break("CreateFile");
				return pModule;
			}
		
			DWORD fileSize = GetFileSize(h, NULL);
			DWORD bytesRead = 0;
			buffer = (char*)malloc(fileSize + 1);

			if (!ReadFile(h, buffer, fileSize, &bytesRead, NULL)) {
				format_winerr_break("ReadFile");
				return pModule;
			}

			buffer[fileSize] = '\0';
			CloseHandle(h);
		}
		CUSuccess(cuModuleLoadFatBinary(&pModule, buffer));
		comfy::DebugPrintF("Loaded Cuda Modules %s", filename);

		last_write_time = GetWriteTime(sourcename);

		return pModule;
	}

	// Can return null
	CUmodule LoadAndCompile(const char* filename, const comfy::vector<const char*>& params)
	{
		if(GetWriteTime(filename) <= last_write_time)
			return pModule;

		char* buffer = NULL;
		{
			HANDLE h = CreateFile(
				filename, GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
				NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

			if (h == INVALID_HANDLE_VALUE)
				return pModule;
				//format_winerr_break("CreateFile");

			DWORD fileSize = GetFileSize(h, NULL);
			DWORD bytesRead = 0;
			buffer = (char*)malloc(fileSize+1);

			if (!ReadFile(h, buffer, fileSize, &bytesRead, NULL))
				format_winerr_break("ReadFile");

			buffer[fileSize] = '\0';
			CloseHandle(h);
		}



		comfy::vector<char*> headerdata;
		for (const char* dep : deps)
		{
			char * depdata = GetFile(dep);
			if (depdata == NULL)
			{
				comfy::DebugPrintF("Failed reading dependancy %s", dep);
				return pModule;
			}
			headerdata.push_back(depdata);
		}


		nvrtcProgram prog = NULL;
		char* ptx = NULL;
		{

			NVCRTSuccess(nvrtcCreateProgram(&prog, 				// prog
				buffer,											// buffer
				"comfy.cu",						   				// name
				int(headerdata.size()),							// numHeaders
				headerdata.data(),								// headers
				deps.data()));							   		// includeNames

			comfy::DebugPrintF("\r\nStarting Compile for %s", filename);
			comfy::DebugPrintLine("----------------------------------------");

			nvrtcResult compileResult = nvrtcCompileProgram(prog,   	// prog
				int(params.size()),									  	// numOptions
				params.data());											// options

			size_t logSize = 0;
			NVCRTSuccess(nvrtcGetProgramLogSize(prog, &logSize));

			char* log = new char[logSize];
			NVCRTSuccess(nvrtcGetProgramLog(prog, log));
			comfy::DebugPrint(log);
			delete[] log;

			if (compileResult == NVRTC_SUCCESS)
			{
				size_t ptxSize;
				NVCRTSuccess(nvrtcGetPTXSize(prog, &ptxSize));

				ptx = new char[ptxSize]; //@LEAK

				NVCRTSuccess(nvrtcGetPTX(prog, ptx));
				NVCRTSuccess(nvrtcDestroyProgram(&prog));

				CUSuccess(cuModuleLoadDataEx(&pModule, ptx, 0, 0, 0));
				comfy::DebugPrintF("Loaded Cuda Modules %s", filename);
			}

			comfy::DebugPrintLine("----------------------------------------");
			comfy::DebugPrint("End Compile\r\n\r\n");

			last_write_time = GetWriteTime(filename);
		}

		for (char * headercontent : headerdata)
			free(headercontent);

		return pModule;
	}
};
*/