#pragma once
#include "shared.hpp"

#define NOMINMAX
#define LEAN_AND_MEAN
#include <windows.h>

#include <cmath>
#include <inttypes.h>
#include <mutex>

#include <strsafe.h>
#include <DbgHelp.h>

#define EASTL_EASTDC_VSNPRINTF 0
inline int Vsnprintf8(char* pDestination, size_t n, const char* pFormat, va_list arguments) {
    return vsnprintf(pDestination, n, pFormat, arguments);
}
#include <EASTL/array.h>
#include <EASTL/vector.h>
#include <EASTL/string.h>
#include <EASTL/hash_map.h>
#include <string>

#define COMFY_FAIL() { __debugbreak(); }
#define LOG() {  }
#define CTAssert(Expr) static_assert(Expr, "Assertion failed: " #Expr)
#define Assert(Expression) if(!(Expression)) {*(volatile int *)0 = 0;}

namespace comfy {
    class ID3DCudaDevice;

    void Update();
    void Comfy(ID3DCudaDevice * dev);
}

// Debugs
namespace comfy
{
    inline u32 GetThreadID(void)
    {
        byte *ThreadLocalStorage = (byte *)__readgsqword(0x30);
        u32 ThreadID = *(u32 *)(ThreadLocalStorage + 0x48);
        return(ThreadID);
    }
    inline void DebugPrint(const char* m)
    {
        OutputDebugString(m);
    }
    inline void DebugPrintLine(const char* m)
    {
        OutputDebugString(m);
        OutputDebugString("\r\n");
    }
    inline void DebugPrintF(const char* m, ...)
    {
		static char buffer[2048];
		va_list argptr;
		va_start(argptr, m);
		vsprintf_s(buffer, m, argptr);
		va_end(argptr);
		OutputDebugString(buffer);
		OutputDebugString("\r\n");
    }
    inline void DebugPrint(const wchar_t* m)
    {
        OutputDebugStringW(m);
    }
    inline void DebugPrintLine(const wchar_t* m)
    {
        OutputDebugStringW(m);
        OutputDebugStringW(L"\r\n");
    }
    inline void DebugPrintF(const wchar_t* m, ...)
    {
        static wchar_t buffer[2048];
        va_list argptr;
        va_start(argptr, m);
        vswprintf_s(buffer, m, argptr);
        va_end(argptr);
        OutputDebugStringW(buffer);
        OutputDebugStringW(L"\r\n");
    }
}

namespace comfy
{
    using eastl::array;
    using eastl::vector;
    typedef std::string stdstring;
    typedef std::wstring stdwstring;
    typedef eastl::string string;
    typedef eastl::wstring wstring;
    using eastl::hash_map;
}