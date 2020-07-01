@echo off
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx86\x64
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin
nvcc --ptxas-options=-v -maxrregcount=0 -src-in-ptx  -use_fast_math -lineinfo -O3 -arch=sm_61 -res-usage -cubin -o data\kernel.cubin "C:\source\ComfyMath\src\comfy.cu"

rem nvcc -src-in-ptx  -v -O0 --debug -Xptxas --device-debug -restrict -use_fast_math -arch=sm_61 -res-usage -cubin -o x64\FastBuild\comfy.cu.cubin "C:\source\ComfyMath\src\comfy.cu"