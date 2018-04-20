#ifndef PROCESS_CUH
#define PROCESS_CUH

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SAFE_CUDA(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
    }
}

 void cudaProcessImage(unsigned char* in_d, unsigned char* out_d, float sat, int width, int height);

#endif // PROCESS_CUH
