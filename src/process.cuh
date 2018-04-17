#ifndef PROCESS_CUH
#define PROCESS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

 void cudaProcessImage(unsigned char* in_d, unsigned char* out_d, float sat, int width, int height);

#endif // PROCESS_CUH
