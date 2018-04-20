#include "process.cuh"
//#include <iostream>

extern "C"

//process 1 pixels per thread
__global__
void changeBightnessSaturation(unsigned char *inData_d, unsigned char* outData_d, const float sat, const float bright,  const int imgWidth, const int imgHeight )
{
    //Thread index
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(xIndex>imgWidth || yIndex>imgHeight)    //Only valid threads perform memory I/O
        return;

    const int step=imgWidth*3;
    const int rgb_id = yIndex * step + (3*xIndex);     //index for RGB channel array

    unsigned char *inPixelPtr = &inData_d[rgb_id];
    unsigned char *outPixelPtr = &outData_d[rgb_id];
    int R, G ,B;

    R = inPixelPtr[0] * bright;
    G = inPixelPtr[1] * bright;
    B = inPixelPtr[2] * bright;

    //Weight for saturation
    const double  W=sqrt(R*R *0.299 + G*G*0.587 + B*B*0.114) ;

    R = W + (R - W) * sat;
    G = W + (G - W) * sat;
    B = W + (B - W) * sat;

    //Clamp  0 to 255 and write to output
    outPixelPtr[0]=(R <0)?0:((R>255)?255:R);
    outPixelPtr[1]=(G<0)?0:((G>255)?255:G);
    outPixelPtr[2]=(B<0)?0:((B>255)?255:B);

    //Move pointers to next pixel
    inPixelPtr=inPixelPtr+3*(blockDim.x);
    outPixelPtr=outPixelPtr+3*(blockDim.x);

}

// Kernel laucher
void cudaProcessImage(unsigned char* in_d, unsigned char* out_d, const float sat, float bright,  int width, int height)
{
    const dim3 block(width/4,1);    //should be able to process up to 4K videos
    const dim3 grid(4,height);

    changeBightnessSaturation <<< grid, block >>>(in_d, out_d, sat, bright, width, height);

}
