#include "process.cuh"
//#include <iostream>

extern "C"

//process 4 pixels per thread
__global__
void changeSaturation(unsigned char *inData_d, unsigned char* outData_d, const int imgWidth, const int imgHeight , const float sat)
{
    //Thread index
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(xIndex>imgWidth/4 || yIndex>imgHeight)    //Only valid threads perform memory I/O
        return;

    const int step=imgWidth*3;
    const int rgb_id = yIndex * step + (3*xIndex);     //index for RGB channel array

    unsigned char *inPixelPtr = &inData_d[rgb_id];
    unsigned char *outPixelPtr = &outData_d[rgb_id];
    unsigned char R, G ,B;

     for(int i=0;i<4;i++)
    {
        R = inPixelPtr[0];
        G = inPixelPtr[1];
        B = inPixelPtr[2];

        //Weight for saturation
        const double  W=sqrt(R*R *0.299 + G*G*0.587 + B*B*0.114) ;

        int vR = W + (R - W) * sat;
        int vG = W + (G - W) * sat;
        int vB = W + (B - W) * sat;

        //Clamp  0 to 255 and write to output
        outPixelPtr[0]=(vR <0)?0:((vR>255)?255:vR);
        outPixelPtr[1]=(vG<0)?0:((vG>255)?255:vG);
        outPixelPtr[2]=(vB<0)?0:((vB>255)?255:vB);

        //Move pointers to next pixel
        inPixelPtr=inPixelPtr+3*(blockDim.x);
        outPixelPtr=outPixelPtr+3*(blockDim.x);
    }

}

// Kernel laucher
void cudaProcessImage(unsigned char* in_d, unsigned char* out_d, const float sat, const int width, const int height)
{
    const dim3 block(width/4,1);
    const dim3 grid(1,height);

    changeSaturation <<< grid, block >>>(in_d, out_d, width, height, sat);

}
