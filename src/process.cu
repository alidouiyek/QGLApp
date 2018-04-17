#include "process.cuh"
//#include <iostream>

extern "C"

__device__
void saturate3Pixels(unsigned char* inPtr_d, unsigned char* outPtr_d, int step, float sat)
{
    const float wR=0.299;
    const float wG=0.587;
    const float wB=0.114;

    for(int i=0;i<4;i++)
    {
        //Pixel 1
        unsigned char R= inPtr_d[0];
        unsigned char G= inPtr_d[1];
        unsigned char B= inPtr_d[2];

        double  P=sqrt(R*R *wR + G*G*wG + B*B*wB) ;

        int vR=P+(R-P)*sat;
        int vG=P+(G-P)*sat;
        int vB=P+(B-P)*sat;

        vR=(vR<0)?0:vR;
        vG=(vG<0)?0:vG;
        vB=(vB<0)?0:vB;
        vR=(vR>255)?255:vR;
        vG=(vG>255)?255:vG;
        vB=(vB>255)?255:vB;

        outPtr_d[0]=vR;
        outPtr_d[1]=vG;
        outPtr_d[2]=vB;

        inPtr_d=inPtr_d+3*(step);
        outPtr_d=outPtr_d+3*(step);
    }
}

__global__
void changeSaturation(unsigned char *inData_d, unsigned char* outData_d, int imgWidth, int imgHeight , float sat) {

    //Only valid threads perform memory I/O

    int inStep=imgWidth*3;
    int xIndex = (blockIdx.x * blockDim.x + threadIdx.x);
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(xIndex>imgWidth/4 || yIndex>imgHeight)
        return;

    const int rgb_id = yIndex * inStep + (3*xIndex);     //index for RGB channel array

    unsigned char *iPtr = &inData_d[rgb_id];
    unsigned char *oPtr = &outData_d[rgb_id];

    saturate3Pixels(iPtr, oPtr,(blockDim.x),sat);
}

void cudaProcessImage(unsigned char* in_d, unsigned char* out_d, float sat, int width, int height)
{

    const dim3 block(width/4,1);
    const dim3 grid(1,height);

    changeSaturation <<< grid, block >>>(in_d, out_d, width, height, sat);

}
