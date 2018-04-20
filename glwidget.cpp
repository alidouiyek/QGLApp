#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "glwidget.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "process.cuh"

GLWidget::GLWidget(QWidget *parent, cv::Size imageSize) : QGLWidget(parent)
{
    W=imageSize.width;
    H=imageSize.height;
}

GLWidget::~GLWidget()
{
    glutExit();
   std::cout<<"Cleaning GLWidget"<<std::endl<<std::flush;
    if (pixelBufferObject)
    {
        cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
        cudaGraphicsUnregisterResource(cudaPboResource);
        glDeleteBuffers(1, &pixelBufferObject);
        glDeleteTextures(1, &texture);
    }

    if(inputPtr_d!=NULL)
        SAFE_CUDA(cudaFree(inputPtr_d),"Error cudaFree inputPtr_d");

//  Ouput pointer is already freed by cudaGraphics  unmap and unregister
//    if(outputPtr_d!=NULL)
//        SAFE_CUDA(cudaFree(outputPtr_d),"Error cudaFree outputPtr_d"); //seems to be freed by the

}

void GLWidget::setSaturation(float sat)
{
    this->saturationLevel=sat;
}

//Static function that calls the cuda processing function
void GLWidget::process(uchar* in, uchar* out, float sat, int w,int h)
{
     cudaProcessImage(in, out, sat, w, h);
}

// Init glut, glew and cuda
void GLWidget::initializeGL()
{
    if(inputPtr_d!=NULL)
        SAFE_CUDA(cudaFree(inputPtr_d),"Error cudaFree inputPtr_d");
    size_t byteSize = W*H * 3;
    SAFE_CUDA(cudaMalloc((void **)&inputPtr_d, byteSize), "Error cudaMalloc inputPtr_d");

    //No need to allocate output pointer because cudaGraphics functions below will allocate it
//    if(outputPtr_d!=NULL)
//        SAFE_CUDA(cudaFree(outputPtr_d),"Error cudaFree outputPtr_d");
//  SAFE_CUDA(cudaMalloc((void **)&outputPtr_d, byteSize), "Error cudaMalloc outputPtr_d");

    makeCurrent();

    int c=1;    char* v = "";
    glutInit( &c, &v );
    glewInit();
    gluOrtho2D(0, W,H, 0);

    glGenBuffers(1, &pixelBufferObject);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObject);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * sizeof(GLubyte) * W*H, 0, GL_STREAM_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaPboResource, pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsMapResources(1, &cudaPboResource, 0);    // map PBO to get CUDA device pointer
    cudaGraphicsResourceGetMappedPointer((void **)&outputPtr_d, 0, cudaPboResource);

    glGenTextures  (1, &texture);
    glBindTexture  (GL_TEXTURE_2D, texture);
    glTexImage2D   (GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,  GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture  (GL_TEXTURE_2D, 0);

    glPixelStorei  (GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei  (GL_PACK_ALIGNMENT, 1);
}

//Resize video to fit the viewer
void GLWidget::resizeGL(int w, int h)
{
    const float srcAspect = (float) W / (float) H;
    const float windowAspect = (float) w / (float) h;

    float scaleW = (float) w / (float) W;
    float scaleH = (float) h / (float) H;
    if (windowAspect > srcAspect) {
        scaleW = scaleH;
    } else {
        scaleH = scaleW;
    }

    float margin_x = (w -W * scaleW) / 2;
    float margin_y = (h - H * scaleH) / 2;

    glViewport(margin_x, margin_y, W * scaleW, H * scaleH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, W / srcAspect, H / srcAspect, 0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}

//Copy input image  from host memory to device memory
void GLWidget::updateImage(uchar* ptr_h)
{
    if(ptr_h && inputPtr_d &&  outputPtr_d)
    {
        if(saturationLevel==1.0) // copy directly to output pointer, there is no need to apply cuda processing
            cudaMemcpy(outputPtr_d, ptr_h,3*W*H*sizeof(uchar),cudaMemcpyHostToDevice);
        else
        {
            cudaMemcpy(inputPtr_d, ptr_h,3*W*H*sizeof(uchar),cudaMemcpyHostToDevice);
            process(inputPtr_d, outputPtr_d, saturationLevel, W,H);
        }
        paintGL();
    }
    else
        std::cout<<"Pointer empty"<<std::endl;

}

//Draw the image
void GLWidget::paintGL()
{
    makeCurrent();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);

    size_t num_bytes;
    cudaGraphicsMapResources(1, &cudaPboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&outputPtr_d, &num_bytes, cudaPboResource);
    cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObject);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,GL_BGR, GL_UNSIGNED_BYTE, (char *)NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBegin(GL_QUADS);
    glVertex2f(-1, -1); glTexCoord2f(0, 0);
    glVertex2f(-1,  1); glTexCoord2f(1, 0);
    glVertex2f( 1,  1); glTexCoord2f(1, 1);
    glVertex2f( 1, -1); glTexCoord2f(0, 1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    swapBuffers();
}
