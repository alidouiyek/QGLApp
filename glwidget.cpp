#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "glwidget.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "process.cuh"

GLWidget::GLWidget(QWidget *parent) : QGLWidget(parent)
{

}

GLWidget::~GLWidget()
{
    if (pixelBufferObject)
    {
        cudaGraphicsUnmapResources(1, &cudaPboResource, 0);
        cudaGraphicsUnregisterResource(cudaPboResource);
        glDeleteBuffers(1, &pixelBufferObject);
        glDeleteTextures(1, &texture);
    }

    if(inputPtr_d!=NULL)
        cudaFree(inputPtr_d);
    if(outputPtr_d!=NULL)
        cudaFree(outputPtr_d);
}

void GLWidget::setImageSize(cv::Size size)
{
        W=size.width;
        H=size.height;
        printf("Input Size: %ix%i  \n", W, H);

        if(inputPtr_d!=NULL)
            cudaFree(inputPtr_d);
        if(outputPtr_d!=NULL)
            cudaFree(outputPtr_d);

        size_t byteSize = W*H * 3;
        cudaMalloc((void **)&inputPtr_d, byteSize);
        cudaMalloc((void **)&outputPtr_d, byteSize);
}

void GLWidget::setSaturation(float sat)
{
    this->saturationLevel=sat;
}

void GLWidget::process(uchar* in, uchar* out, float sat, int w,int h)
{
     cudaProcessImage(in, out, sat, w, h);
}


void GLWidget::initializeGL()
{
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

void GLWidget::resizeGL(int w, int h)
{
    printf("Resizing %i x %i", w,h);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h); // set origin to bottom left corner
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}

void GLWidget::updateImage(uchar* ptr_h)
{
    if(ptr_h && inputPtr_d ){
        cudaMemcpy(inputPtr_d, ptr_h,3*W*H*sizeof(uchar),cudaMemcpyHostToDevice);
        process(inputPtr_d, outputPtr_d, saturationLevel, W,H);
    }
    else{
        std::cout<<"No image"<<std::endl;
    }
    paintGL();

}


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
