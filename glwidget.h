#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <GL/freeglut.h>
#include <QtOpenGL/QGLWidget>
#include <opencv2/opencv.hpp>

class GLWidget : public QGLWidget {

    Q_OBJECT // must include this if you use Qt signals/slots

public:
    GLWidget(QWidget *parent, cv::Size imageSize);
    ~GLWidget();

    uchar* testPtr;
    uchar* inputPtr_d=NULL;
    uchar* outputPtr_d=NULL;

    int W=0;
    int H =0;
    float saturationLevel=1.0;
    float brightnessLevel=1.0;

    GLuint texture=0;
    GLuint pixelBufferObject=0;
    struct cudaGraphicsResource *cudaPboResource;

    void setSaturation(float sat) ;
    void setBrightness(float bright) ;
    void updateImage(uchar *ptr_h) ;
    static void process(uchar* in, uchar* out, float saturation, float brightness, int width, int height) ;

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void paintGLOK();

};

#endif  /* _GLWIDGET_H */
