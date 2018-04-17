#ifndef _GLWIDGET_H
#define _GLWIDGET_H

#include <GL/freeglut.h>
#include <QtOpenGL/QGLWidget>
#include <opencv2/opencv.hpp>

class GLWidget : public QGLWidget {

    Q_OBJECT // must include this if you use Qt signals/slots

public:
    GLWidget(QWidget *parent);
    ~GLWidget();

    uchar* inputPtr_d;
    uchar* outputPtr_d;

    int W=1280;
    int H =720;
    float saturationLevel=1.0;

    GLuint texture=0;
    GLuint pixelBufferObject=0;
    struct cudaGraphicsResource *cudaPboResource;

    void setImageSize(cv::Size size) ;
    void setSaturation(float sat) ;
    void updateImage(uchar *ptr_h) ;
    static void process(uchar* in, uchar* out, float sat, int w,int h) ;

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void paintGLOK();

};

#endif  /* _GLWIDGET_H */
