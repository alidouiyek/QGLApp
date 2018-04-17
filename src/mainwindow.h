#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include "glwidget.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include </usr/local/cuda-8.0/include/cuda.h>
#include </usr/local/cuda-8.0/include/cuda_gl_interop.h>
#include "/usr/local/cuda-8.0/include/cuda_runtime.h"

#define SAFE_CUDA(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)
//enum Exposure {INIT_EXPOSURE,LOW_EXPOSURE,MEDIUM_EXPOSURE,HIGH_EXPOSURE};
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
//        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define NN 2
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void allocImg(std::string path, int id);
    void timerEvent(QTimerEvent *);

    cv::VideoCapture cap;
//    cv::Mat imgSrc(cv::Size(640,480),CV_8UC3);
    cv::Mat imgSrc;
    QTimer timer;
    int frameCounter=0;

private slots:
    void on_jumpButton_clicked();

    void on_openButton_clicked();

    void on_satSlider_valueChanged(int value);

private:
    Ui::MainWindow *ui;
//    uchar* inputPtr_d=NULL;
    GLWidget* gw;
//    cv::cuda::GpuMat imgSrc_g[NN];
//    uchar* imgPtr_a_d[NN]={NULL};

};

#endif // MAINWINDOW_H
