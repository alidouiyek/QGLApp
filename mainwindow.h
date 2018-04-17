#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include "glwidget.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void timerEvent(QTimerEvent *);
    void updateTimeLabel();

private slots:
    void on_satSlider_valueChanged(int value);

    void on_openButton_clicked();
    void on_backButton_clicked();
    void on_forwardButton_clicked();

    void on_progressSlider_valueChanged(int value);

private:
    Ui::MainWindow *ui;
    GLWidget* gw;
    cv::VideoCapture cap;
    cv::Mat imgSrc;
    QTimer timer;
    int milliSec=0;

};

#endif // MAINWINDOW_H
