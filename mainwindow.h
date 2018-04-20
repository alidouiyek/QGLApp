#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMessageBox>
#include <QFileDialog>
#include <QDir>
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

    void updateTimeLabel();

private slots:
    void on_browseButton_clicked();
    void on_backButton_clicked();
    void on_forwardButton_clicked();
    void on_pauseButton_clicked();

    void on_progressSlider_sliderMoved(int position);
    void on_satSlider_sliderMoved(int position);

    void on_grabFrame();
    void on_updateGui();


private:
    Ui::MainWindow *ui;
    GLWidget* videoWidget=NULL;

    cv::VideoCapture cap;
    cv::Mat imgSrc;
    QTimer grabFramesTimer;
    QTimer updateGuiTimer;

    float ellapsedTimeMs=0;
    float frameDurationMs=55;
};

#endif // MAINWINDOW_H
