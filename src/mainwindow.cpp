#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    gw=ui->m_glwidget;


//    ui->m_glwidget->setGeometry(0,0,imgSrc.cols,imgSrc.rows);
//    ui->m_glwidget->setFixedSize(imgSrc.cols, imgSrc.rows);

    startTimer(40);
}


void MainWindow::timerEvent(QTimerEvent *)
{
    if(!cap.isOpened())
        return;

    cap.read(imgSrc);
    frameCounter+=40;
//    cv::cvtColor(imgSrc,imgSrc,cv::COLOR_RGB2BGR);
    gw->updateImage(imgSrc.ptr());

//    cv::imshow("Em", imgSrc);    cv::waitKey(1);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::allocImg(std::string imgPath, int id)
{
 /*   imgSrc_a[id]=cv::imread(imgPath,cv::IMREAD_COLOR);

    if(!imgSrc_a[id].data)
        {std::cout<<"Source file not found "<<imgPath<<std::endl; exit(EXIT_FAILURE);}
    else
        { std::cout<<"Loaded image: "<<imgPath<<std::endl; }

    cv::cvtColor(imgSrc_a[id],imgSrc_a[id],cv::COLOR_BGR2RGB);
//    cv::resize(imgSrc_a[id],imgSrc_a[id],  cv::Size(imgSrc_a[id].cols/1.5, imgSrc_a[id].rows/1.5));
    cv::resize(imgSrc_a[id],imgSrc_a[id],  cv::Size(1000,1000));
    imgPtr_a_h[id]=imgSrc_a[id].ptr();
*/
//    SAFE_CUDA(cudaMalloc<unsigned char>(&imgPtr_d[id],  3*imgSrc_a[id].total()*sizeof(uchar)), "CUDA Malloc ptr_d Failed");

//    if(imgPtr_d[id]==NULL)
//        {std::cout<<"ptr_d alloc failed "<<id<<std::endl; exit(EXIT_FAILURE);}

//    imgSrc_a_g[id]=cv::cuda::GpuMat(imgSrc_a[id].size(),CV_8UC3,imgPtr_d[id]);
//    imgSrc_a_g[id].upload(imgSrc_a[id]);
}

void MainWindow::on_jumpButton_clicked()
{
    frameCounter+=5000;
//    CV_CAP_PROP_POS_MSEC CV_CAP_PROP_POS_AVI_RATIO
    cap.set(CV_CAP_PROP_POS_MSEC,frameCounter);
    std::cout<<frameCounter<<std::endl;
}

void MainWindow::on_openButton_clicked()
{
    std::string path=ui->pathTextBox->toPlainText().toStdString();
    if(path=="")
        return;

    if(cap.isOpened())
        cap.release();

    cap.open(path);
    cap.read(imgSrc);
    gw->setImageSize(imgSrc.size());
    frameCounter=0;
}

void MainWindow::on_satSlider_valueChanged(int value)
{
    float valf=(float)value/10;
    ui->satLabel->setText("Saturation: "+QString::number(valf));
    gw->setSaturation(valf);
}
