#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    gw=ui->m_glwidget;
}

//Read frames and update UI
void MainWindow::timerEvent(QTimerEvent *)
{
    if(!cap.isOpened())
        return;

    if(!cap.read(imgSrc))
    {
        milliSec=0;
        cap.set(CV_CAP_PROP_POS_MSEC,milliSec);
    }
    milliSec+=30;
    gw->updateImage(imgSrc.ptr());

    if(milliSec%1000 <=30)
    {
        //Time Label
        int min=milliSec/60000;
        int sec=(milliSec%60000)/1000;
        QString textTime=((min<10)?"0":"")+QString::number(min)+":"+((sec<10)?"0":"")+QString::number(sec);
        ui->timeLabel->setText(textTime);

        //Progress bar
        float lenght=cap.get(CV_CAP_PROP_FRAME_COUNT)*30;
        float current= 100* milliSec/lenght;
        ui->progressSlider->setValue(current);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

//Open video
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
    milliSec=0;

    startTimer(30);
}

//Change Saturation
void MainWindow::on_satSlider_valueChanged(int value)
{
    float valf=(float)value/10;
    ui->satLabel->setText("Saturation: "+QString::number(valf));
    gw->setSaturation(valf);
}

//Rewind 10sec
void MainWindow::on_backButton_clicked()
{
    milliSec-=10000;
    if(milliSec<0)
        milliSec=0;

    cap.set(CV_CAP_PROP_POS_MSEC,milliSec);

    std::cout<<milliSec<<std::endl;
}

//Go forward 10sec
void MainWindow::on_forwardButton_clicked()
{
    milliSec+=10000;
    cap.set(CV_CAP_PROP_POS_MSEC,milliSec);
    std::cout<<milliSec<<std::endl;
}

//Scroll video with progress bar
void MainWindow::on_progressSlider_valueChanged(int value)
{
    if(!cap.isOpened())
    {
        milliSec=0;
        ui->progressSlider->setValue(0);
        return;
    }

    float lenght=cap.get(CV_CAP_PROP_FRAME_COUNT)*30;
    milliSec= value*lenght/100;
    cap.set(CV_CAP_PROP_POS_MSEC,milliSec);
}
