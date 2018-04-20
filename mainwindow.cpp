#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(&grabFramesTimer, SIGNAL(timeout()), this, SLOT(on_grabFrame()));
    connect(&updateGuiTimer, SIGNAL(timeout()), this, SLOT(on_updateGui()));
    updateGuiTimer.start(500);
}

//Clean memory
MainWindow::~MainWindow()
{
    if(videoWidget!=NULL)
        delete videoWidget;

    delete ui;
}

//Open a window to browse to the video source and then start reading the video
void MainWindow::on_browseButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select video", QDir::homePath(), "Video files (*.*)");
    std::string videoPath=fileName.toStdString();

    if(videoPath=="")  //if the user pressed "Cancel"
        return;

    if(!cap.open(videoPath)){
        QMessageBox::information (0, "Error", QString("Invalid path"));
                  return;
    }
    cap.read(imgSrc);
    frameDurationMs=1000/cap.get(CV_CAP_PROP_FPS);

    if(videoWidget!=NULL)
        delete videoWidget;

    videoWidget=new GLWidget(ui->viewerGroupBox, imgSrc.size());
    ui->viewerGroupBox->layout()->addWidget(videoWidget);
    videoWidget->show();
    ellapsedTimeMs=0;
    on_satSlider_sliderMoved(ui->satSlider->value());

    grabFramesTimer.stop();
    grabFramesTimer.start(frameDurationMs);
}

//Update progress bar & time  label
void MainWindow::on_updateGui()
{
    if(!cap.isOpened())
        return;

    //Time Label
    int minutes=ellapsedTimeMs/60000;
    int seconds=((int)ellapsedTimeMs%60000)/1000;
    QString textTime=((minutes<10)?"0":"")+QString::number(minutes)+":"+((seconds<10)?"0":"")+QString::number(seconds);
    ui->timeLabel->setText(textTime);

    //Progress bar
    int videoLenght=cap.get(CV_CAP_PROP_FRAME_COUNT)*frameDurationMs;
    int progress= ui->progressSlider->maximum()* ellapsedTimeMs/videoLenght;
    ui->progressSlider->setValue(progress);
}

//Reads new frames
void MainWindow::on_grabFrame()
    {
    if(!cap.isOpened())
        return;

    if(!cap.read(imgSrc))
    {
        ellapsedTimeMs=0;
        cap.set(CV_CAP_PROP_POS_MSEC,ellapsedTimeMs);
    }
    ellapsedTimeMs+=frameDurationMs;

    if(imgSrc.data)
        videoWidget->updateImage(imgSrc.ptr());
}

//Change Saturation
void MainWindow::on_satSlider_sliderMoved(int position)
{
    if(videoWidget!=NULL){
        float val=(float)position/10;
        ui->satLabel->setText("Saturation: "+QString::number(val));
        videoWidget->setSaturation(val);
    }
    else
        ui->satSlider->setValue(10);
}

//Rewind 10sec
void MainWindow::on_backButton_clicked()
{
    if(cap.isOpened())
    {
        ellapsedTimeMs-=10000;
        if(ellapsedTimeMs<0)
            ellapsedTimeMs=0;
        cap.set(CV_CAP_PROP_POS_MSEC,ellapsedTimeMs);
    }
}

//Go forward 10sec
void MainWindow::on_forwardButton_clicked()
{
    if(cap.isOpened())
    {
        ellapsedTimeMs+=10000;
        cap.set(CV_CAP_PROP_POS_MSEC,ellapsedTimeMs);
    }
}

//Toggle Pause / Play
void MainWindow::on_pauseButton_clicked()
{
    if(grabFramesTimer.isActive())
        grabFramesTimer.stop();
    else
        grabFramesTimer.start();

}

//Scroll video with progress bar
void MainWindow::on_progressSlider_sliderMoved(int position)
{
    if(cap.isOpened())
    {
        float lenght=cap.get(CV_CAP_PROP_FRAME_COUNT)*frameDurationMs;
        ellapsedTimeMs= position*lenght/ui->progressSlider->maximum();
        cap.set(CV_CAP_PROP_POS_MSEC,ellapsedTimeMs);
    }
    else
    {
        ellapsedTimeMs=0;
        ui->progressSlider->setValue(0);
        return;
    }
}
