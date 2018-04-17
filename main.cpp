#include "mainwindow.h"
#include <QApplication>
#include <GL/glut.h>

int main(int argc, char *argv[])
{
    cudaGLSetGLDevice(0);

    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
