
QT  += core gui
QT  += widgets
QT       += opengl

CONFIG += c++11

TARGET = QGLApp
TEMPLATE = app


#OpenGL
LIBS += -lglut -lGLU -lGL -lGLEW

#OpenCV
INCLUDEPATH += -I/usr/local/include/opencv -I/usr/local/include
LIBS += `pkg-config --libs opencv`

# Cuda
INCLUDEPATH += -I/usr/local/cuda-8.0/include                     # CUDA
INCLUDEPATH += -I/usr/include/libdrm
LIBS += -L/usr/local/cuda-8.0/lib64 -lcuda -lcublas -lcudart -lcufft



# Cuda
INCLUDEPATH += -I/usr/local/cuda-8.0/include                     # CUDA
INCLUDEPATH += -I/usr/include/libdrm
LIBS += -L/usr/local/cuda-8.0/lib64 -lcuda -lcublas -lcudart -lcufft

#SETUP NVCC COMPILER
CUDA_LIBDIR = /usr/local/cuda/lib64
CUDALIB = -L$$CUDA_LIBDIR -L$$CUDASDK_LIBDIR -lcudart -lcutil -lglut -lGLU -lGL -lcublas -lcudart -lcufft

CUDA_DIR = /usr/local/cuda
CUDA_ARCH = sm_61 # as supported by the GTX 1080 Ti

#SOURCES FOR NVCC
CUDA_SOURCES = src/process.cu
CUDA_DIR = /usr/local/cuda

cuda.commands = $$CUDA_DIR/bin/nvcc -std=c++11 -c -arch=$$CUDA_ARCH -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}  -I/opt/euresys/coaxlink/include ${CUDALIB}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc --gpu-architecture=sm_61 -M ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.o
QMAKE_EXTRA_COMPILERS += cuda
INCLUDEPATH += $$CUDA_DIR/include


SOURCES += \
            src/main.cpp \
            src/mainwindow.cpp \
            src/glwidget.cpp

HEADERS += \
            src/mainwindow.h \
            src/glwidget.h \
            src/process.cuh

FORMS += \
        src/mainwindow.ui

DISTFILES += \
    src/process.cu
