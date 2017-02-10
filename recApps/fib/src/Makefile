# enviroment
CUDA_VERSION = cuda-8.0
CPP = g++

NVCC = /usr/local/$(CUDA_VERSION)/bin/nvcc
CUDA_LIB_PATH = /usr/local/$(CUDA_VERSION)/lib64

SOURCEDIR = .

CPP_SOURCES = $(SOURCEDIR)/fib.cpp
CU_SOURCES = $(SOURCEDIR)/fib_kernel.cu

CFLAGS = -std=c++11

NVCCFLAGS = -g -O3

export LD_LIBRARY_PATH := $(INST_LIB_DIR):$(LD_LIBRARY_PATH)

ARCH = -gencode arch=compute_35,code=sm_35

EXE = $(SOURCEDIR)/fib

DEVICE_OBJS = $(SOURCEDIR)/fib_cu.o

LINK_OBJS = $(SOURCEDIR)/fib_cu_dlink.o

all: fib

fib: fib_cu_dlink.o
	$(CPP) $(CFLAGS) -o $(EXE) $(DEVICE_OBJS) $(LINK_OBJS) $(CPP_SOURCES) -L$(CUDA_LIB_PATH) -lcudart -lcuda -lcudadevrt

fib_cu_dlink.o: fib_cu.o
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xcompiler '-fPIC' -dlink $(DEVICE_OBJS) -o $(LINK_OBJS) -lcudadevrt

fib_cu.o:
	$(NVCC) $(NVCCFLAGS) $(ARCH) -Xcompiler '-fPIC' -dc -m64 $(CU_SOURCES) -o $(DEVICE_OBJS)

clean:
	rm -f $(DEVICE_OBJS)
	rm -f $(LINK_OBJS)
	rm -f $(EXE)
