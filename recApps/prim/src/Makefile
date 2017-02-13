CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20
CC_FLAGS = --compiler-options '-fopenmp' -arch=sm_35 -rdc=true -lcudadevrt -O2 #-Xptxas=-v

#CC_FLAGS += -Xptxas=-v

#CC_FLAGS += -G -g

INC = -I./ -I../../common

all: global.o prim

global.o: ../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

prim: global.o prim.cpp prim.h prim_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o prim_kernel.cu prim.cpp -o $@

clean:
	rm -rf *.o prim
