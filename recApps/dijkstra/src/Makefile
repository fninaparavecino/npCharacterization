CC = nvcc
#CC_FLAGS = --compiler-options '-fopenmp' -Xptxas=-v -arch=sm_20
CC_FLAGS = --compiler-options '-fopenmp' -arch=compute_35 -code=sm_35 -rdc=true -lcudadevrt  #-Xptxas=-v

#CC_FLAGS += -Xptxas=-v

#CC_FLAGS += -G -g

INC = -I./ -I../../common

all: global.o dijkstra

global.o: ../../common/global.cpp
	$(CC) $(CC_FLAGS) $(INC) -c $^ -o $@

dijkstra: global.o dijkstra.cpp dijkstra.h dijkstra_kernel.cu
	$(CC) $(CC_FLAGS) $(INC) global.o dijkstra_kernel.cu dijkstra.cpp -o $@

clean:
	rm -rf *.o dijkstra
