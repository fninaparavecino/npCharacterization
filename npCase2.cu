#include <stdio.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
using namespace std;
//#define ROWS 1024
//#define COLS 1024
__device__ int parentIdx[1024];

__global__ void childKernel(int* A, int *B, int *C, int parentIdxVar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	C[parentIdxVar+idx] = A[parentIdxVar+idx] + B[parentIdxVar+idx];
}
__global__ void parentKernel(int* A, int *B, int *C, int rows, int cols)
{
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(A[idx*cols] == 1)
	{	
		parentIdx[idx] = idx*cols;		
		childKernel<<<1, cols>>>(A, B, C, parentIdx[idx]);
	}
}
__global__ void singleKernel(int* A, int *B, int *C, int rows, int cols)
{
	int idx = blockIdx.x *blockDim.x + threadIdx.x;
	if(A[idx*cols] == 1)
	{	
		for(int i=0; i < cols; i++)
			C[idx*cols+i] = A[idx*cols+i]+B[idx*cols+i];
	}
}
void printOutput(int *A, int rows, int cols)
{
	for(int i=0; i < rows; i++)
	{
		for(int j=0; j < cols; j++){
			printf("%d ", A[i*cols+j]);
		}
		printf("\n");
	}
}
bool check(int *c1, int *c2, int rows, int cols){
	
	bool same = true;
	for(int i=0; i < rows; i++)
	{
		for(int j=0; j < cols; j++){
			if(c1[i*cols+j] != c2[i*cols+j]){
				same = false;
				break;
			}				
		}
		if (!same)
			break;
	}
	return same;
}
double getWallTime(){
        struct timeval time;
        if(gettimeofday(&time,NULL)){
                printf("Error getting time\n");
                return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
int main(int argC, char** argV)
{
	printf("NP - Characterization\n");
	int ROWS = 1024, COLS = 1024;
	for(int i=1; i<argC; i++)
	{
		if(strcmp(argV[i], "-size") == 0)
		{
			if(i+1 < argC)
			{
				ROWS = atoi(argV[i+1]);
				COLS = ROWS;
				printf("%d %d\n", ROWS, COLS);
				if(ROWS < 1)
				{
					cerr << "Size must be greater than 0." << endl;
					exit(1);
				}
			}
			else
			{
				printf("Error...\n");
				exit(1);
			}
		}
		else if(strcmp(argV[i], "-h") == 0 || strcmp(argV[i], "--help") == 0)
		{
			cout << "Usage: " << argV[0] << " [OPTIONS] -size <number>" << endl;
			cout << "  -h, --help            Display this information and exit." << endl;

			exit(0);
		}
		else
		{
			cerr << "Did not recognize '" << argV[i] << "'. Try '" << argV[0]
				<< " --help' for additional information." << endl;
			exit(1);
		}
	}
	
	printf("NP Case2: [%d x %d]\n", ROWS, COLS);
	int *a = (int*) malloc(ROWS*COLS*sizeof(int));
	int *b = (int*) malloc(ROWS*COLS*sizeof(int));
	int *c = (int*) malloc(ROWS*COLS*sizeof(int));
	for (int i=0; i<ROWS; i++){
		for(int j=0; j<COLS; j++){
			if(i%8 == 0){
				a[i*COLS+j] = 1;
				b[i*COLS+j] = 2;
			}
		}
	}
	// Sequential
	double wallS0, wallS1;
	wallS0 = getWallTime();
	int *cHost = (int*)malloc(ROWS*COLS*sizeof(int));
	for(int i=0; i<ROWS; i++){
		if(a[i*COLS] == 1)
			for(int j=0; j<COLS; j++){
				cHost[i*COLS+j] = a[i*COLS+j] + b[i*COLS+j];
			}
	}
	wallS1 = getWallTime();
	printf("\tSequential Job Time: %f ms\n", (wallS1-wallS0)*1000);
	// Time variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
		
	int *devA;
	int *devB;
	cudaMalloc((void**)&devA, ROWS*COLS*sizeof(int));
	cudaMalloc((void**)&devB, ROWS*COLS*sizeof(int));
		
	//Copying [A] and [B] from host memory to device memory.
	cudaMemcpy(devA, a, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
		
	// Single Kernel **********************************************************
	int *devC2;
	cudaMalloc((void**)&devC2, ROWS*COLS*sizeof(int));	
	cudaMemcpy(devC2, c, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	singleKernel<<<1,ROWS>>>(devA, devB, devC2, ROWS, COLS);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time single kernel: %.2f ms\n", time);
	
	//Retrieve results from device
	cudaMemcpy(c, devC2, ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);
	//Verify correctness	
	check(c, cHost, ROWS, COLS) ? printf("Results are correct.\n") : printf("ERROR! Results are not the same");

	// NP Case ****************************************************************
	int *devC;
	cudaMalloc((void**)&devC, ROWS*COLS*sizeof(int));
	cudaMemcpy(devC, c, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	parentKernel<<<1,ROWS>>>(devA, devB, devC, ROWS, COLS);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time: %.2f ms\n", time);
	
	//Retrieve results from device
	cudaMemcpy(c, devC, ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);
	//Verify correctness	
	check(c, cHost, ROWS, COLS) ? printf("Results are correct.\n") : printf("ERROR! Results are not the same");
}
