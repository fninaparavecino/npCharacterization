#include <stdio.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
using namespace std;
__global__ void childKernel(int* A, int *B, int *C, int parentIdxVar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	C[parentIdxVar+idx] = A[parentIdxVar+idx] + B[parentIdxVar+idx];
}
__global__ void parentKernel(int* A, int *B, int *C, int *npId, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(A[idx*cols] == 1)
	{	
		npId[idx] = idx*cols;
		if (cols > 1024){
			childKernel<<<cols/1024, 1024>>>(A, B, C, npId[idx]);
		}
		else{			
			childKernel<<<1, cols>>>(A, B, C, npId[idx]);
		}
	}
}
__global__ void childKernelSync(int* A, int *B, int *C, int parentIdxVar)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	C[parentIdxVar+idx] = A[parentIdxVar+idx] + B[parentIdxVar+idx];
}
__global__ void parentKernelSync(int* A, int *B, int *C, int *npId, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(A[idx*cols] == 1)
	{	
		npId[idx] = idx*cols;
		if (cols > 1024){
			childKernelSync<<<cols/1024, 1024>>>(A, B, C, npId[idx]);
		}
		else{
			//clock_t start, stop;
			//__synchthreads();
			//start = clock();
			
			childKernelSync<<<1, cols>>>(A, B, C, npId[idx]);			
			cudaDeviceSynchronize();
			//stop = clock();
			//if (idx== 0) //just for the first thread
			//	printf("Number of clocks: %d... \n", (int)(stop-start));
		}
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
				printf("ERROR...[%d %d] ", i, j);
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
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("=================================\n");
    return;
}

int main(int argC, char** argV)
{
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);
 
	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(devProp);
	}
	///*******************************
	int mod = 2;
	int ROWS = 1024, COLS = 1024;
	for(int i=1; i<argC; i=i+2)
	{
		if(strcmp(argV[i], "--size") == 0)
		{
			if(i+1 < argC)
			{
				ROWS = atoi(argV[i+1]);
				COLS = ROWS;				
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
		else if(strcmp(argV[i], "--div") == 0){
			if(i+1 < argC)
			{
				mod = 100/atoi(argV[i+1]);
				if(mod <= 0)
				{
					cerr << "Divergence must be greater than 0." << endl;
					exit(1);
				}
				break;
			}
			else
			{
				printf("Error...\n");
				exit(1);
			}
		}
		else if(strcmp(argV[i], "-h") == 0 || strcmp(argV[i], "--help") == 0)
		{
			cout << "Usage: " << argV[0] << " [OPTIONS] --size <number> --div <number>" << endl;
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

	printf("NP - Characterization: %d percentage of divergence\n", (100/mod));
	printf("NP Case2: [%d x %d]\n", ROWS, COLS);
	
	int *a = (int*) malloc(ROWS*COLS*sizeof(int));
	int *b = (int*) malloc(ROWS*COLS*sizeof(int));
	int *c = (int*) malloc(ROWS*COLS*sizeof(int));
	int nroChildKernels = 0;
	for (int i=0; i<ROWS; i++){
		if(i%mod == 0)
			nroChildKernels++;
		for(int j=0; j<COLS; j++){
			if(i%mod == 0){
				a[i*COLS+j] = 1;
				b[i*COLS+j] = 2;
			}
		}
	}
	printf("Number of child kernels: %d\n", nroChildKernels);
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
	dim3 threads, blocks;
	if (ROWS >1024){
		threads.x = 1024; threads.y = 1; threads.z = 1;
		blocks.x = ROWS/threads.x; blocks.y = 1; blocks.z = 1;
	}
	else{
		threads.x = ROWS; threads.y = 1; threads.z = 1;
		blocks.x = 1; blocks.y = 1; blocks.z = 1; 
	}
	
	singleKernel<<<blocks,threads>>>(devA, devB, devC2, ROWS, COLS);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time single kernel: %.2f ms\n", time);
	
	//Retrieve results from device
	cudaMemcpy(c, devC2, ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);
	//Verify correctness	
	check(c, cHost, ROWS, COLS) ? printf("Results are correct.\n") : printf("Results are not correct.\n");

	// NP Case ****************************************************************
	int *devC, *devNpId;
	int *cNp = (int*)malloc(ROWS*COLS*sizeof(int));
	int *npId = (int*)malloc(ROWS*COLS*sizeof(int));
	cudaMalloc((void**)&devC, ROWS*COLS*sizeof(int));
	cudaMalloc((void**)&devNpId, ROWS*COLS*sizeof(int));
	cudaMemcpy(devC, cNp, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devNpId, npId, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	
	parentKernel<<<blocks, threads>>>(devA, devB, devC, devNpId, ROWS, COLS);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel NP Job time: %.2f ms\n", time);
	
	//Retrieve results from device
	cudaMemcpy(cNp, devC, ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);
	//Verify correctness	
	check(cNp, cHost, ROWS, COLS) ? printf("Results are correct.\n") : printf("Results are not correct.\n");
	
	// NP Sync Case ****************************************************************
		int *devCSync;
		int *cNpSync = (int*)malloc(ROWS*COLS*sizeof(int));
		//int *npId = (int*)malloc(ROWS*COLS*sizeof(int));
		cudaMalloc((void**)&devCSync, ROWS*COLS*sizeof(int));
		//cudaMalloc((void**)&devNpId, ROWS*COLS*sizeof(int));
		cudaMemcpy(devCSync, cNpSync, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(devNpId, npId, ROWS*COLS*sizeof(int), cudaMemcpyHostToDevice);
		cudaEventRecord(start, 0);
		
		parentKernel<<<blocks, threads>>>(devA, devB, devCSync, devNpId, ROWS, COLS);
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		
		//Display time
		cudaEventElapsedTime(&time, start, stop);
		printf("\tParallel NP Sync Job time: %.2f ms\n", time);
		
		//Retrieve results from device
		cudaMemcpy(cNpSync, devCSync, ROWS*COLS*sizeof(int), cudaMemcpyDeviceToHost);
		//Verify correctness	
		check(cNpSync, cHost, ROWS, COLS) ? printf("Results are correct.\n") : printf("Results are not correct.\n");
}
