/*******************
 * npCase1: Matrix Multiplication
 * Author : Fanny Nina-Paravecino
 * Date   : October 2016
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#define COL 1024 
#define ROW 1024
double wallS0, wallS1, wallP0, wallP1;
float sum;
double getWallTime(){
        struct timeval time;
        if(gettimeofday(&time,NULL)){
                printf("Error getting time\n");
                return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

__global__ void matrixMulKernel( float *devA, float *devB, float *devC, int row, int col, const int k){
	int txID = blockIdx.x * blockDim.x + threadIdx.x;//Col of devC
	int tyID = blockIdx.y * blockDim.y + threadIdx.y;//Row of devC.

	if ((txID < col) && (tyID < row)){
		float Pvalue = 0;
		for(int w=0; w<k; w++){
			Pvalue += devA[tyID*k+w] * devB[w*k+txID];
		}
		devC[tyID*k+txID] = Pvalue;
	}
}

void matrixMultiplication(float *a, float *b, float *c, int row, int col, int k, int blockX, int blockY)
{
	//Setting device memory space.
	int sizeA = row*k*sizeof(float);
	int sizeB = k*col*sizeof(float);
	int sizeC = row*col*sizeof(float);
	float *devA, *devB, *devC;
        
	//Time variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	cudaMalloc((void**)&devA, sizeA);
	cudaMalloc((void**)&devB, sizeB);
	cudaMalloc((void**)&devC, sizeC);
	
	//Copying [A] and [B] from host memory to device memory.
	cudaMemcpy(devA, a, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, b, sizeB, cudaMemcpyHostToDevice);

	//Setting execution configuration.
	dim3 dimBlock(blockX, blockY, 1);
	dim3 dimGrid((COL+dimBlock.x-1)/dimBlock.x, (ROW+dimBlock.y-1)/dimBlock.y, 1);
	printf("\tBlock(%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
	printf("\tGrid(%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
	//Launching device computation threads.
	matrixMulKernel<<<dimGrid, dimBlock>>>(devA, devB, devC, row, col, k);
	//Transferring [C] from device to host.
	cudaMemcpy(c, devC, sizeC, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time: %.2f ms", time);
	//Freeing device matrices.
	cudaFree(devA); 
	cudaFree(devB); 
	cudaFree(devC);
}

bool checkResults(float *test, float *c, int row, int col){
	bool b= true;
	for(int i=0; i<row; i++){
			for(int j=0; j<col; j++){
					if(test[i*col+j] != c[i*col+j]){
							b=false;
							printf("test[%d, %d] = %.2f \t c[%d, %d] = %.2f\n", i, j, test[i*col+j], i, j, c[i*col+j]);
							break;
					}
			}
	}
	return b;
}

int main(int argC, char** argV)
{
	float *a, *b, *c, *test;
	//Setting matrix parameters.
	int row = ROW;
	int col = COL;
	int k = COL;
	//Setting host memory space.
	a = (float *) malloc(row*k*sizeof(float));
	b = (float *) malloc(k*col*sizeof(float));
	c = (float *) malloc(row*col*sizeof(float));
	test = (float *) malloc(row*col*sizeof(float));

	//Initializing [A] and [B] with random values from 1 to 10.
	for(int i=0; i<row; i++){
			for(int j=0; j<k; j++){
					a[i*k+j] = rand()%10;
			}
	}
	for(int i=0; i<k; i++){
			for(int j=0; j<col; j++){
					b[i*col+j] = rand()%10;
			}
	}
	printf("Matrix Multiplication: \nA[%d, %d] * B[%d, %d] = C[%d, %d]\n", row, k, k, col, row, col);

	//Performing sequential job.
	wallS0 = getWallTime();
	for(int i=0; i<row; i++){
			for(int j=0; j<col; j++){
					sum = 0;
					for(int w=0; w<k; w++){
							sum += a[i*k+w] * b[w*col+j];
					}
					test[i*col+j]=sum;
			}
	}
	wallS1 = getWallTime();
	printf("\tSequential Job Time: %f ms\n", (wallS1-wallS0)*1000);
	
	//Calling stub function to allocate device memory, perform data transfer, and launch kernel.
	int blockX = 32;
	if (argV[1] != NULL)
		blockX = atoi(argV[1]);
	int blockY = 32;
	if (argV[2] != NULL)
		blockY = atoi(argV[2]);
       
	if (!blockX)
		blockX = 32;
	if (!blockY)
		blockY = 32;
	matrixMultiplication(a, b, c, row, col, k, blockX, blockY);
        
	//Verify results
	if(checkResults(test, c, row, col)){
		printf("\nResults are correct.\n");
	}
}