#include <stdio.h>
#include <cuda.h>
#include "prim.h"

#ifndef THREADS_PER_BLOCK // nested kernel block size
#define THREADS_PER_BLOCK 64
#endif

#ifndef WARP_SIZE // nested kernel block size
#define WARP_SIZE 32
#endif

int *d_vertexArray;
int *d_edgeArray;
int *d_levelArray;
bool *d_visitedArray;
int *d_keyArray;
int *d_weightArray;

#define cudaErrCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void primRec(int node, int numNodes, int* vertexArray, int* edgeArray,
                        int* weightArray, bool* visitedArray, int* keyArray, int* mstParent){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > numNodes)
    return;

  // mark node as visited
  visitedArray[node] = true;

  int child = edgeArray[vertexArray[node] + idx];
  int childId = vertexArray[node] + idx;
  int childWeight = weightArray[vertexArray[node]] + idx;

  // set min to first child
  int minChild = edgeArray[vertexArray[node]];
  int minChildId = vertexArray[node];
  int minChildWeight = weightArray[vertexArray[node]];

  // find minChild among children
  for (int i=1;i < blockDim.x && i < WARP_SIZE; i++){
    int childShfl = __shfl(child, i);
    int childIdShfl = __shfl(childId, i);
    int weightShfl = __shfl(childWeight, i);

    if (weightShfl < minChildWeight){
      minChildWeight = weightShfl;
      minChild = childShfl;
      minChildId = childIdShfl;
    }
  }

  // if Child explored is different than childSelected
  if (child != minChild)
    return;

  if (visitedArray[minChild] == false && weightArray[minChildId] < keyArray[minChild]){
    //printf("===GPU Kernel=== child selected: %d\n", minChild);
    mstParent[minChild] = node;
    keyArray[minChild] = weightArray[minChildId];
    int grandChildren = vertexArray[minChild+1] - vertexArray[minChild];
    primRec<<<1, grandChildren>>>(minChild, numNodes, vertexArray, edgeArray, weightArray, visitedArray, keyArray, mstParent);
  }
}

// ----------------------------------------------------------
// Recursive MST using Prim's algorithm
// ----------------------------------------------------------
void primRecWrapper()
{
	cudaEvent_t start, stop;
	float time;
	/* prepare GPU */

	int children = graph.vertexArray[source+1]-graph.vertexArray[source];
	unsigned block_size = min (children, THREADS_PER_BLOCK);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	primRec<<<1,block_size>>>(source, noNodeTotal, d_vertexArray, d_edgeArray, d_weightArray, d_visitedArray, d_keyArray, d_levelArray);
	cudaErrCheck( cudaGetLastError());
	cudaErrCheck( cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time: %.2f ms\n", time);

	if (DEBUG)
		printf("===> GPU Prim rec.\n");
}

void prepare_gpu()
{
	start_time = gettime_ms();
	cudaFree(NULL);
	end_time = gettime_ms();
	init_time += end_time - start_time;

	if (DEBUG) {
		fprintf(stderr, "Choose CUDA device: %d\n", config.device_num);
		fprintf(stderr, "cudaSetDevice:\t\t%lf\n",end_time-start_time);
	}

	start_time = gettime_ms();
	size_t limit = 0;
	if (DEBUG) {
		cudaErrCheck(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
		printf("cudaLimistMallocHeapSize: %u\n", (unsigned)limit);
	}
	limit = 102400000;
	cudaErrCheck( cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));
	if (DEBUG) {
		cudaErrCheck(cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
		printf("cudaLimistMallocHeapSize: %u\n", (unsigned)limit);
	}
	end_time = gettime_ms();

	/* Allocate GPU memory */
	start_time = gettime_ms();
	cudaErrCheck(cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaErrCheck(cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
	cudaErrCheck(cudaMalloc( (void**)&d_levelArray, sizeof(int)*noNodeTotal ) );
  cudaErrCheck(cudaMalloc( (void**)&d_visitedArray, sizeof(bool)*noNodeTotal ) );
  cudaErrCheck(cudaMalloc( (void**)&d_weightArray, sizeof(int)*noEdgeTotal ) );
  cudaErrCheck(cudaMalloc( (void**)&d_keyArray, sizeof(int)*noNodeTotal ) );

	end_time = gettime_ms();
	d_malloc_time += end_time - start_time;

	start_time = gettime_ms();
	cudaErrCheck( cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaErrCheck( cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaErrCheck( cudaMemcpy( d_levelArray, graph.levelArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_visitedArray, graph.visited, sizeof(bool)*noNodeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_weightArray, graph.weightArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_keyArray, graph.keyArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
	end_time = gettime_ms();
	h2d_memcpy_time += end_time - start_time;
}
void clean_gpu()
{
	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_levelArray);
  cudaFree(d_visitedArray);
	cudaFree(d_weightArray);
  cudaFree(d_keyArray);
}

void primGPU()
{
	cudaErrCheck( cudaSetDevice(config.device_num) );
	cudaErrCheck( cudaDeviceReset());
	prepare_gpu();

#ifdef GPU_PROFILE
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif

	start_time = gettime_ms();
	switch (config.solution) {
		case 0:  primRecWrapper();	//
			break;
		default:
			break;
	}
	cudaErrCheck(cudaDeviceSynchronize() );
	end_time = gettime_ms();
	ker_exe_time += end_time - start_time;
#ifdef GPU_PROFILE
	gpu_statistics<<<1,1>>>(config.solution);
	cudaDeviceSynchronize();
#endif
	//copy the level array from GPU to CPU;
	start_time = gettime_ms();
	cudaErrCheck( cudaMemcpy( graph.levelArray, d_levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime_ms();
	d2h_memcpy_time += end_time - start_time;

	clean_gpu();
}
