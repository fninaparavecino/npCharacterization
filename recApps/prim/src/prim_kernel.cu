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
int *d_nodesVisited;

#define cudaErrCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
/***********************************************************
        Prim Recursive version 0
************************************************************/
__global__ void primRec(int node, int numNodes, int* vertexArray, int* edgeArray,
                        int* weightArray, bool* visitedArray, int* keyArray, int* mstParent){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > numNodes)
    return;

  // mark node as visited
  visitedArray[node] = true;

  int child = edgeArray[vertexArray[node] + idx];
  int childId = vertexArray[node] + idx;
  int childWeight = weightArray[vertexArray[node] + idx];

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
/***********************************************************
        Prim Recursive version 1
************************************************************/
__global__ void primPhase2(int node, int numChildren, int* vertexArray, int* edgeArray,
                        int* weightArray, bool* visitedArray, int* keyArray, int* mstParent){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > numChildren)
    return;

  int child = edgeArray[vertexArray[node] + idx];
  int childId = vertexArray[node] + idx;

  if (visitedArray[child] == false && weightArray[childId] < keyArray[child]){
    //printf("===GPU Kernel=== child selected: %d\n", minChild);
    mstParent[child] = node;
    keyArray[child] = weightArray[childId];
    int grandChildren = vertexArray[child+1] - vertexArray[child];
    primRec<<<1, grandChildren>>>(child, grandChildren, vertexArray, edgeArray, weightArray, visitedArray, keyArray, mstParent);
  }
}

__global__ void primPhase1(int numNodes, int* nodesVisited, int* vertexArray, int* edgeArray,
                        int* weightArray, bool* visitedArray, int* keyArray, int* mstParent){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > numNodes)
    return;

  int node = idx;
  int nodeKey = keyArray[node];

  int minNode = 0;
  int minNodeKey = keyArray[0];

  //while(nodesVisited[0] < numNodes){
    // find minChild among children
    for (int i=1; i < numNodes && i < WARP_SIZE; i++){
      int nodeShfl = __shfl(node, i);
      int keyShfl = __shfl(nodeKey, i);

      if (visitedArray[keyShfl] == false && keyShfl < minNodeKey){
        minNodeKey = keyShfl;
        minNode = nodeShfl;
      }
    }

    if(minNode != node)
      return;

    // mark node as visited
    visitedArray[minNode] = true;
    atomicAdd(&nodesVisited[0], 1);
    __syncthreads();
    int children = vertexArray[minNode+1] - vertexArray[minNode];
    primPhase2<<<1, children>>>(minNode, children, vertexArray, edgeArray,
                             weightArray, visitedArray, keyArray, mstParent);

  //}
}
// ----------------------------------------------------------
// Implementation 0: Recursive MST using Prim's algorithm
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

// ----------------------------------------------------------
// Implementation 1: Prim's algorithm using 2 phases
// ----------------------------------------------------------
void primWrapper2Phases()
{
  cudaEvent_t start, stop;
  float time;

  // prepare GPU
  unsigned block_size = min (noNodeTotal, THREADS_PER_BLOCK);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  primPhase1<<<1, block_size>>>(noNodeTotal, d_nodesVisited, d_vertexArray, d_edgeArray, d_weightArray, d_visitedArray, d_keyArray, d_levelArray);
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
// ----------------------------------------------------------
// Implementation 2:
// ----------------------------------------------------------
void primWrapperControl()
{
  cudaEvent_t start, stop;
  float time;

  // prepare GPU
  int children = graph.vertexArray[source+1] - graph.vertexArray[source];

  unsigned block_size = min (children, THREADS_PER_BLOCK);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int minNode = source;
  int minKey = graph.keyArray[source];

  while(graph.nodesVisited[0] < noNodeTotal){
    for(int i=0; i< noNodeTotal; i++){
      if (graph.visited[i] == false && graph.keyArray[i] < minKey){
        minNode = i;
        minKey = graph.keyArray[i];
      }
    }

    primRec<<<1, block_size>>>(minNode, noNodeTotal, d_vertexArray, d_edgeArray, d_weightArray, d_visitedArray, d_keyArray, d_levelArray);
    cudaErrCheck( cudaDeviceSynchronize());
    cudaErrCheck( cudaMemcpy( graph.visited, d_visitedArray, sizeof(char)*noNodeTotal, cudaMemcpyDeviceToHost) );
    cudaErrCheck( cudaMemcpy( graph.keyArray, d_keyArray, sizeof(int)*noNodeTotal, cudaMemcpyDeviceToHost) );
    cudaErrCheck( cudaMemcpy( graph.nodesVisited, d_nodesVisited, sizeof(int)*1, cudaMemcpyDeviceToHost) );
  }

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
  cudaErrCheck(cudaMalloc( (void**)&d_nodesVisited, sizeof(int)*1 ) );

	end_time = gettime_ms();
	d_malloc_time += end_time - start_time;

	start_time = gettime_ms();
	cudaErrCheck( cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaErrCheck( cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaErrCheck( cudaMemcpy( d_levelArray, graph.levelArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_visitedArray, graph.visited, sizeof(bool)*noNodeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_weightArray, graph.weightArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_keyArray, graph.keyArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_nodesVisited, graph.nodesVisited, sizeof(int)*1, cudaMemcpyHostToDevice) );
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
  cudaFree(d_nodesVisited);
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
		case 0: primRecWrapper();	//GPU rec implementation
			break;
    case 1: primWrapper2Phases(); // Prim GPU with 2 phases
      break;
		default:
      printf("===ERROR=== Solution selected not available\n");
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
