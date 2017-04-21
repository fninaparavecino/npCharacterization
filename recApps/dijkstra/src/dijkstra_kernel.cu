#include <stdio.h>
#include <cuda.h>
#include "dijkstra.h"

#ifndef THREADS_PER_BLOCK // nested kernel block size
#define THREADS_PER_BLOCK 64
#endif

#ifndef WARP_SIZE // nested kernel block size
#define WARP_SIZE 32
#endif

#define GPU_PROFILE
#define GPU_WORKEFFICIENCY
#ifdef GPU_PROFILE
// records the number of kernel calls performed
__device__ unsigned nested_calls = 0;
__device__ unsigned total_threads = 0;

__global__ void gpu_statistics(unsigned solution){
	printf("====> GPU #%u - number of kernel calls: %u\n",solution, nested_calls);
	printf("====> GPU #%u - number of total threads: %u\n",solution, total_threads);
}

__global__ void reset_gpu_statistics(){
	nested_calls = 0;
	total_threads = 0;
}
#endif

#ifdef GPU_WORKEFFICIENCY
// records the number of kernel calls performed
__device__ unsigned work_efficiency = 0;

__global__ void gpu_statisticsWE(unsigned solution){
	printf("====> GPU #%u - number of wasted work of threads: %u\n",solution, work_efficiency);
}

__global__ void reset_gpu_statisticsWE(){
	work_efficiency = 0;
}
#endif

__device__ unsigned source_node = UNDEFINED;

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
        Dijsktra nonNP version 0
************************************************************/
__global__ void dijkstraNonNP(int numNodes, int* vertexArray, int* edgeArray,
                        int* weightArray, bool* visitedArray, int* distArray,
                        int* nodesVisited){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // find shortest path for all vertices
  while (nodesVisited[0] < numNodes){
    int min = UNDEFINED;
    int minNode;
		//find minDistance
		for (int node = tid; node < numNodes; node++){
	    if (visitedArray[node] == false && distArray[node] <= min){
	      min = distArray[node];
	      minNode = node;
	    }
	  }
		__syncthreads();

		if (tid == 0){
			source_node = minNode; // so every single thread can see it
			visitedArray[source_node] = true;
			atomicAdd(&nodesVisited[0], 1); // to control to visit all nodes
		}
		__syncthreads();

    for (int edgeId = tid + vertexArray[source_node]; edgeId < vertexArray[source_node+1];
			edgeId += blockDim.x * gridDim.x){
      int v = edgeArray[edgeId];
      if (visitedArray[v] == false && distArray[source_node] != UNDEFINED
				&& distArray[source_node] + weightArray[edgeId] <  distArray[v]){
        //printf("Edge added: %d with weight %d\n", v, graph.weightArray[edgeId]);
        distArray[v]  = distArray[source_node] + weightArray[edgeId];
      }
    }
  }
}
/***********************************************************
        Dijsktra Recursive OptNP version 2
************************************************************/
__global__ void dijkstraNPOpt(int numNodes, int min, int* vertexArray, int* edgeArray,
                        int* weightArray, bool* visitedArray, int* distArray,
                        int* nodesVisited){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  #ifdef GPU_PROFILE
  		if (threadIdx.x+blockDim.x*blockIdx.x==0) atomicInc(&nested_calls, INF);
  #endif

  if (tid > numNodes)
    return;

  if (distArray[tid] == min){
		//printf("node source: %d\n", tid);

    int minNode = tid;
    //min  = UNDEFINED;
    visitedArray[minNode] = true;
    atomicAdd(&nodesVisited[0], 1); // to control to visit all nodes
    // printf("minNode %d with minKey: %d\n", minNode, keyArray[minNode]);

    for (int edgeId = vertexArray[minNode]; edgeId < vertexArray[minNode+1]; edgeId++){
      int v = edgeArray[edgeId];
      // printf("Neighbor key[%d]: %d, graph.weight[%d]: %d\n", v, keyArray[v], v, weightArray[edgeId]);
      if (visitedArray[v] == false && distArray[minNode] != UNDEFINED &&
				distArray[minNode] + weightArray[edgeId] <  distArray[v]){
        #ifdef GPU_WORKEFFICIENCY
        		if (distArray[minNode] + weightArray[edgeId] <  distArray[v]) atomicInc(&work_efficiency, INF);
        #endif
        // printf("Edge added: %d with weight %d\n", v, weightArray[edgeId]);
        distArray[v]  = minNode + weightArray[edgeId];
        if(distArray[v] < min){
          min = distArray[v];
        }
        // printf("New min: %d\n", min);
      }
    }

    __syncthreads();
    if (visitedArray[0] < numNodes){
      // if (min == UNDEFINED) {// this path is not leading to any MST
        for(int i = 0; i <numNodes; i++){
          if (visitedArray[i] == false && distArray[i] <= min){
            min = distArray[i];
          }
        }
      // }
      // printf("Launching kernel with min: %d\n", min);
      int block_size = THREADS_PER_BLOCK;
      if (numNodes < block_size)
        block_size = numNodes;
      int grid_size = (numNodes + block_size-1)/ block_size;
      dijkstraNPOpt<<<grid_size, block_size >>>(numNodes, min, vertexArray, edgeArray,
                              weightArray, visitedArray, distArray,
                              nodesVisited);
    }
  }
  else{
    #ifdef GPU_WORKEFFICIENCY
        atomicInc(&work_efficiency, INF);
    #endif
  }
}
// ----------------------------------------------------------
// Implementation 0: NonNP dijkstra Wrapper
// ----------------------------------------------------------
void dijkstraNonNPWrapper()
{
	dijkstraNonNP<<<1, 32>>>(noNodeTotal, d_vertexArray, d_edgeArray, d_weightArray, d_visitedArray, d_levelArray, d_nodesVisited);
  cudaErrCheck( cudaDeviceSynchronize());
  if (DEBUG)
  	printf("===> GPU Prim #%d Flat Non-NP implementation\n", config.solution);

}

// ----------------------------------------------------------
// Implementation 1: Naive NP dijkstra Wrapper
// ----------------------------------------------------------
void dijkstraNaiveNPWrapper()
{

}
// ----------------------------------------------------------
// Implementation 2: Opt NP dijkstra Wrapper
// ----------------------------------------------------------
void dijkstraNPOptWrapper()
{
	int block_size = min(noNodeTotal, THREADS_PER_BLOCK);
  int grid_size = (noNodeTotal+block_size+1)/block_size;
  if (DEBUG)
		printf("===> GPU #%d - rec gpu optimized parallelism. gridSize: %d, blockSize: %d\n", config.solution, grid_size, block_size);

  dijkstraNPOpt<<<grid_size, block_size>>>(noNodeTotal, 0, d_vertexArray, d_edgeArray, d_weightArray, d_visitedArray, d_levelArray, d_nodesVisited);
  cudaErrCheck( cudaDeviceSynchronize());
  if (DEBUG)
  	printf("===> GPU Prim #%d NP Opt implementation\n", config.solution);

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
  cudaErrCheck(cudaMalloc( (void**)&d_nodesVisited, sizeof(int)*1 ) );

	end_time = gettime_ms();
	d_malloc_time += end_time - start_time;

	start_time = gettime_ms();
	cudaErrCheck( cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaErrCheck( cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaErrCheck( cudaMemcpy( d_levelArray, graph.levelArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_visitedArray, graph.visited, sizeof(bool)*noNodeTotal, cudaMemcpyHostToDevice) );
  cudaErrCheck( cudaMemcpy( d_weightArray, graph.weightArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
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
  cudaFree(d_nodesVisited);
}

void dijkstraGPU()
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
		case 0: dijkstraNonNPWrapper();	//GPU rec implementation
			break;
		// case 1: dijkstraNaiveNPWrapper(); // Prim GPU with 2 phases
	  //     break;
    case 2: dijkstraNPOptWrapper(); // Prim GPU with 2 phases
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
#ifdef GPU_WORKEFFICIENCY
	gpu_statisticsWE<<<1,1>>>(config.solution);
	cudaDeviceSynchronize();
#endif
	//copy the level array from GPU to CPU;
	start_time = gettime_ms();
	cudaErrCheck( cudaMemcpy( graph.levelArray, d_levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime_ms();
	d2h_memcpy_time += end_time - start_time;

	clean_gpu();
}
