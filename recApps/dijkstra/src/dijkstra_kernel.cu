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

// ----------------------------------------------------------
// Implementation 0: NonNP dijkstra Wrapper
// ----------------------------------------------------------
void dijkstraNonNPWrapper()
{

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
void dijkstraOptNPWrapper()
{

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
	// switch (config.solution) {
	// 	case 0: dijsktraNonNPWrapper();	//GPU rec implementation
	// 		break;
	// 	case 1: dijkstraNaiveNPWrapper(); // Prim GPU with 2 phases
	//       break;
  //   case 2: dijkstraOptNPWrapper(); // Prim GPU with 2 phases
  //     break;
	// 	default:
  //     printf("===ERROR=== Solution selected not available\n");
	// 		break;
	// }
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
