#include <stdio.h>
#include <cuda.h>
#include "bfs_rec.h"

#define QMAXLENGTH 10240000*10
#define GM_BUFF_SIZE 10240000*10

#ifndef THREADS_PER_BLOCK_FLAT	//block size for flat parallelism
#define THREADS_PER_BLOCK_FLAT 128
#endif

#ifndef NUM_BLOCKS_FLAT
#define NUM_BLOCKS_FLAT 256
#endif

#ifndef THREADS_PER_BLOCK // nested kernel block size
//#define THREADS_PER_BLOCK 64
#define THREADS_PER_BLOCK 128
#endif

#ifndef CONSOLIDATE_LEVEL
#define CONSOLIDATE_LEVEL 0
#endif

#define STREAMS 0

#include "bfs_rec_kernel.cu"

int *d_vertexArray;
int *d_edgeArray;
int *d_levelArray;
int *d_work_queue;
char *d_frontier;
char *d_update;

unsigned int *d_queue_length;
unsigned int *d_nonstop;

dim3 dimGrid(1,1,1);	// thread+bitmap
dim3 dimBlock(1,1,1);

//char *update = new char [noNodeTotal] ();
//int *queue = new int [queue_max_length];
unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;
unsigned int nonstop = 0;

inline void cudaCheckError(const char* file, int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: file %s, line %d %s\n", file, line, cudaGetErrorString(ce));
		exit(1);
	}
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
		cudaCheckError( __FILE__, __LINE__, cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
		printf("cudaLimistMallocHeapSize: %u\n", (unsigned)limit);
	}
	limit = 102400000;
	cudaCheckError( __FILE__, __LINE__, cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit));
	if (DEBUG) {
		cudaCheckError( __FILE__, __LINE__, cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize));
		printf("cudaLimistMallocHeapSize: %u\n", (unsigned)limit);
	}
	end_time = gettime_ms();
	//fprintf(stderr, "Set Heap Size:\t\t%.2lf ms.\n", end_time-start_time);

	/* Allocate GPU memory */
	start_time = gettime_ms();
	cudaCheckError( __FILE__, __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaCheckError( __FILE__, __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
	cudaCheckError( __FILE__, __LINE__, cudaMalloc( (void**)&d_levelArray, sizeof(int)*noNodeTotal ) );
	printf("DEBUG levelArray : %d \n", noNodeTotal);
	//cudaCheckError( __LINE__, cudaMalloc( (void**)&d_nonstop, sizeof(unsigned int) ) );
	end_time = gettime_ms();
	d_malloc_time += end_time - start_time;

	start_time = gettime_ms();
	cudaCheckError( __FILE__, __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __FILE__, __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	//copy the level array from CPU to GPU
	cudaCheckError( __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
	end_time = gettime_ms();
	h2d_memcpy_time += end_time - start_time;
}

void clean_gpu()
{
	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_levelArray);
}

// ----------------------------------------------------------
// version #0 - flat parallelism - level-based BFS traversal
// ----------------------------------------------------------

void bfs_flat_gpu()
{
	/* prepare GPU */

	bool queue_empty = false;
	bool *d_queue_empty;

	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_queue_empty, sizeof(bool)) );
  // printf("Grid configuration gridxblocks, %d x %d\n", NUM_BLOCKS_FLAT, THREADS_PER_BLOCK_FLAT);
	if (DEBUG)
		printf("===> GPU #%d - flat parallelism, gridSize: %d, blockSize: %d\n", config.solution, 1, THREADS_PER_BLOCK_FLAT);
	unsigned level = 0;

	//level-based traversal
	while (!queue_empty){
		cudaCheckError(  __FILE__, __LINE__, cudaMemset( d_queue_empty, true, sizeof(bool)) );

		bfs_kernel_flat<<<1, THREADS_PER_BLOCK_FLAT>>>(level,noNodeTotal, d_vertexArray, d_edgeArray, d_levelArray, d_queue_empty);
		cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
		cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( &queue_empty, d_queue_empty, sizeof(bool), cudaMemcpyDeviceToHost) );
		level++;
	}

	if (DEBUG)
		printf("===> GPU #%d - flat parallelism.\n", config.solution);

}

// ----------------------------------------------------------
// version #1 - dynamic parallelism - naive
// ----------------------------------------------------------
void bfs_rec_dp_naive_gpu()
{
	cudaEvent_t start, stop;
	float time;
	/* prepare GPU */

	int children = graph.vertexArray[source+1]-graph.vertexArray[source];
	unsigned block_size = min (children, THREADS_PER_BLOCK);
	if (DEBUG)
		printf("===> GPU #%d - nested parallelism naive gridSize: %d bockSize: %d.\n", config.solution, 1, block_size);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	bfs_kernel_dp<<<1,block_size>>>(source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time: %.2f ms", time);

	if (DEBUG)
		printf("===> GPU #%d - nested parallelism naive.\n", config.solution);
}

// ----------------------------------------------------------
// version #2 - dynamic parallelism - hierarchical
// ----------------------------------------------------------
void bfs_rec_dp_hier_gpu()
{
	//recursive BFS traversal - hierarchical
	int children = graph.vertexArray[source+1]-graph.vertexArray[source];
	bfs_kernel_dp_hier<<<children, THREADS_PER_BLOCK>>>(source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	if (DEBUG)
		printf("===> GPU #%d - nested parallelism hierarchical %f.\n", config.solution, gettime_ms()-start_time);
}

// ----------------------------------------------------------
// version #3 - dynamic parallelism - consolidation
// ----------------------------------------------------------
void bfs_rec_dp_cons_gpu()
{
	//recursive BFS traversal - dynamic parallelism consolidation
	unsigned int *d_buffer;
	unsigned int *d_idx;
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_buffer, sizeof(unsigned int)*GM_BUFF_SIZE) );
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_idx, sizeof(unsigned int)) );
    bfs_kernel_dp_cons_prepare<<<1,1>>>(d_levelArray, d_buffer, d_idx, source);

	int children = 1;
	switch (config.solution) {
	case 3:
		if (DEBUG)
			fprintf(stdout, "warp level consolidation\n");
		bfs_kernel_dp_warp_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
												d_buffer, children, d_buffer, d_idx);
		//bfs_kernel_dp_warp_cons_back<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
		//										d_buffer, d_buffer, d_idx);
		//bfs_kernel_dp_warp_malloc_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
		//										d_buffer, d_buffer, d_idx);
		break;
	case 4:
		if (DEBUG)
			fprintf(stdout, "block level consolidation\n");
		bfs_kernel_dp_block_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
												d_buffer, children, d_buffer, d_idx);
		//bfs_kernel_dp_block_malloc_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
		//										d_buffer, d_buffer, d_idx);
		break;
	case 5:
		// queue and buffer are different
		// buffer stores the active working set
		unsigned int *d_queue;
		unsigned int *d_qidx;
		unsigned int *d_count;
		cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_queue, sizeof(unsigned int)*GM_BUFF_SIZE) );
		cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_qidx, sizeof(unsigned int)) );
		cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_count, sizeof(unsigned int)) );
		cudaCheckError(  __FILE__, __LINE__, cudaMemset( d_qidx, 0, sizeof(unsigned int)) );
		cudaCheckError(  __FILE__, __LINE__, cudaMemset( d_count, 0, sizeof(unsigned int)) );
		if (DEBUG)
    		fprintf(stdout, "grid level consolidation\n");
		// by default, it utilize malloc
		dp_grid_cons_init<<<1,1>>>();
		bfs_kernel_dp_grid_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
												d_buffer, d_idx, d_queue, d_qidx, d_count);
		/*	bfs_kernel_dp_grid_malloc_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
												d_buffer, d_idx, d_queue, d_qidx, d_count);
		*/
		break;
	default:
		printf("Unsopported solutions\n");
		exit(0);
	}
	cudaDeviceSynchronize();
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());

	if (DEBUG)
		printf("===> GPU #%d - nested parallelism consolidation %f.\n", config.solution, end_time-start_time);
	//gpu_print<<<1,1>>>(d_idx);
	cudaCheckError( __FILE__, __LINE__, cudaFree(d_buffer) );
	cudaCheckError( __FILE__, __LINE__, cudaFree(d_idx) );
#if (CONSOLIDATE_LEVEL==2)
	cudaCheckError( __FILE__, __LINE__, cudaFree(d_queue) );
	cudaCheckError( __FILE__, __LINE__, cudaFree(d_qidx) );
	cudaCheckError( __FILE__, __LINE__, cudaFree(d_count) );
#endif
}

// ----------------------------------------------------------
// version #6 - recursive GPU bfs
// ----------------------------------------------------------
void bfs_rec()
{
	cudaEvent_t start, stop;
	float time;
	/* prepare GPU */

	int children = graph.vertexArray[source+1]-graph.vertexArray[source];
	unsigned block_size = min (children, THREADS_PER_BLOCK);
	if (DEBUG)
		printf("===> GPU #%d - BFS rec. gridSize: %d blockSize: %d\n", config.solution, 1, block_size);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	bfs_kernel_rec<<<1,block_size>>>(source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time: %.2f ms", time);

	if (DEBUG)
		printf("===> GPU #%d - BFS rec.\n", config.solution);
}
// ----------------------------------------------------------
// version #7 - flat parallelism  pure GPU
// ----------------------------------------------------------

void bfs_flat_pure_gpu()
{
	cudaEvent_t start, stop;
	float time;
	/* prepare GPU */
	// bool queue_empty = false;
	// bool *d_queue_empty;

	// cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_queue_empty, sizeof(bool)) );
  //printf("Grid configuration gridxblocks, %d x %d\n", NUM_BLOCKS_FLAT, THREADS_PER_BLOCK_FLAT);
	if (DEBUG)
		printf("===> GPU #%d - flat pure gpu parallelism. gridSize: 1, blockSize: 1\n", config.solution);
	unsigned level = 0;

	//level-based traversal
	// while (!queue_empty){
		// cudaCheckError(  __FILE__, __LINE__, cudaMemset( d_queue_empty, true, sizeof(bool)) );
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		bfs_kernel_flat_gpu<<<1, 1>>>(level, noNodeTotal, d_vertexArray, d_edgeArray, d_levelArray);
		cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
		// cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( &queue_empty, d_queue_empty, sizeof(bool), cudaMemcpyDeviceToHost) );
		cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		//Display time
		cudaEventElapsedTime(&time, start, stop);
		printf("\tParallel Job time: %.2f ms\n", time);
		// level++;
	// }

	if (DEBUG)
		printf("===> GPU #%d - flat pure gpu parallelism.\n", config.solution);

}

// ----------------------------------------------------------
// version #8 - GPU rec BFS Optimized
// ----------------------------------------------------------

void bfs_flat_rec_optimized()
{
	cudaEvent_t start, stop;
	float time;
	/* prepare GPU */
	// bool queue_empty = false;
	// bool *d_queue_empty;

	//  cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_queue_empty, sizeof(bool)) );
  // printf("Grid configuration gridxblocks, %d x %d\n", NUM_BLOCKS_FLAT, THREADS_PER_BLOCK_FLAT);
	unsigned level = 0;
	if (DEBUG)
		printf("===> GPU #%d - rec gpu optimized parallelism. gridSize: %d, blockSize: %d\n", config.solution, 1, 32);
	// unsigned block_size = min(THREADS_PER_BLOCK, noNodeTotal);
	// unsigned grid_size = (noNodeTotal+ block_size-1)/block_size;

	//level-based traversal
	// cudaCheckError(  __FILE__, __LINE__, cudaMemset( d_queue_empty, false, sizeof(bool)) );
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	bfs_kernel_recOptimized<<<1, 32>>>(level, noNodeTotal, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	// cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( &queue_empty, d_queue_empty, sizeof(bool), cudaMemcpyDeviceToHost) );
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job time: %.2f ms\n", time);

	if (DEBUG)
		printf("===> GPU #%d - rec gpu optimized parallelism.\n", config.solution);

}
void BFS_REC_GPU()
{
	cudaCheckError( __FILE__, __LINE__, cudaSetDevice(config.device_num) );
	cudaCheckError( __FILE__, __LINE__, cudaDeviceReset());
	prepare_gpu();

#ifdef GPU_PROFILE
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif

#ifdef GPU_WORKEFFICIENCY
	reset_gpu_statisticsWE<<<1,1>>>();
	cudaDeviceSynchronize();
#endif

	start_time = gettime_ms();
	switch (config.solution) {
		case 0:  bfs_flat_gpu();	//
			break;
		case 1:  bfs_rec_dp_naive_gpu();	//
			break;
		case 2:  bfs_rec_dp_hier_gpu();	//
			break;
		case 3:
		case 4:
		case 5:  bfs_rec_dp_cons_gpu();	//
			break;
		case 6:  bfs_rec();	//
				break;
		case 7:  bfs_flat_pure_gpu();	//
						break;
		case 8:  bfs_flat_rec_optimized();	//
					break;
		default:
			break;
	}
	cudaCheckError( __FILE__, __LINE__, cudaDeviceSynchronize() );
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
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph.levelArray, d_levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime_ms();
	d2h_memcpy_time += end_time - start_time;

	clean_gpu();
}
