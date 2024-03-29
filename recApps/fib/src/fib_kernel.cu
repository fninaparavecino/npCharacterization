// fib parallel
#include <stdio.h>

#define cudaErrCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void fib_kernel_plain(int n, unsigned long int* vFib){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid > n/32)
        return;

    if (n == 0 || n == 1){
        return;
    }
    //printf("===KERNEL=== ThreadIdx: %d\n", tid);
    for(int i=tid*32 + 2; i <= n && i < tid*32 + 32; i++){
        vFib[i] = vFib[i-1] + vFib[i-2];
        //printf("===KERNEL=== fib of %d: %ld\n", i, vFib[i]);
    }
}

__global__ void fib_kernel_rec(int n, unsigned long int* vFib){
    if (n == 0 || n == 1)
        return;

    if (vFib[n-1] != 0){
      vFib[n] = vFib[n-1] + vFib[n-2];
    }
    else{
        fib_kernel_rec<<<1, 1>>>(n-1, vFib);
    }
}

__global__ void fib_kernel_par_rec(int n, unsigned long int* vFib){
    if (n == 0 || n == 1)
        return;

    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    fib_kernel_par_rec<<<1, 1>>>(n-2, vFib);
    fib_kernel_par_rec<<<1, 1>>>(n-1, vFib);
    cudaDeviceSynchronize();
    vFib[n] = vFib[n-1] + vFib[n-2];

    // if (vFib[n-1] != 0){
    //   printf("Idx: %d with vFib[%d] : %d\n", idx, n-1, vFib[n-1]);
    //   vFib[n] = vFib[n-1] + vFib[n-2];
    //   printf("Idx: %d with vFib[%d] : %d\n", idx, n, vFib[n]);
    // }

}

void fibGPU(int n, unsigned long int* arrayN)
{
  // time variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  // device arrayN
  unsigned long int *devArrayN = 0;

  // define device
  cudaErrCheck(cudaSetDevice(0));

  // cuda malloc for devArrayN
  cudaErrCheck(cudaMalloc((void**)&devArrayN, sizeof(unsigned long int)*(n+1)));

  // cuda memcopy
  cudaErrCheck(cudaMemcpy(devArrayN, arrayN, sizeof(unsigned long int)*(n+1), cudaMemcpyHostToDevice));

  // call the kernel
  dim3 threadsPerBlock(32, 1, 1);
  dim3 blocksPerGrid((n+31)/32, 1, 1);

  printf("Launching fib_kernel (%d x %d)...\n", blocksPerGrid.x, threadsPerBlock.x);
  cudaEventRecord(start, 0);
  fib_kernel_plain<<<blocksPerGrid, threadsPerBlock>>>(n, devArrayN);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  //Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Kernel time: %.2f ms\n", time);

  // retrieve results
  cudaErrCheck(cudaMemcpy(arrayN, devArrayN, sizeof(unsigned long int)*(n+1), cudaMemcpyDeviceToHost));

  //Free resource
  cudaFree(devArrayN);
}

void fibGPURec(int n, unsigned long int* arrayN)
{
  // time variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  // device arrayN
  unsigned long int *devArrayN = 0;

  // define device
  cudaErrCheck(cudaSetDevice(0));

  // cuda malloc for devArrayN
  cudaErrCheck(cudaMalloc((void**)&devArrayN, sizeof(unsigned long int)*(n+1)));

  // cuda memcopy
  cudaErrCheck(cudaMemcpy(devArrayN, arrayN, sizeof(unsigned long int)*(n+1), cudaMemcpyHostToDevice));

  // call the kernel
  dim3 threadsPerBlock(1, 1, 1);
  dim3 blocksPerGrid(1, 1, 1);

  printf("Launching fib_kernel_rec (%d x %d)...\n", blocksPerGrid.x, threadsPerBlock.x);
  cudaEventRecord(start, 0);
  fib_kernel_rec<<<blocksPerGrid, threadsPerBlock>>>(n, devArrayN);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  //Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Kernel time: %.2f ms\n", time);

  // retrieve results
  cudaErrCheck(cudaMemcpy(arrayN, devArrayN, sizeof(unsigned long int)*(n+1), cudaMemcpyDeviceToHost));

  //Free resource
  cudaFree(devArrayN);
}
void fibGPUParRec(int n, unsigned long int* arrayN)
{
  // time variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  // device arrayN
  unsigned long int *devArrayN = 0;

  // define device
  cudaErrCheck(cudaSetDevice(0));

  // cuda malloc for devArrayN
  cudaErrCheck(cudaMalloc((void**)&devArrayN, sizeof(unsigned long int)*(n+1)));

  // cuda memcopy
  cudaErrCheck(cudaMemcpy(devArrayN, arrayN, sizeof(unsigned long int)*(n+1), cudaMemcpyHostToDevice));

  // call the kernel
  dim3 threadsPerBlock(1, 1, 1);
  dim3 blocksPerGrid(1, 1, 1);

  printf("Launching fib_kernel_par_rec (%d x %d)...\n", blocksPerGrid.x, threadsPerBlock.x);
  cudaEventRecord(start, 0);
  fib_kernel_par_rec<<<blocksPerGrid, threadsPerBlock>>>(n, devArrayN);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  //Display time
	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Kernel time: %.2f ms\n", time);

  // retrieve results
  cudaErrCheck(cudaMemcpy(arrayN, devArrayN, sizeof(unsigned long int)*(n+1), cudaMemcpyDeviceToHost));

  //Free resource
  cudaFree(devArrayN);
}
