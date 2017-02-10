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

__global__ void fib_kernel_plain(int n, long int* vFib){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n/32)
        return;

    if (n == 0 || n == 1){
        return;
    }
    printf("ThreadIdx: %d\n", tid);
    for(int i=tid*32 + 2; i < tid*32 + 32; i++){
        vFib[i] = vFib[i-1] + vFib[i-2];
        printf("fib of %d: %ld\n", i, vFib[i]);
    }
}

void fibGPU(int n, long int* arrayN)
{
  long int *devArrayN = 0;

  // define device
  cudaErrCheck(cudaSetDevice(0));

  // cuda malloc for devArrayN
  cudaErrCheck(cudaMalloc((void**)&devArrayN, sizeof(long int)*n));

  // cuda memcopy
  cudaErrCheck(cudaMemcpy(devArrayN, arrayN, sizeof(long int)*n, cudaMemcpyHostToDevice));

  // call the kernel
  dim3 threadsPerBlock(32, 1, 1);
  dim3 blocksPerGrid((n+31)/32, 1, 1);
  printf("Launching fib_kernel (%d x %d)", blocksPerGrid.x, threadsPerBlock.x);
  fib_kernel_plain<<<blocksPerGrid, threadsPerBlock>>>(n, devArrayN);
  cudaErrCheck(cudaDeviceSynchronize());

  // retrieve results
  cudaErrCheck(cudaMemcpy(arrayN, devArrayN, sizeof(long int)*n, cudaMemcpyDeviceToHost));

  //Free resource
  cudaFree(devArrayN);

}
