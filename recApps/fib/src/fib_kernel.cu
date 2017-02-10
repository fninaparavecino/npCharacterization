// fib parallel
__global__ void fib_kernel_plain(int n, long int* vFib){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid > n)
        return;
    
    if (n == 0 || n == 1){
        return;
    }
    
    for(int i=tid+2; i < tid*32; i++){
        vFib[i] = vFib[i-1] + vFib[i-2];
    }
}
