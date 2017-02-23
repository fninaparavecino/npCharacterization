#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "fib.h"
using namespace std;

double getWallTime(){
        struct timeval time;
        if(gettimeofday(&time,NULL)){
                printf("Error getting time\n");
                return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void usage() {
    fprintf(stderr,"\n");
    fprintf(stderr,"Usage:  fib [option]\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "    --help,-h      print this message\n");
    fprintf(stderr, "    --number,-n   fibonacci(number) to compute\n");
}
int parse_arguments(int argc, char** argv, int *n) {
    int i = 1;
    if ( argc<2 ) {
        usage();
        return 0;
    }
    while ( i<argc ) {
        if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) {
            usage();
            return 0;
        }
        else if ( strcmp(argv[i], "-n")==0 || strcmp(argv[i], "--number")==0 ) {
            ++i;
            if (i==argc) {
                fprintf(stderr, "Value number is missing.\n");
            }
            *n = atoi(argv[i]);
        }
        else{
            fprintf(stderr,"Input parameters invalid\n");
            return 0;
        }
        ++i;
    }

    return 1;
}
long long int fib(long long int n){
    if (n == 0){
        return 0;
    }
    else if (n == 1){
        return 1;
    }
    else{
        return fib(n-1) + fib(n-2);
    }
}
long long int fibSeq(int n, unsigned long int *arrayNSeq){

    for (int i = 2; i <= n; i++){
      arrayNSeq[i] = arrayNSeq[i-1]+ arrayNSeq[i-2];
    }
    return arrayNSeq[n];
}
long long int fibRecMemo(int n, unsigned long int* array){
  if (n == 0 || n == 1)
    return array[n];
  if (array[n] != 0)
    return array[n];

  array[n] = fibRecMemo(n-1, array) + array[n-2];

  return array[n];
}
void verifyArrays(int n, unsigned long int *array1, unsigned long int *array2, const char *message)
{
	int flag = 1;
	for (int i=0; i < n; i++){
		if (array1[i]!=array2[i]){
			printf("ERROR: validation error at %d: %s !\n", i, message);
			printf("Index %d : %ld vs %ld\n", i, array1[i], array2[i]);
			flag = 0; break;
		}
	}
	if (flag) printf("PASS: %s !\n", message);
}

int main(int argc, char** argv){

    int n = 0;
    double wall0, wall1;

    // seeds
    unsigned long int arraySeed[6] = {0, 1, 2178309, 3524578, 10610209857723, 17167680177565};

    if ( !parse_arguments(argc, argv, &n) ) return 0;

    printf("Computing fib(%d)\n", n);
    wall0 = getWallTime();
    fib(n); // it takes time
    wall1 = getWallTime();
    printf("Fib Rec\n\tTime Performance: %f ms\n", wall1 - wall0);

    // Sequential using iterative approach
    unsigned long int* arrayNSeq = (unsigned long int*)malloc(sizeof(unsigned long int) * (n+1));
    memset(arrayNSeq, 0, (n+1)*sizeof(unsigned long int));
    arrayNSeq[0] = 0; arrayNSeq[1] = 1;

    wall0 = getWallTime();
    fibSeq(n, arrayNSeq);
    wall1 = getWallTime();
    printf("Fib Seq \n\tTime Performance: %f ms\n", (wall1- wall0));

    //Sequential using recursive with memoization approach
    unsigned long int* arrayNRec = (unsigned long int*) malloc(sizeof(unsigned long int) * (n+1));
    memset(arrayNRec, 0, (n+1)*sizeof(unsigned long int));
    arrayNRec[0] = 0; arrayNRec[1] = 1;
    wall0 = getWallTime();
    fibRecMemo(n, arrayNRec);
    wall1 = getWallTime();
    printf("Fib Rec Memo\n\tTime Performance: %f ms\n", (wall1- wall0));
    verifyArrays(n+1, arrayNSeq, arrayNRec, "\tCPU Seq vs CPU Rec Memo");

    // Call GPU kernel
    unsigned long int* arrayN = (unsigned long int*)malloc((n+1)*sizeof(unsigned long int));
    memset(arrayN, 0, (n+1)*sizeof(unsigned long int));
    int auxIndex = 0;
    for(int i=0; i <= n; i=i+32){
      arrayN[i] = arraySeed[auxIndex];
      arrayN[i+1] = arraySeed[auxIndex+1];
      auxIndex += 2;
    }
    fibGPU(n, arrayN);

    // Verify results
    verifyArrays(n+1, arrayNSeq, arrayN, "CPU vs GPU");

    // // Call Rec GPU Kernel
    // unsigned long int* arrayNGPURec = (unsigned long int*)malloc((n+1)*sizeof(unsigned long int));
    // memset(arrayNGPURec, 0, (n+1)*sizeof(unsigned long int));
    // arrayNGPURec[0] = 0; arrayNGPURec[1] = 1;
    // fibGPURec(n, arrayNGPURec);
    // // Verify results
    // verifyArrays(n+1, arrayNSeq, arrayNGPURec, "CPU vs GPURec");

    // Call Rec GPU Kernel
    unsigned long int* arrayNGPUParRec = (unsigned long int*)malloc((n+1)*sizeof(unsigned long int));
    memset(arrayNGPUParRec, 0, (n+1)*sizeof(unsigned long int));
    arrayNGPUParRec[0] = 0; arrayNGPUParRec[1] = 1;
    fibGPUParRec(n, arrayNGPUParRec);
    // Verify results
    verifyArrays(n+1, arrayNSeq, arrayNGPUParRec, "CPU vs GPUParRec");

    // Free memory
    free(arrayN); free(arrayNSeq);
    // free(arrayNGPURec);
    free(arrayNGPUParRec);
    return 0;
}
