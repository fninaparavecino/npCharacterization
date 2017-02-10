#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fib.h"
using namespace std;

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
long long int fib(int n){
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
long long int fibSeq(int n, long int *arrayNSeq){

    for (int i = 2; i <= n; i++){
      arrayNSeq[i] = arrayNSeq[i-1]+ arrayNSeq[i-2];
    }
    return arrayNSeq[n];
}
void verifyArrays(int n, long int *array1, long int *array2, const char *message)
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

    // seeds
    long int arraySeed[4] = {0, 1, 2178309, 3524578};

    if ( !parse_arguments(argc, argv, &n) ) return 0;

    printf("Computing fib(%d)\n", n);

    // printf("Fib of %d: %lld\n", n, fib(n)); // Commenting out, due to it is taking long time to compute

    long int* arrayNSeq = (long int*)malloc(sizeof(long int) * (n+1));
    memset(arrayNSeq, 0, (n+1)*sizeof(long int));
    arrayNSeq[0] = 0; arrayNSeq[1] = 1;

    fibSeq(n, arrayNSeq);
    printf("Fib Seq of %d: %ld\n", n, arrayNSeq[n]);

    // Call GPU kernel
    long int* arrayN = (long int*)malloc((n+1)*sizeof(long int));
    memset(arrayN, 0, (n+1)*sizeof(long int));
    int auxIndex = 0;
    for(int i=0; i <= n; i=i+32){
      arrayN[i] = arraySeed[auxIndex];
      arrayN[i+1] = arraySeed[auxIndex+1];
      auxIndex += 2;
    }
    fibGPU(n, arrayN);

    printf("Fib GPU of %d: %ld\n", n, arrayN[n]);

    verifyArrays(n+1, arrayNSeq, arrayN, "CPU vs GPU");

    free(arrayN); free(arrayNSeq);
    return 0;
}
