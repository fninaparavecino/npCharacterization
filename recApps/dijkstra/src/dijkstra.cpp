#include "dijkstra.h"

#define N 1

using namespace std;

CONF config;

double start_time = 0;
double end_time = 0;
double init_time = 0;
double d_malloc_time = 0;
double h2d_memcpy_time = 0;
double ker_exe_time = 0;
double d2h_memcpy_time = 0;

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
	fprintf(stderr,"Usage:  dijkstra [option]\n");
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "    --help,-h      print this message\n");
	fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
	fprintf(stderr, "    --debug,-d     enhanced verbosity level\n");
	fprintf(stderr, "\nOther:\n");
	fprintf(stderr, "    --import,-i <graph_file>           import graph file\n");
	fprintf(stderr, "    --format,-f <number>               specify the input format\n");
 	fprintf(stderr, "                 0 - DIMACS9\n");
	fprintf(stderr, "                 1 - DIMACS10\n");
	fprintf(stderr, "                 2 - SLNDC\n");
	fprintf(stderr, "    --solution,-s <number>             specify the solution\n");
	fprintf(stderr, "                 0 - GPU rec\n");
	fprintf(stderr, "    --device,-e <number>               select the device\n");
}

void print_conf() {
	fprintf(stderr, "\nCONFIGURATION:\n");
	if (config.graph_file) {
		fprintf(stderr, "- Graph file: %s\n", config.graph_file);
		fprintf(stderr, "- Graph format: %d\n", config.data_set_format);
 	}
	fprintf(stderr, "- Solution: %d\n", config.solution);
	fprintf(stderr, "- GPU Device: %d\n", config.device_num);
	if (config.verbose && config.debug) fprintf(stderr, "- verbose mode\n");
 	if (config.debug) fprintf(stderr, "- debug mode\n");
}

void init_conf() {
	config.data_set_format = 0;
	config.solution = 0;
	config.device_num = 0;
	config.verbose = false;
	config.debug = false;
}

int parse_arguments(int argc, char** argv) {
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
		else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) {
			VERBOSE = config.verbose = 1;
		}
		else if ( strcmp(argv[i], "-d")==0 || strcmp(argv[i], "--debug")==0 ) {
			DEBUG = config.debug = 1;
		}
		else if ( strcmp(argv[i], "-f")==0 || strcmp(argv[i], "--format")==0 ) {
			++i;
			config.data_set_format = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-s")==0 || strcmp(argv[i], "--solution")==0 ) {
			++i;
			config.solution = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-e")==0 || strcmp(argv[i], "--device")==0 ) {
			++i;
			config.device_num = atoi(argv[i]);
		}
		else if ( strcmp(argv[i], "-i")==0 || strcmp(argv[i], "--import")==0 ) {
			++i;
			if (i==argc) {
				fprintf(stderr, "Graph file name missing.\n");
			}
			config.graph_file = argv[i];
		}
		++i;
	}

	return 1;
}

void dijkstraCPU(int *parent)
{

}

void setArrays(int size, int *arrays, int value)
{
	for (int i=0; i<size; ++i) {
		arrays[i] = value;
	}
}

void validateArrays(int n, int *array1, int *array2, const char *message)
{
	int flag = 1;
	for (int node=0; node<n; node++){
		if (array1[node]!=array2[node]){
			printf("ERROR: validation error at %d: %s !\n", node, message);
			printf("Node %d: %d v.s. %d\n", node, array1[node], array2[node]);
			flag = 0; break;
		}
	}
	if (flag) printf("PASS: %s !\n", message);
}

int main(int argc, char* argv[])
{
	init_conf();
	if ( !parse_arguments(argc, argv) ) return 0;

	print_conf();

	if (config.graph_file!=NULL) {
		fp = fopen(config.graph_file, "r");
		if ( fp==NULL ) {
			fprintf(stderr, "Error: NULL file pointer.\n");
			return 1;
		}
	}
	else
		return 0;

	double time, end_time;
	time = gettime();
	switch(config.data_set_format) {
		case 0: readInputDIMACS9(); break;
		case 1: readInputDIMACS10(); break;
		case 2: readInputSLNDC(); break;
		default: fprintf(stderr, "Wrong code for dataset\n"); break;
	}
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "Import graph:\t\t%lf\n",end_time-time);

	time = gettime();
	convertCSR();
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "AdjList to CSR:\t\t%lf\n",end_time-time);

	/*
	printf("%d\n", noNodeTotal);
	for (int i=0; i<noNodeTotal; ++i) {
		int num_edge = graph.vertexArray[i+1] - graph.vertexArray[i];
		printf("%d ", num_edge);
	}
	printf("\n");*/

  int *parentMST = (int*) malloc(sizeof(int) * noNodeTotal);
  memset(parentMST, 0, noNodeTotal*sizeof(int));
  double wall0, wall1;
  wall0 = getWallTime();
  dijkstraCPU(parentMST);
  wall1 = getWallTime();
  printf("===MAIN=== :: Dijkstra CPU performance: %.2f ms\n", wall1- wall0);

  // Call GPU
  setArrays(noNodeTotal, graph.levelArray, UNDEFINED);
  setArrays(noNodeTotal, graph.keyArray, UNDEFINED);
  graph.keyArray[source] = 0;
  graph.levelArray[source] = -1;
  printf("====> source: %d\n", source);
  dijkstraGPU();

  //printMST(graph.levelArray);
  validateArrays(noNodeTotal, parentMST, graph.levelArray,  "parentMST versus graph.levelArray");
  if (VERBOSE) {
    fprintf(stdout, "===MAIN=== :: CUDA runtime init:\t\t%.2lf ms.\n", init_time/N);
    fprintf(stdout, "===MAIN=== :: CUDA device cudaMalloc:\t\t%.2lf ms.\n", d_malloc_time/N);
    fprintf(stdout, "===MAIN=== :: CUDA H2D cudaMemcpy:\t\t%.2lf ms.\n", h2d_memcpy_time/N);
    fprintf(stdout, "===MAIN=== :: CUDA kernel execution:\t\t%.2lf ms.\n", ker_exe_time/N);
    fprintf(stdout, "===MAIN=== :: CUDA D2H cudaMemcpy:\t\t%.2lf ms.\n", d2h_memcpy_time/N);
  }

	//starts execution
	printf("\n===MAIN=== :: [num_nodes,num_edges] = %u, %u\n", noNodeTotal, noEdgeTotal);

	clear();
	return 0;
}
