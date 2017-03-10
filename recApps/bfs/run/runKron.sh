#!/bin/bash

BIN=../src/gpu-bfs-rec
DATA_FILE=../input/kron_g500-simple-logn16.graph

	EXE=$BIN
	$EXE -f 1 -i ${DATA_FILE} -v -s 0 #>> ${LOG_FILE}
	$EXE -f 1 -i ${DATA_FILE} -v -s 1 #>> ${LOG_FILE}
	$EXE -f 1 -i ${DATA_FILE} -v -s 6 #>> ${LOG_FILE}
	$EXE -f 1 -i ${DATA_FILE} -v -s 7 #>> ${LOG_FILE}
	$EXE -f 1 -i ${DATA_FILE} -v -s 8 #>> ${LOG_FILE}
