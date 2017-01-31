#!/bin/bash

BIN=../src/gpu-bfs-rec
DATA_FILE=../input/data_test.gr

	EXE=$BIN
	$EXE -f 0 -i ${DATA_FILE} -v -s 0 #>> ${LOG_FILE}
#	$EXE -f 0 -i ${DATA_FILE} -v -s 4 #>> ${LOG_FILE}
#	$EXE -f 0 -i ${DATA_FILE} -v -s 5 #>> ${LOG_FILE}
