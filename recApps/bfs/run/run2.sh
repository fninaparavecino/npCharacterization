#!/bin/bash

BIN=../src/gpu-bfs-rec
DATA_FILE=../input/data_test2.gr

	EXE=$BIN
	$EXE -f 0 -i ${DATA_FILE} -v -s 0 #>> ${LOG_FILE}
	$EXE -f 0 -i ${DATA_FILE} -v -s 1 #>> ${LOG_FILE}
	$EXE -f 0 -i ${DATA_FILE} -v -s 6 #>> ${LOG_FILE}
	$EXE -f 0 -i ${DATA_FILE} -v -s 7 #>> ${LOG_FILE}
	$EXE -f 0 -i ${DATA_FILE} -v -s 8 #>> ${LOG_FILE}
