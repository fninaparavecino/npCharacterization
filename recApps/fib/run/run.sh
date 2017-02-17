#!/bin/bash
counter=1
while [ $counter -le 31 ]
do
	echo $counter
	../src/fib -n $counter
	((counter++))
done
