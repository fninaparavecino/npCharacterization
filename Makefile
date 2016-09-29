npCase1:
	nvcc -arch=compute_35 -code=sm_35 -rdc=true npMain.cu -o npCase1 -lcudadevrt

npCase2:
	nvcc -arch=compute_35 -code=sm_35 -rdc=true npCase2.cu -o npCase2 -lcudadevrt
	
clean:
	rm -rf npCase1 npCase2
