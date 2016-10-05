all:
	nvcc -G -arch=compute_20 -rdc=true npCase1.cu -o npCase1 -lcudadevrt
	nvcc -G -arch=compute_35 -code=compute_35,sm_35,sm_52,sm_61 -rdc=true npCase2.cu -o npCase2 -lcudadevrt
	nvcc -G -arch=compute_35 -code=compute_35,sm_35,sm_52,sm_61 -rdc=true npCase3.cu -o npCase3-lcudadevrt
npCase1:
	nvcc -arch=compute_35 -code=sm_35 -rdc=true npCase1.cu -o npCase1 -lcudadevrt

npCase2:
	nvcc -arch=compute_35 -code=compute_35,sm_35,sm_52,sm_61 -rdc=true npCase2.cu -o npCase2 -lcudadevrt

npCase3:
	nvcc -lineinfo -arch=compute_35 -code=compute_35,sm_35,sm_52,sm_61 -rdc=true npCase3.cu -o npCase3 -lcudadevrt
	
npCase3CFG:
	cuobjdump npCase3 -xelf npCase3.3.sm_35.cubin
	nvdisasm npCase3.3.sm_35.cubin -g > sm_35_80.txt
	nvdisasm npCase3.3.sm_35.cubin -cfg > sm_35_80.dot
	dot -Tps -o sm_35_80.ps sm_35_80.dot
		
clean:
	rm -rf npCase1 npCase2 npCase3
