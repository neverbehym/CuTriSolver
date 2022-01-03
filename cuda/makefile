
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand


NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math -std=c++11 
all:		BS tridiagonal_test

BS:	BS.cu tridiagonal_solver.h makefile
	nvcc BS.cu -o BS $(INC) $(NVCCFLAGS) $(LIB)
tridiagonal_test:	tridiagonal_test.cu tridiagonal_solver.h
	nvcc tridiagonal_test.cu  -o tridiagonal_test $(INC) $(NVCCFLAGS) $(LIB)


clean:
	rm -f BS tridiagonal_test
