cc=g++
exe=main
obj=main.o FDMethod.o PDE_solver.o Option.o

$(exe):$(obj)
	$(cc) -o $(exe) $(obj)
main.o:main.cpp
	$(cc) -std=c++11 -c main.cpp
FDMethod.o:FDMethod.cpp
	$(cc) -c FDMethod.cpp
PDE_solver.o:PDE_solver.cpp
	$(cc) -c PDE_solver.cpp
Option.o:Option.cpp
	$(cc) -c Option.cpp

clean:
	rm -rf *.o main
