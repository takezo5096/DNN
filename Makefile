CC=g++
NVCC=nvcc
CUDA_TOP=/usr/local/cuda
INC=-I$(CUDA_TOP)/include -I../cuMat
LIB=-L$(CUDA_TOP)/lib64 -L../cuMat -lcublas -lcudart -lcudnn -lm -lcumat -lcusparse -lboost_serialization -lmecab -lboost_system -lpng
#OTHER_OPTS=-std=c++11 -pthread -O2 -fpermissive
OTHER_OPTS=-std=c++11 -pthread -O2

test: test.o variable.o function.o dataset.o mnist.o optimizer.o graph.o
	$(CC) -o test test.o variable.o function.o dataset.o mnist.o optimizer.o graph.o $(INC) $(LIB) $(OTHER_OPTS)

graph.o: graph.cpp
	$(CC) $(INC) $(OTHER_OPTS) $(LIB) -c graph.cpp

variable.o: variable.cpp
	$(CC) $(INC) $(OTHER_OPTS) $(LIB) -c variable.cpp

function.o: function.cpp
	$(CC) $(INC) $(OTHER_OPTS) $(LIB) -c function.cpp

dataset.o: dataset.cpp
	$(CC) $(INC) $(OTHER_OPTS) $(LIB) -c dataset.cpp

mnist.o: mnist.cpp
	$(CC) $(INC) $(OTHER_OPTS) $(LIB) -c mnist.cpp

optimizer.o: optimizer.cpp
	$(CC) $(INC) $(OTHER_OPTS) $(LIB) -c optimizer.cpp

test.o: test.cpp
	$(CC) -c test.cpp $(INC) $(OTHER_OPTS) $(LIB)


clean:
	rm -f test
	rm -f *.o

