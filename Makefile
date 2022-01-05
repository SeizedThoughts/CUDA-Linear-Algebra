#build main.c

#compiler
CC = nvcc
#target file name
LIB_NAME = cuda-linalg
all:
	$(CC) ./src/$(LIB_NAME).cu -c -o ./bin/$(LIB_NAME).o -I ./includes
	$(CC) ./tests/main.cu ./bin/$(LIB_NAME).o -o ./tests/main -I ./includes