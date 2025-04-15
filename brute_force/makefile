# Optional parameters to pass in
# Input data dimensionality (i.e., number of features of the dataset).
INPUT_DATA_DIM ?= 18
# Dimensionality used for the tensor cores, based on the dimensionality of the
# data. Should fit the size of the matrices used by the tensor cores, depending
# on the precision and configuration. Typically, the next multiple of 8 or 16
# of INPUT_DATA_DIM.
COMPUTE_DIM ?= 32

SOURCES = utils.cu dataset.cu kernel_join.cu dmmaTensorCoresGemm.cu wmmaTensorCoresGemm.cu gpu_join.cu main.cu
OBJECTS = utils.o dataset.o kernel_join.o dmmaTensorCoresGemm.o wmmaTensorCoresGemm.o gpu_join.o main.o
CC = nvcc
EXECUTABLE = main

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES  -DINPUT_DATA_DIM=$(INPUT_DATA_DIM) -DCOMPUTE_DIM=$(COMPUTE_DIM)

ampere:
	echo "Compiling for Ampere generation (CC=86)"
	$(MAKE) all ARCH=compute_86 CODE=sm_86 LIBS="/home/benoit/dev/cuda-samples/Common"

turing:
	echo "Compiling for Turing generation (CC=75)"
	$(MAKE) all ARCH=compute_75 CODE=sm_75 LIBS="/home/bgallet/cuda-samples/Common"

monsoon:
	echo "Compiling for Monsoon cluster with A100 (CC=80)"
	$(MAKE) all ARCH=compute_80 CODE=sm_80 LIBS="/home/bc2497/cuSimSearch/cuda-samples/Common"

all: $(EXECUTABLE)

%.o: %.cu
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(LIBS) $^ -c $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(LIBS) $^ -o $@

clean:
	rm -f $(OBJECTS)
	rm -f $(EXECUTABLE)
