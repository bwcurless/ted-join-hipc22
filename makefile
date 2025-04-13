# Optional parameters to pass in
#
OUTPUT_NEIGHBORS ?= false
# Input data dimensionality (i.e., number of features of the dataset).
INPUT_DATA_DIM ?= 18
# Dimensionality used for the tensor cores, based on the dimensionality of the
# data. Should fit the size of the matrices used by the tensor cores, depending
# on the precision and configuration. Typically, the next multiple of 8 or 16
# of INPUT_DATA_DIM.
COMPUTE_DIM ?= 32

SOURCES = utils.cu dataset.cu index.cu main.cu sort_by_workload.cu gpu_join.cu kernel_join.cu
OBJECTS = utils.o dataset.o index.o sort_by_workload.o gpu_join.o kernel_join.o main.o
CC = nvcc
EXECUTABLE = main

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -DOUTPUT_NEIGHBORS=$(OUTPUT_NEIGHBORS) -DINPUT_DATA_DIM=$(INPUT_DATA_DIM) -DCOMPUTE_DIM=$(COMPUTE_DIM)

# Add or modify the rules below to fit your architecture
ampere:
	echo "Compiling for Ampere generation (CC=86)"
	$(MAKE) all ARCH=compute_86 CODE=sm_86 BOOST=/home/benoit/research/boost_1_75_0

turing:
	echo "Compiling for Turing generation (CC=75)"
	$(MAKE) all ARCH=compute_75 CODE=sm_75 BOOST=/home/bgallet/boost_1_75_0

monsoon:
	echo "Compiling for Monsoon cluster with A100 (CC=80)"
	$(MAKE) all ARCH=compute_80 CODE=sm_80 BOOST=/home/bc2497/cuSimSearch/boost_1_76_0

all: $(EXECUTABLE)

%.o: %.cu
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(BOOST) $^ -c $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(BOOST) $^ -o $@

clean:
	rm -f $(OBJECTS)
	rm -f $(EXECUTABLE)
