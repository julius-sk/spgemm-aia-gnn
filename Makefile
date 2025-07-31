# Makefile for SpGEMM Comparison Testing
# Compares Hash without AIA, Hash with AIA, and cuSPARSE

# CUDA and compiler settings
NVCC = nvcc
CXX = g++
CUDA_PATH = /usr/local/cuda
NSPARSE_PATH = ./nsparse/cuda-cpp

# Compiler flags
NVCCFLAGS = -O3 -arch=sm_70 -std=c++14 -Xcompiler -fopenmp
CXXFLAGS = -O3 -std=c++14 -fopenmp

# Include paths
INCLUDES = -I$(CUDA_PATH)/include \
           -I$(NSPARSE_PATH)/inc \
           -I$(CUDA_PATH)/samples/common/inc \
           -I.

# Library paths and libraries
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcusparse -lcurand -lgomp

# Source files
SOURCES = spgemm_comparison_test.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Target executable
TARGET = spgemm_comparison_test

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

# Compile .cu files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Test with sample data path
test: $(TARGET)
	./$(TARGET) /path/to/spgemm-pruning/kernels/graphs

# Install dependencies (if needed)
install-deps:
	@echo "Make sure you have:"
	@echo "1. CUDA toolkit installed"
	@echo "2. nsparse library in ./nsparse/cuda-cpp/"
	@echo "3. spgemm-pruning dataset in specified path"
	@echo "4. Graph files in .indices/.indptr format"

# Help target
help:
	@echo "SpGEMM Comparison Build System"
	@echo "=============================="
	@echo "Targets:"
	@echo "  all         - Build the comparison executable"
	@echo "  clean       - Remove build files"
	@echo "  test        - Run test (update path)"
	@echo "  install-deps- Show dependency requirements"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  ./spgemm_comparison_test /path/to/spgemm-pruning/kernels/graphs"
	@echo ""
	@echo "Expected data format:"
	@echo "  /path/to/graphs/reddit.indices"
	@echo "  /path/to/graphs/reddit.indptr"
	@echo "  /path/to/graphs/flickr.indices"
	@echo "  /path/to/graphs/flickr.indptr"
	@echo "  etc."

.PHONY: all clean test install-deps help
