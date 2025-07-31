# Makefile for SpGEMM Comparison Testing
# Following nsparse makefile pattern

CXX = nvcc
NVCC = nvcc

CFLAGS = -O3 -g
CFLAGS += -L. ${REAL} -lm
LDFLAGS = ${CFLAGS}

# for Device Code
CUDA_PATH = /usr/local/cuda
LDFLAGS += -L${CUDA_PATH}/lib64
LDFLAGS += -arch=sm_70 -lcudart -lcusparse
INCLUDE = -I./nsparse/cuda-cpp/inc
INCLUDE += -I${CUDA_PATH}/include
INCLUDE += -I${CUDA_PATH}/samples/common/inc
INCLUDE += -I.

BIN = ./bin
SRC = ./
OBJ = ./obj

OBJ_SUF = .o
OS_SUF = .s.o
OD_SUF = .d.o
TS_SUF = _s
TD_SUF = _d

SAMPLE_COMPARISON = spgemm_comparison_test.cu
SAMPLE_COMPARISON_TARGET = $(SAMPLE_COMPARISON:%.cu=%)

all: comparison

comparison: $(SAMPLE_COMPARISON_TARGET)$(TS_SUF) $(SAMPLE_COMPARISON_TARGET)$(TD_SUF)

$(BIN)/%$(TS_SUF): $(OBJ)/%$(OS_SUF)
	mkdir -p $(dir $@)
	$(NVCC) -o $@ $^ $(LDFLAGS) $(INCLUDE)

$(BIN)/%$(TD_SUF): $(OBJ)/%$(OD_SUF)
	mkdir -p $(dir $@)
	$(NVCC) -o $@ $^ $(LDFLAGS) $(INCLUDE)

%$(TS_SUF): $(OBJ)/%$(OS_SUF)
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $^ $(LDFLAGS) $(INCLUDE)

%$(TD_SUF): $(OBJ)/%$(OD_SUF)
	mkdir -p $(BIN)
	$(NVCC) -o $(BIN)/$@ $^ $(LDFLAGS) $(INCLUDE)

$(OBJ)/%$(OS_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DFLOAT $(LDFLAGS) $(INCLUDE) -o $@ $<

$(OBJ)/%$(OD_SUF) : $(SRC)/%.cu
	mkdir -p $(dir $@)
	$(NVCC) -c -DDOUBLE $(LDFLAGS) $(INCLUDE) -o $@ $

clean:
	rm -rf $(BIN)/*
	rm -rf $(OBJ)/*

# Test targets
test_s: spgemm_comparison_test$(TS_SUF)
	$(BIN)/spgemm_comparison_test$(TS_SUF) /path/to/spgemm-pruning/kernels/graphs

test_d: spgemm_comparison_test$(TD_SUF)
	$(BIN)/spgemm_comparison_test$(TD_SUF) /path/to/spgemm-pruning/kernels/graphs

# Help target
help:
	@echo "SpGEMM Comparison Build System (following nsparse pattern)"
	@echo "=========================================================="
	@echo "Targets:"
	@echo "  all           - Build both single and double precision"
	@echo "  comparison    - Same as all"
	@echo "  clean         - Remove build files"
	@echo "  test_s        - Run single precision test"
	@echo "  test_d        - Run double precision test"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Output executables:"
	@echo "  bin/spgemm_comparison_test_s  (single precision)"
	@echo "  bin/spgemm_comparison_test_d  (double precision)"
	@echo ""
	@echo "Usage:"
	@echo "  make"
	@echo "  ./bin/spgemm_comparison_test_s /path/to/spgemm-pruning/kernels/graphs"
	@echo ""
	@echo "Expected data format:"
	@echo "  /path/to/graphs/reddit.indices"
	@echo "  /path/to/graphs/reddit.indptr"
	@echo "  /path/to/graphs/flickr.indices"
	@echo "  /path/to/graphs/flickr.indptr"
	@echo "  etc."

.PHONY: all comparison clean test_s test_d help
