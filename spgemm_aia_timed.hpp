// spgemm_comparison_test.cu - EXACT copy of original pattern
// Just copy spgemm_hash.cu three times with different implementations

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string>
#include <iostream>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <SpGEMM.hpp>
#include <HashSpGEMM_volta.hpp>      // AIA implementation
#include <HashSpGEMM_volta_old.hpp>  // Non-AIA implementation

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

// EXACT copy of original spgemm_hash function - just with AIA
template <class idType, class valType>
void spgemm_hash_aia(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    idType i;
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);

    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            c.release_csr();
        }
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash(a, b, c);  // Uses volta.hpp (AIA version)
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash with AIA): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    c.memcpyDtH();
    c.release_csr();

    a.release_csr();
    b.release_csr();

    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

// EXACT copy of original spgemm_hash function - no AIA 
template <class idType, class valType>
void spgemm_hash_old(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    idType i;
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);

    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            c.release_csr();
        }
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash(a, b, c);  // Uses volta_old.hpp (no AIA version)
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash without AIA): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    c.memcpyDtH();
    c.release_csr();

    a.release_csr();
    b.release_csr();

    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

// EXACT copy of original cusparse pattern
template <class idType, class valType>
void spgemm_cusparse_test(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    idType i;
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);

    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            c.release_csr();
        }
        cudaEventRecord(event[0], 0);
        SpGEMM_cuSPARSE(a, b, c);  // Uses cuSPARSE
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (cuSPARSE): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    c.memcpyDtH();
    c.release_csr();

    a.release_csr();
    b.release_csr();

    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

/*Main Function*/
int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    CSR<IT, VT> a, b;
    CSR<IT, VT> c1, c2, c3;

    /* Set CSR reading from MM file */
    printf("Initialize Matrix A\n");
    printf("Read matrix data from %s\n", argv[1]);
    a.init_data_from_mtx(argv[1]);

    printf("Initialize Matrix B\n");
    printf("Read matrix data from %s\n", argv[1]);
    b.init_data_from_mtx(argv[1]);
  
    printf("\n=== SpGEMM Comparison ===\n");
    
    /* 1. cuSPARSE */
    printf("\n1. Testing cuSPARSE:\n");
    spgemm_cusparse_test(a, b, c1);
    
    /* 2. Hash without AIA */
    printf("\n2. Testing Hash without AIA:\n");
    spgemm_hash_old(a, b, c2);
    
    /* 3. Hash with AIA */
    printf("\n3. Testing Hash with AIA:\n");
    spgemm_hash_aia(a, b, c3);
    
    a.release_cpu_csr();
    b.release_cpu_csr();
    c1.release_cpu_csr();
    c2.release_cpu_csr();
    c3.release_cpu_csr();
  
    return 0;
}
