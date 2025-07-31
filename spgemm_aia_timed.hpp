// spgemm_aia_timed.hpp - ONLY modified parts for AIA timing
// This file contains ONLY the functions that need modification from HashSpGEMM_volta.hpp
// All other functions from HashSpGEMM_volta.hpp remain UNMODIFIED

#ifndef SPGEMM_AIA_TIMED_HPP
#define SPGEMM_AIA_TIMED_HPP

#include <HashSpGEMM_volta.hpp>  // Include original implementation

// Forward declaration of PerformanceResult structure
struct PerformanceResult {
    float total_time_ms;
    float aia1_time_ms;
    float aia2_time_ms;
    float symbolic_time_ms;
    float numeric_time_ms;
    float gflops;
    long long int flop_count;
    std::string implementation;
    std::string dataset;
    float sparsity;
    int feature_dim;
};

// MODIFIED: Add timing to the main SpGEMM function
template <bool sort, class idType, class valType>
void SpGEMM_Hash_AIA_Timed(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c,
                           float &total_aia1_time, float &total_aia2_time, 
                           float &total_symbolic_time, float &total_numeric_time)
{
    BIN<idType, BIN_NUM> bin(a.nrow);
    
    cudaEvent_t event[8];
    float msec;
    
    for (int i = 0; i < 8; i++) {
        cudaEventCreate(&(event[i]));
    }

    // AIA data structures
    idType *aia_d_row_1, *aia_d_nnz_1;
    cudaMalloc((void **)&aia_d_row_1, sizeof(idType) * (2 * a.nrow));
    cudaMalloc((void **)&aia_d_nnz_1, sizeof(idType) * (2 * a.nnz));

    bin.set_load_banlance(a, b);

    // TIMING: First AIA phase
    cudaEventRecord(event[0], 0);
    read_1_new(aia_d_row_1, aia_d_nnz_1, a.d_rpt, bin.d_permutation, b.d_rpt, a.d_colids, a.nrow);
    cudaEventRecord(event[1], 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[0], event[1]);
    total_aia1_time += msec;
    cout << "AIA1 runtime: " << msec << " ms" << endl;

    // TIMING: Symbolic phase
    cudaEventRecord(event[2], 0);
    hash_symbolic(a.d_rpt, b.d_colids, c, aia_d_row_1, aia_d_nnz_1, bin, a.nrow);
    cudaEventRecord(event[3], 0);   
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[2], event[3]);
    total_symbolic_time += msec;
    cout << "Symbolic phase: " << msec << " ms" << endl;
    
    // Allocate output arrays
    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
    
    bin.set_min_bin(a.nrow, TS_N_P, TS_N_T);
    
    // TIMING: Second AIA phase
    cudaEventRecord(event[4], 0);
    read_1_new(aia_d_row_1, aia_d_nnz_1, a.d_rpt, bin.d_permutation, b.d_rpt, a.d_colids, a.nrow);
    cudaEventRecord(event[5], 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[4], event[5]);
    total_aia2_time += msec;
    cout << "AIA2 runtime: " << msec << " ms" << endl;

    // TIMING: Numeric phase
    cudaEventRecord(event[6], 0);
    hash_numeric<idType, valType, sort>(a.d_values, b.d_colids, b.d_values, c, aia_d_row_1, aia_d_nnz_1, bin);
    cudaEventRecord(event[7], 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[6], event[7]);
    total_numeric_time += msec;
    cout << "Numeric phase: " << msec << " ms" << endl;

    // Cleanup
    cudaFree(aia_d_row_1);
    cudaFree(aia_d_nnz_1);
    
    for (int i = 0; i < 8; i++) {
        cudaEventDestroy(event[i]);
    }
}

// Wrapper function for the benchmark
template <class idType, class valType>
PerformanceResult spgemm_hash_aia_timed(CSR<idType, valType> adj, CSR<idType, valType> features, 
                                        CSR<idType, valType> &output, const std::string& dataset, float sparsity) {
    PerformanceResult result;
    result.implementation = "Hash_with_AIA";
    result.dataset = dataset;
    result.sparsity = sparsity;
    result.feature_dim = features.ncolumn;  // Use ncolumn
    
    idType i;
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec;
    float total_aia1 = 0, total_aia2 = 0, total_symbolic = 0, total_numeric = 0;
    
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
    
    // Copy matrices to device
    adj.memcpyHtD();
    features.memcpyHtD();
    
    // Count FLOPs
    get_spgemm_flop(adj, features, flop_count);
    result.flop_count = flop_count;
    
    // Execute SpGEMM with detailed timing
    ave_msec = 0;
    
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            output.release_csr();
        }
        
        float aia1_time = 0, aia2_time = 0, symbolic_time = 0, numeric_time = 0;
        
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash_AIA_Timed<true, idType, valType>(adj, features, output, 
                                                     aia1_time, aia2_time, symbolic_time, numeric_time);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
        
        if (i > 0) {
            ave_msec += msec;
            total_aia1 += aia1_time;
            total_aia2 += aia2_time;
            total_symbolic += symbolic_time;
            total_numeric += numeric_time;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;
    
    result.total_time_ms = ave_msec;
    result.aia1_time_ms = total_aia1 / (SpGEMM_TRI_NUM - 1);
    result.aia2_time_ms = total_aia2 / (SpGEMM_TRI_NUM - 1);
    result.symbolic_time_ms = total_symbolic / (SpGEMM_TRI_NUM - 1);
    result.numeric_time_ms = total_numeric / (SpGEMM_TRI_NUM - 1);
    result.gflops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    
    output.memcpyDtH();
    
    adj.release_csr();
    features.release_csr();
    
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
    
    return result;
}

#endif // SPGEMM_AIA_TIMED_HPP
