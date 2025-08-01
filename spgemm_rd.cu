#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include "cnpy.h"
#include <iostream>

// ... (keep the CHECK_CUDA and CHECK_CUSPARSE macros as they are) ...
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_npz_file>" << std::endl;
        return EXIT_FAILURE;
    }

    // Load NPZ file
    std::string npz_path = argv[1];
    cnpy::npz_t npz = cnpy::npz_load(npz_path);
    

    // Extract CSR matrix data
    cnpy::NpyArray npzArray_data = npz["data"];
    cnpy::NpyArray npzArray_indices = npz["indices"];
    cnpy::NpyArray npzArray_indptr = npz["indptr"];

    const int A_num_rows = npzArray_indptr.shape[0] - 1;
    const int A_nnz = npzArray_indices.shape[0];
    const int A_num_cols = A_num_rows;

    float* hA_values = npzArray_data.data<float>();
    int* hA_columns = npzArray_indices.data<int>();
    int* hA_csrOffsets = npzArray_indptr.data<int>();

    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    float *dA_values;
        // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA( cudaEventCreate(&start) )
    CHECK_CUDA( cudaEventCreate(&stop) )

    // Start timing
    CHECK_CUDA( cudaEventRecord(start) )

    // Allocate device memory for A
    CHECK_CUDA( cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**)&dA_values, A_nnz * sizeof(float)) )

    // Copy A from host to device
    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice) )

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    int *dC_csrOffsets;
CHECK_CUDA( cudaMalloc((void**)&dC_csrOffsets, (A_num_rows + 1) * sizeof(int)) )

    // Create sparse matrix C for the result
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, A_num_cols, 0,
                                      dC_csrOffsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // SpGEMM Computation (A * A)
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // Set up SpGEMM parameters
    float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    // ask bufferSize1 bytes for external memory
    // SpGEMM work estimation
    CHECK_CUSPARSE( cusparseSpGEMM_workEstimation(handle, opA, opA,
                                                  &alpha, matA, matA, &beta, matC,
                                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**)&dBuffer1, bufferSize1) )

    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE( cusparseSpGEMM_workEstimation(handle, opA, opA,
                                                  &alpha, matA, matA, &beta, matC,
                                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                  spgemmDesc, &bufferSize1, dBuffer1) )




    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opA,
                                           &alpha, matA, matA, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc(&dBuffer2, bufferSize2) )
    
    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opA,
                                           &alpha, matA, matA, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )

    // Get matrix C non-zero entries
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
     CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1) )                                        
    printf("Number of non-zero elements in matrix C: %ld\n", C_nnz1);

    // Allocate matrix C
    int  *dC_columns;
    float *dC_values;

    CHECK_CUDA( cudaMalloc((void**)&dC_columns, C_nnz1 * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**)&dC_values, C_nnz1 * sizeof(float)) )

    // Update matC with the new pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

    // Copy the final products to the matrix C
    CHECK_CUSPARSE( cusparseSpGEMM_copy(handle, opA, opA,
                                        &alpha, matA, matA, &beta, matC,
                                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    // Stop timing

    // int*   hC_csrOffsets_tmp = new int[A_num_rows + 1];
    // int*  hC_columns_tmp = new int [C_nnz1];
    // float* hC_values_tmp= new float[C_nnz1];
    // CHECK_CUDA( cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
    //                        (A_num_rows + 1) * sizeof(int),
    //                        cudaMemcpyDeviceToHost) )
    // CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz1 * sizeof(int),
    //                        cudaMemcpyDeviceToHost) )
    // CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, C_nnz1 * sizeof(float),
    //                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaEventRecord(stop) )
    CHECK_CUDA( cudaEventSynchronize(stop) )
    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, start, stop) )

    printf("SpGEMM computation completed successfully.\n");
    printf("Matrix multiplication time: %f milliseconds\n", milliseconds);

    // for (int i = 0; i < 100; i++) {
    //         std::cout << hC_columns_tmp[i] << " ";
    // }
    // for (int i = 0; i < 100; i++) {
    //         std::cout << hC_values_tmp[i] << " ";
    // }


    // Clean up
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    // Free device memory
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dC_csrOffsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )

    // Destroy CUDA events
    CHECK_CUDA( cudaEventDestroy(start) )
    CHECK_CUDA( cudaEventDestroy(stop) )
    // delete[] hC_csrOffsets_tmp;
    // delete[] hC_columns_tmp;
    // delete[] hC_values_tmp;
    return EXIT_SUCCESS;
}