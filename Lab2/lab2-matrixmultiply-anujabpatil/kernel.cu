/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float mShared[TILE_SIZE][TILE_SIZE];
    __shared__ float nShared[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float pVal = 0.0;

    for(int phase = 0; phase < (k-1)/TILE_SIZE + 1; ++phase) {
    	if(row < m && (phase * TILE_SIZE + tx) < k) {
	    mShared[ty][tx] = A[row * k + phase * TILE_SIZE + tx];	
	} else {
	    mShared[ty][tx] = 0.0;
	}
	if((phase * TILE_SIZE + ty) < k && col < n) {
	    nShared[ty][tx] = B[(phase * TILE_SIZE + ty) * n + col];
	} else {
	    nShared[ty][tx] = 0.0;
	} 
	__syncthreads();

	if(row < m && col < n) {
	    for(int i = 0; i < TILE_SIZE; ++i) {
                pVal += mShared[ty][i] * nShared[i][tx];
            }	
	}
	__syncthreads();
    }
    if(row < m && col < n) {
	C[row * n + col] = pVal;
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 dimGrid((n-1)/TILE_SIZE + 1, (m-1)/TILE_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1); 

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
}


