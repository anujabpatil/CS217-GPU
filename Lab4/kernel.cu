/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
    

__global__ void VecAdd(int n, const float *A, const float *B, float *C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
	C[i] = A[i] + B[i];
    }
}

