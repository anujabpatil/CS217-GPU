/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    __device__ __shared__ float partialSumVector[BLOCK_SIZE * 2];
    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x * blockDim.x * 2;
    if((start + t) < size) {
        partialSumVector[t] = in[start + t];
    } else partialSumVector[t] = 0.0;
    if((start + blockDim.x + t) < size) {
        partialSumVector[blockDim.x + t] = in[start + blockDim.x + t];
    } else partialSumVector[blockDim.x + t] = 0.0;

    for(unsigned int stride = 1; stride <= blockDim.x; stride = stride * 2) {
        __syncthreads();
        if(t % stride == 0) {
                partialSumVector[2*t] += partialSumVector[2*t + stride];
        }
    }

    out[blockIdx.x] = partialSumVector[0];
}
