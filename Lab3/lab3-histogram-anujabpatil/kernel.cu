/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to
#define BLOCK_SIZE 512

// INSERT KERNEL(S) HERE

__global__ void fillBins(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {
    
    extern __shared__ unsigned int private_histogram[];

    //Initializing private histogram bins
    int bin_stride = 0;
    while((threadIdx.x + bin_stride) < num_bins) {
	private_histogram[threadIdx.x + bin_stride] = 0;
	bin_stride += blockDim.x;
    }
    __syncthreads();

    //Computation of private histogram
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while(i < num_elements) {
     	atomicAdd(&private_histogram[input[i]], 1);
	i += stride;
    }
    __syncthreads();

    //Merging private history bins with global history bins
    bin_stride = 0;
    while((threadIdx.x + bin_stride) < num_bins) {
        atomicAdd(&bins[threadIdx.x + bin_stride], private_histogram[threadIdx.x + bin_stride]);
        bin_stride += blockDim.x;
    }
    __syncthreads();
}





/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
    dim3 dimGrid((num_elements - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    fillBins<<<dimGrid, dimBlock, num_bins * sizeof(unsigned int)>>>(input, bins, num_elements, num_bins);
}
