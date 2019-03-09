/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
    
    //Create 3 streams
    cudaStream_t stream0, stream1, stream2;
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

    float *h_A, *h_B, *h_C;
    float *d_A0, *d_B0, *d_C0; //device memory for stream0
    float *d_A1, *d_B1, *d_C1; //device	memory for stream1
    float *d_A2, *d_B2, *d_C2; //device	memory for stream2
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize;
   
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 1000000;

      } else if (argc == 2) {
      VecSize = atoi(argv[1]);   
      
      
      }
  
      else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    
    //h_A = (float*) malloc( sizeof(float)*A_sz );
    cudaHostAlloc((void**) &h_A,VecSize*sizeof(float),cudaHostAllocDefault);
    for (unsigned int i=0; i < A_sz; i++) { h_A[i] = (rand()%100)/100.00; }

    //h_B = (float*) malloc( sizeof(float)*B_sz );
    cudaHostAlloc((void**) &h_B,VecSize*sizeof(float),cudaHostAllocDefault);
    for (unsigned int i=0; i < B_sz; i++) { h_B[i] = (rand()%100)/100.00; }

    //h_C = (float*) malloc( sizeof(float)*C_sz );
    cudaHostAlloc((void**) &h_C,VecSize*sizeof(float),cudaHostAllocDefault);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("Size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    int segSize = (VecSize - 1)/3 + 1; 

    cudaMalloc((void**) &d_A0, sizeof(float) * segSize);
    cudaMalloc((void**) &d_B0, sizeof(float) * segSize);
    cudaMalloc((void**) &d_C0, sizeof(float) * segSize);
    cudaMalloc((void**) &d_A1, sizeof(float) * segSize);
    cudaMalloc((void**) &d_B1, sizeof(float) * segSize);
    cudaMalloc((void**) &d_C1, sizeof(float) * segSize);
    cudaMalloc((void**) &d_A2, sizeof(float) * segSize);
    cudaMalloc((void**) &d_B2, sizeof(float) * segSize);
    cudaMalloc((void**) &d_C2, sizeof(float) * segSize);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device, launching kernels and copying data back from device to host..."); fflush(stdout);
    startTime(&timer);

    int blkSz = 512;
    int gridSz = (segSize - 1)/blkSz + 1;

    for(int i = 0; i < VecSize; i = i + segSize*3) {

	//Memory copy from host to device for stream0
	cudaMemcpyAsync(d_A0, h_A + i, sizeof(float) * segSize, cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_B0, h_B + i, sizeof(float) * segSize, cudaMemcpyHostToDevice, stream0);
	
	//Memory copy from host	to device for stream1
	cudaMemcpyAsync(d_A1, h_A + i + segSize, sizeof(float) * segSize, cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_B1, h_B + i + segSize, sizeof(float) * segSize, cudaMemcpyHostToDevice, stream1);

	//Memory copy from host	to device for stream2
	cudaMemcpyAsync(d_A2, h_A + i + (2 * segSize), sizeof(float) * segSize, cudaMemcpyHostToDevice, stream2);
	cudaMemcpyAsync(d_B2, h_B + i + (2 * segSize), sizeof(float) * segSize, cudaMemcpyHostToDevice, stream2);

	//Kernel calls for 3 streams
	VecAdd<<<gridSz, blkSz, 0, stream0>>>(segSize, d_A0, d_B0, d_C0);
	VecAdd<<<gridSz, blkSz, 0, stream1>>>(segSize, d_A1, d_B1, d_C1);
	VecAdd<<<gridSz, blkSz, 0, stream2>>>(segSize, d_A2, d_B2, d_C2);

	//Memory copy from device to host for 3 streams
	cudaMemcpyAsync(h_C + i, d_C0, sizeof(float) * segSize, cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(h_C + i + segSize, d_C1, sizeof(float) * segSize, cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(h_C + i + (2 * segSize), d_C2, sizeof(float) * segSize, cudaMemcpyDeviceToHost, stream2);
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    //cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(h_A, h_B, h_C, VecSize);

    // Free memory ------------------------------------------------------------

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    cudaFree(d_A0);
    cudaFree(d_B0);
    cudaFree(d_C0);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);

    return 0;

}
