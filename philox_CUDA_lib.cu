#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define SIZE 100000000
#define NUM_MULTIPROCESSORS 80
#define THREADS_PER_BLOCK 848
#define NUM_TOTAL_THREADS (NUM_MULTIPROCESSORS * THREADS_PER_BLOCK)

#define ELEMENTS_PER_THREAD (SIZE / NUM_TOTAL_THREADS)


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandStatePhilox4_32_10_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        x = curand(&localState);
        if(x == 0) {
            result[i % 32] = x;
        }
    }
}

#define CURAND_HOST_API 1

int main(int argc, char *argv[]) {
#if CURAND_HOST_API
    size_t n = SIZE;
    curandGenerator_t gen;
    float *devData, *hostData;

    /* Allocate n floats on host */
    hostData = (float *)calloc(n, sizeof(float));

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_PHILOX4_32_10));

    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
                1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
        cudaMemcpyDeviceToHost));

    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    free(hostData);
    return EXIT_SUCCESS;
#else
    unsigned int *d_output;

    curandStatePhilox4_32_10_t *devPHILOXStates;
    CUDA_CALL(cudaMalloc((void **)&devPHILOXStates, NUM_TOTAL_THREADS *
                        sizeof(curandStatePhilox4_32_10_t)));
    cudaMalloc(&d_output, 32  * sizeof(unsigned int));

    setup_kernel<<<NUM_MULTIPROCESSORS, THREADS_PER_BLOCK>>>(devPHILOXStates);
    cudaDeviceSynchronize();

    generate_kernel<<<NUM_MULTIPROCESSORS, THREADS_PER_BLOCK>>>(devPHILOXStates, 0, d_output);

    cudaDeviceSynchronize();
    cudaFree(devPHILOXStates);
    cudaFree(d_output);
    return 0;
#endif
}
