#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define SIZE 100000000
#define NUM_MULTIPROCESSORS 100
#define THREADS_PER_BLOCK 128
#define NUM_TOTAL_THREADS (NUM_MULTIPROCESSORS * THREADS_PER_BLOCK)

#define ELEMENTS_PER_THREAD (SIZE / NUM_TOTAL_THREADS)

#define VECTOR_SIZE 32

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


__global__ void setup_kernel(unsigned* sobolDirectionVectors,
                             unsigned* sobolScrambleConstants,
                             curandStateSobol32_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(sobolDirectionVectors + VECTOR_SIZE*id,
                sobolScrambleConstants[id],
                &state[id]);
}

__global__ void generate_kernel(curandStateSobol32_t *state,
                                int n,
                                unsigned int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandStateSobol32_t localState = state[id];
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
                CURAND_RNG_QUASI_SOBOL32));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float),
        cudaMemcpyDeviceToHost));

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    free(hostData);
    return EXIT_SUCCESS;
#else
    unsigned int* d_output;

    curandStateSobol32_t* devSOBOLStates;
    CUDA_CALL(cudaMalloc((void **)&devSOBOLStates, NUM_TOTAL_THREADS *
                        sizeof(curandStateSobol32_t)));
    cudaMalloc(&d_output, 32  * sizeof(unsigned int));

    curandDirectionVectors32_t* hostVectors32;
    unsigned* hostScrambleConstants32;
    unsigned* devDirectionVectors32;
    unsigned* devScrambleConstants32;

    CURAND_CALL(curandGetDirectionVectors32(&hostVectors32,
                                            CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    CURAND_CALL(curandGetScrambleConstants32(&hostScrambleConstants32));

    CUDA_CALL(cudaMalloc((void **)&(devDirectionVectors32),
                       3 *  NUM_TOTAL_THREADS * VECTOR_SIZE * sizeof(unsigned)));

    CUDA_CALL(cudaMemcpy(devDirectionVectors32, hostVectors32,
                         3 * NUM_TOTAL_THREADS * VECTOR_SIZE * sizeof(unsigned),
                         cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void **)&devScrambleConstants32,
                         3 *  NUM_TOTAL_THREADS * sizeof(unsigned)));

    CUDA_CALL(cudaMemcpy(devScrambleConstants32, hostScrambleConstants32,
                         3 * NUM_TOTAL_THREADS * sizeof(unsigned),
                         cudaMemcpyHostToDevice));

    setup_kernel<<<NUM_MULTIPROCESSORS, THREADS_PER_BLOCK>>>(devDirectionVectors32,
                                                             devScrambleConstants32, devSOBOLStates);
    cudaDeviceSynchronize();

    generate_kernel<<<NUM_MULTIPROCESSORS, THREADS_PER_BLOCK>>>(devSOBOLStates, 0, d_output);
    cudaDeviceSynchronize();

    cudaFree(devDirectionVectors32);
    cudaFree(devScrambleConstants32);
    cudaFree(devSOBOLStates);
    cudaFree(d_output);
    return 0;
#endif
}
