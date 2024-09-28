#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define A 1664525
#define C 1013904223

#define SIZE 100000000
#define NUM_MULTIPROCESSORS 80
#define THREADS_PER_BLOCK 1024
#define NUM_TOTAL_THREADS (NUM_MULTIPROCESSORS * THREADS_PER_BLOCK)

#define ELEMENTS_PER_THREAD (SIZE / NUM_TOTAL_THREADS)

__global__ void linear_congruent_kernel(unsigned seed, unsigned* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned rnd = seed + idx;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        rnd = (A * rnd + C);
        if (rnd == 0)
            output[rnd % 32] = rnd;
    }
}

int main() {
    unsigned* d_random_numbers;
    cudaMalloc((void**)&d_random_numbers, 32 * sizeof(unsigned));

    unsigned seed = time(NULL);
    linear_congruent_kernel<<<NUM_MULTIPROCESSORS, THREADS_PER_BLOCK>>>(seed, d_random_numbers);

    cudaDeviceSynchronize();
    cudaFree(d_random_numbers);
    return 0;
}
