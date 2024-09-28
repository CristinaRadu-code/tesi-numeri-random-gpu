#include <stdio.h>
#include <stdint.h>

#define SIZE 100000000
#define NUM_MULTIPROCESSORS 80
#define THREADS_PER_BLOCK 1024
#define NUM_TOTAL_THREADS (NUM_MULTIPROCESSORS * THREADS_PER_BLOCK)

#define ELEMENTS_PER_THREAD (SIZE / NUM_TOTAL_THREADS)

__device__ __forceinline__ unsigned squares32(long long unsigned ctr, long long unsigned key) {
    long long unsigned x, y, z;
    y = x = ctr * key;           // Initialize y and x with ctr * key
    z = y + key;                 // z is y + key
    x = x * x + y;               // Square x and add y
    x = (x >> 32) | (x << 32);   // Rotate x
    x = x * x + z;               // Square x and add z
    x = (x >> 32) | (x << 32);   // Rotate x again
    x = x * x + y;               // Square x and add y
    x = (x >> 32) | (x << 32);   // Rotate x once more
    return (x * x + z) >> 32;    // Final square, add z, and return upper 32 bits
}

// CUDA kernel to generate random values using squares32
__global__ void generate_random(unsigned *output, long long unsigned key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        unsigned rnd = squares32(i, key + idx);
        if (rnd == 0)
            output[rnd % 32] = rnd;
    }
}

int main() {
    long long unsigned key = 123456789;    // Example key

    unsigned *d_random_values;
    unsigned *h_random_values = (unsigned *)malloc(32 * sizeof(unsigned));

    // Allocate memory on the GPU
    cudaMalloc(&d_random_values, 32 * sizeof(unsigned));

    // Launch the kernel with N threads (1 block, N threads in this case)
    generate_random<<<NUM_MULTIPROCESSORS, THREADS_PER_BLOCK>>>(d_random_values, key);

    // Copy the results back to the CPU
    cudaMemcpy(h_random_values, d_random_values, 32 * sizeof(unsigned), cudaMemcpyDeviceToHost);

    free(h_random_values);
    cudaFree(d_random_values);

    return 0;
}
