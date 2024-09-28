#include <stdio.h>
#include <time.h>

#define N 624
#define M 397
#define N2 (N + 224)  // N + 224 defines the size of the shared memory
#define UPPER_MASK 0x80000000UL  /* Most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL  /* Least significant r bits */
#define BLOCK_SIZE 256  /* Number of threads per block */

#define SIZE 100000000
#define NUM_MULTIPROCESSORS 80
#define THREADS_PER_BLOCK 848
#define NUM_TOTAL_THREADS (NUM_MULTIPROCESSORS * THREADS_PER_BLOCK)

#define ELEMENTS_PER_THREAD (SIZE / NUM_TOTAL_THREADS)


// CUDA Constants used for tempering
__device__ unsigned int constants[2] = {0x0UL, 0x9908b0dfUL};

// Kernel function signature
__global__ void mersenne_twister_kernel(unsigned *global_state, unsigned *output) {
    __shared__ unsigned int state[N2];  // Declare shared memory for state
    int num_loops = ELEMENTS_PER_THREAD;
    // Determine block start and thread indices
    int idx = threadIdx.x;
    int output_start = blockIdx.x * BLOCK_SIZE;

    // Copy values into state from global memory
    state[idx] = global_state[idx] + idx;
    __syncthreads();  // Synchronize all threads in the block

    // Define thread-specific variables
    int k1 = idx;
    int k2 = k1 + 1;
    int k3 = k1 + M;
    int k4 = k1 + N;
    int k5 = output_start + k1;
    unsigned int y;

    // Main loop: number of 224 updates of state
    for (; num_loops > 0; num_loops--) {
        // GENERATING phase
        y = state[k1];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680UL;
        y ^= (y << 15) & 0xefc60000UL;
        y ^= (y >> 18);

        // Store the result in global memory
        if (y == 0)
            output[y % 32] = k5;
           // output[k5] = y;

        // UPDATING STATE phase
        y = (state[k1] & UPPER_MASK) | (state[k2] & LOWER_MASK);
        state[k4 % N2] = state[k3 % N2] ^ (y >> 1) ^ constants[y & 0x1UL];

        // Update indices for the next loop
        k1 += 224; k2 += 224; k3 += 224; k4 += 224; k5 += 224;
        if (k1 >= N2) k1 -= N2;
        if (k2 >= N2) k2 -= N2;
        if (k3 >= N2) k3 -= N2;
        if (k4 >= N2) k4 -= N2;

        // Synchronize threads before the next iteration
        __syncthreads();
    }
}

int main() {
    unsigned int *d_global_state, *d_output;
    cudaMalloc(&d_global_state, N2 * sizeof(unsigned int));
    cudaMalloc(&d_output, 32  * sizeof(unsigned int));

    unsigned int h_global_state[N2];
    for (int i = 0; i < N2; i++) {
        h_global_state[i] = time(NULL);
    }
    cudaMemcpy(d_global_state, h_global_state, N2 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    mersenne_twister_kernel<<<NUM_MULTIPROCESSORS, THREADS_PER_BLOCK>>>(d_global_state, d_output);

    cudaDeviceSynchronize();
    cudaFree(d_global_state);
    cudaFree(d_output);
    return 0;
}
