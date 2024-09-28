#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#define N 624
#define M 397
#define UPPER_MASK 0x80000000UL   /* Most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL   /* Least significant r bits */

#define SIZE 100000000

static unsigned long state[N];
static int stateIdx = N + 1;

static unsigned long constants[2] = {0x0UL, 0x9908b0dfUL};

void init_genrand(unsigned long seed) {
    state[0] = seed & 0xffffffffUL;
    for (int i = 1; i < N; i++) {
        state[i] = (1812433253UL * (state[i-1] ^ (state[i-1] >> 30)) + i);
        state[i] &= 0xffffffffUL;
    }
    stateIdx = N;
}

unsigned int genrand_int32(void) {
    unsigned long y;
    int k;

    if (stateIdx >= N) {
        for (k = 0; k < N - M; k++) {
            y = (state[k] & UPPER_MASK) | (state[k + 1] & LOWER_MASK);
            state[k] = state[k + M] ^ (y >> 1) ^ constants[y & 0x1UL];
        }
        for (; k < N - 1; k++) {
            y = (state[k] & UPPER_MASK) | (state[k + 1] & LOWER_MASK);
            state[k] = state[k + (M - N)] ^ (y >> 1) ^ constants[y & 0x1UL];
        }
        y = (state[N - 1] & UPPER_MASK) | (state[0] & LOWER_MASK);
        state[N - 1] = state[M - 1] ^ (y >> 1) ^ constants[y & 0x1UL];

        stateIdx = 0;
    }

    y = state[stateIdx++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return (unsigned int)y;
}

int main() {
    unsigned long seed = time(NULL); //5489UL;  // Default seed value
    init_genrand(seed);
    int array[32];

    struct timeval start, end;
    gettimeofday(&start, NULL);

	for (int i = 0; i < SIZE; i++) {
        unsigned random_number = genrand_int32();
        if (random_number == 0) {
            array[i % 32] = random_number;
        }
    }
    gettimeofday(&end, NULL);

    long time = (end.tv_usec - start.tv_usec) +
                (end.tv_sec - start.tv_sec) * 1000000;
    printf("array[0]: %d, milliseconds: %ld\n", array[0], time / 1000);
    return 0;
}
