#include<stdio.h>
#include<time.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#define SIZE 100000000

unsigned squares32(unsigned long long ctr, unsigned long long key) {
    unsigned long long x, y, z;
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

int main() {
    unsigned long long counter = 1;
    unsigned long long key = time(NULL);
    int array[32];

    struct timeval start, end;
    gettimeofday(&start, NULL);

	for(int i = 0; i < SIZE; i++) {
        unsigned random_number = squares32(i, key);
        if (random_number == 0) {
            array[i % 32] = random_number;
        }
    }
    gettimeofday(&end, NULL);

    long exec_time = (end.tv_usec - start.tv_usec) +
                    (end.tv_sec - start.tv_sec) * 1000000;
    printf("array[0]: %d, milliseconds: %ld\n", array[0], exec_time / 1000);

    return 0;
}
