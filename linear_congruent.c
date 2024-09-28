#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define A 1664525
#define C 1013904223

#define SIZE 100000000

unsigned rnd;

unsigned linear_congruent_random() {
   rnd = (A * rnd + C);
   return rnd;
}

int main(void) {
    srand(time(NULL));
    int array[32];

    struct timeval start, end;
    gettimeofday(&start, NULL);

	for(int i = 0; i < SIZE; i++) {
        unsigned random_number = rand();
        if (random_number == 0) {
            array[i % 32] = random_number;
        }
    }
    gettimeofday(&end, NULL);

    long exec_time = (end.tv_usec - start.tv_usec) +
                    (end.tv_sec - start.tv_sec) * 1000000;
    printf("Funzione da libreria\n");
    printf("array[0]: %d, milliseconds: %ld\n", array[0], exec_time / 1000);
    /// ---------------------------
    // Funzione manuale
    rnd = time(NULL);
    gettimeofday(&start, NULL);

	for(int i = 0; i < SIZE; i++) {
        unsigned random_number = linear_congruent_random();
        if (random_number == 0) {
            array[i % 32] = random_number;
        }
    }
    gettimeofday(&end, NULL);

    exec_time = (end.tv_usec - start.tv_usec) +
                (end.tv_sec - start.tv_sec) * 1000000;
    printf("Funzione manuale\n");
    printf("array[0]: %d, milliseconds: %ld\n", array[0], exec_time / 1000);
    return 0;
}
