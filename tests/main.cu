#include <stdio.h>

#include <cuda-linalg.h>

#define MAX_THREADS 1024

int main(void){
    if(!cudaReady()) return 1;

    printf("CUDA devices have been detected!\n");

    double a[6] = {
        1, 2, 3,
        4, 5, 6
    };

    double b[6] = {
        7, 8,
        9, 10,
        11, 12
    };

    double *c = (double*)malloc(6 * sizeof(double));

    dotProduct(a, b, 2, 3, 3, 2, c);

    for(int i = 0; i < 4; i++){
        printf("%f\n", c[i]);
    }

    sumMatrices(a, b, 2, 3, c);

    for(int i = 0; i < 6; i++){
        printf("%f\n", c[i]);
    }

    return 0;
}