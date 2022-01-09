#include <stdio.h>

int cudaReady(){
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    if(deviceCount == 0) return 0;
    
    return 1;
}

//dot product
__global__ void cudaDotProduct(double *a, double *b, int aY, int aX, int bY, int bX, double *c){
    //perform vector dot product in one thread
    /*
        aX = bY
        __                     __     __                     __     __                                                                                                  __
        | a[1][1] ... a[1][aX]  |     | b[1][1] ... b[1][bX]  |     | a[1][1] * b[1][1] + ... + a[1][aX] * b[1][bX] ... a[1][aX] * b[1][1] + ... + a[aY][aX] * b[1][bX]  |
        |     .  .        .     |     |     .  .        .     |     |                      .                       .                          .                          |
        |     .    .      .     |  *  |     .    .      .     |  =  |                      .                         .                        .                          |
        |     .      .    .     |     |     .      .    .     |     |                      .                           .                      .                          |
        |_a[aY][1]... a[aY][aX]_|     |_b[bY][1]... b[bY][bX]_|     |_a[1][1] * b[bY][1]+ ... + a[1][aX] * b[bY][bX]... a[1][aX] * b[bY][1]+ ... + a[aY][aX] * b[bY][bX]_|
    */

    int sA = aX * threadIdx.y;
    int sB = threadIdx.x;

    c[bX * threadIdx.y + threadIdx.x] = 0;

    int iA = sA;
    int iB = sB;

    while(iB < bY * bX){
        c[bX * threadIdx.y + threadIdx.x] += a[iA] * b[iB];
        iA++;
        iB += bX;
    }
}

void dotProduct(double *a, double *b, int aY, int aX, int bY, int bX, double *c){
    double *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, aY * aX * sizeof(double));
    cudaMalloc(&d_b, bY * bX * sizeof(double));
    cudaMalloc(&d_c, aY * bX * sizeof(double));

    cudaMemcpy(d_a, a, aY * aX * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bY * bX * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 threads(bX, aY);

    cudaDotProduct<<<1, threads>>>(d_a, d_b, aY, aX, bY, bX, d_c);

    cudaMemcpy(c, d_c, aY * bX * sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void cudaSumMatrices(double *a, double *b, int y, int x, double *c){
    int i = x * threadIdx.y + threadIdx.x;

    c[i] = a[i] + b[i];
}

void sumMatrices(double *a, double *b, int y, int x, double *c){
    double *d_a, *d_b, *d_c;
    
    cudaMalloc(&d_a, y * x * sizeof(double));
    cudaMalloc(&d_b, y * x * sizeof(double));
    cudaMalloc(&d_c, y * x * sizeof(double));

    cudaMemcpy(d_a, a, y * x * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, y * x * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 threads(x, y);

    cudaSumMatrices<<<1, threads>>>(d_a, d_b, y, x, d_c);

    cudaMemcpy(c, d_c, y * x * sizeof(double), cudaMemcpyDeviceToHost);
}