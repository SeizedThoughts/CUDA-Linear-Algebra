extern int cudaReady();

extern __global__ void cudaDotProduct(double *a, double *b, int aY, int aX, int bY, int bX, double *c);

extern void dotProduct(double *a, double *b, int aY, int aX, int bY, int bX, double *c);

extern __global__ void cudaSumMatrices(double *a, double *b, int y, int x, double *c);

extern void sumMatrices(double *a, double *b, int y, int x, double *c);