#include "kernel.cuh"

__global__ void addVector(float* c, const float* a, const float* b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}