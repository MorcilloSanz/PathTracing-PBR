#include "kernel.cuh"

#include <iostream>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>

namespace pt
{

// Kernels
__global__ void kernelGenerateImage(unsigned char* image, int width, int height, int channels) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        int index = (y * width + x) * channels;

        image[index] = static_cast<int>(255.0f * x / width);
        image[index + 1] = static_cast<int>(255.0f * y / height);
        image[index + 2] = 255 - static_cast<int>(255.0f * x / width);
        image[index + 3] = 255;
    }
}

// Backends
Image generateCudaImage(int width, int height) {

    constexpr int channels = 4;

    // Generate GPU image
    unsigned char* gpuImage;
    cudaMalloc(&gpuImage, width * height * channels * sizeof(unsigned char));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    kernelGenerateImage <<<gridSize, blockSize>>> (gpuImage, width, height, channels);

    // Copy GPU image to CPU image
    Image image(width, height);
    size_t count = width * height * channels * sizeof(unsigned char);
    cudaMemcpy(&image.getData()[0], gpuImage, count, cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(gpuImage);

    // Return image
    return image;
}

}