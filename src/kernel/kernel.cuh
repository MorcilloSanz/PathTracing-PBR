#pragma once

#include "image/Image.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace pt
{

// CUDA Kernels
__global__ void kernelGenerateImage(unsigned char* image, int width, int height, int channels);

// Backends: GPU and CPU comunication abstraction
Image generateCudaImage(int width, int height);

}