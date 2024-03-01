#pragma once

#include "image/Image.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace pt
{

// Kernels
__global__ void kernelGenerateImage(unsigned char* image, int width, int height, int channels);

// Backends
Image generateCudaImage(int width, int height);

}