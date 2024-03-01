#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addVector(float* c, const float* a, const float* b);