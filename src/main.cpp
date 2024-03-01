#include <iostream>

#include "kernel/kernel.cuh"

using namespace pt;

int main() {

	Image image = generateCudaImage(500, 500);
	image.savePNG("C:/Users/alber/Desktop/img.png");

	return 0;
}