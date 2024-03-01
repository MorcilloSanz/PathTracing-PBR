#include <iostream>

#include "kernel/kernel.cuh"

using namespace pt;

int main() {

	Image image = generateCudaImage(400, 400);
	image.savePNG("C:/Users/alber/Desktop/test.png");

	return 0;
}