#include <iostream>

#include "image/Image.h"

int main() {

	pt::Image image(400, 400);
	image.setPixel(50, 50, pt::Color(255, 0, 0));
	image.savePNG("C:/Users/alber/Desktop/test.png");

	return 0;
}