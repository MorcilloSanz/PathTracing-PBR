#include "Image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

extern "C" {
#include "vendor/stb_image.h"
#include "vendor/stb_image_write.h"
}

namespace pt
{

Image::Image(int _width, int _height) 
	: width(_width), height(_height) {
	data.resize(width * height * 4);
}

Image::Image(int _width, int _height, const std::vector<unsigned char>& _data) 
	: width(_width), height(_height), data(_data) {
}

Image::Image(const Image& image) 
	: width(image.width), height(image.height), data(image.data) {
}

Image::Image(Image&& image) noexcept 
	: width(image.width), height(image.height),
	data(std::move(image.data)) {
}

Image& Image::operator=(const Image& image) {

	width = image.width;
	height = image.height;
	data = image.data;

	return *this;
}

Image& Image::operator=(Image&& image) noexcept {

	width = image.width;
	height = image.height;
	data = std::move(image.data);

	return *this;
}

void Image::setPixel(unsigned int i, unsigned int j, const Color& color) {

	size_t index = i + j * width;
	if (index >= 0 && index < width * height) {

		data[4 * index] = color.r;
		data[4 * index + 1] = color.g;
		data[4 * index + 2] = color.b;
		data[4 * index + 3] = color.a;
	}
}

Color Image::getPixel(unsigned int i, unsigned int j) {

	size_t index = i + j * width;
	if (index >= 0 && index < width * height) {

		return Color(
			data[4 * index],
			data[4 * index + 1],
			data[4 * index + 2],
			data[4 * index + 3]
		);
	}

	return Color(0, 0, 0, 0);
}

void Image::savePNG(const std::string& path) {
	stbi_write_png(path.c_str(), width, height, STBI_rgb_alpha, &data[0], width * STBI_rgb_alpha);
}

}