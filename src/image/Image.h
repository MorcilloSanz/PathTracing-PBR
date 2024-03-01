#pragma once

#include <iostream>
#include <vector>
#include <cstdint>

namespace pt
{

struct Color {

	uint8_t r, g, b, a;

	Color(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a = 255)
		: r(_r), g(_g), b(_b), a(_a) {
	}

	Color() = default;
	~Color() = default;
};

class Image
{
private:
	int width;
	int height;

	std::vector<unsigned char> data;

public:
	Image() = default;
	Image(int _width, int _height);

	Image(const Image& image);
	Image(Image&& image) noexcept;

	~Image() = default;

	Image& operator=(const Image& image);
	Image& operator=(Image&& image) noexcept;

public:
	void setPixel(unsigned int i, unsigned int j, const Color& color);
	Color getPixel(unsigned int i, unsigned int j);

	void savePNG(const std::string& path);

public:
	int getWidth() const { return width; }
	int getHeight() const { return height; }

	std::vector<unsigned char>& getData() { return data; }
};

}