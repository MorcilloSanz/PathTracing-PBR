#pragma once

#include <iostream>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

struct Vertex {

	glm::vec3 pos;
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec2 uv;
	glm::vec3 tan, bitan;

	Vertex(const glm::vec3& _pos, const glm::vec3& _color, const glm::vec3& _normal,
		const glm::vec2& _uv, const glm::vec3& _tan, const glm::vec3& _bitan)
		: pos(_pos), color(_color), normal(_normal), uv(_uv), tan(_tan), bitan(_bitan) {
	}

	Vertex(const glm::vec3& pos, const glm::vec3& color, const glm::vec3& normal,
		const glm::vec2& uv)
		: Vertex(pos, color, normal, uv, glm::vec3(0.0), glm::vec3(0.0)) {
	}

	Vertex(const glm::vec3& pos, const glm::vec3& color, const glm::vec3& normal)
		: Vertex(pos, color, normal, glm::vec2(0.0), glm::vec3(0.0), glm::vec3(0.0)) {
	}

	Vertex(const glm::vec3& pos, const glm::vec3& color)
		: Vertex(pos, color, glm::vec3(0.0), glm::vec2(0.0), glm::vec3(0.0), 
		glm::vec3(0.0)) {
	}

	Vertex(const glm::vec3& pos)
		: Vertex(pos, glm::vec3(1.0), glm::vec3(0.0), glm::vec2(0.0), 
		glm::vec3(0.0), glm::vec3(0.0)) {
	}

	Vertex(const Vertex& vertex)
		: pos(vertex.pos), color(vertex.color), normal(vertex.normal), uv(vertex.uv),
		tan(vertex.tan), bitan(vertex.bitan) {
	}

	Vertex(Vertex&& vertex) noexcept
		: pos(std::move(vertex.pos)), color(std::move(vertex.color)), normal(std::move(vertex.normal)),
		uv(std::move(vertex.uv)), tan(std::move(vertex.tan)), bitan(std::move(vertex.bitan)) {
	}

	Vertex() = default;
	~Vertex() = default;

	Vertex& operator=(const Vertex& vertex) {

		pos = vertex.pos;
		color = vertex.color;
		normal = vertex.normal;
		uv = vertex.uv;
		tan = vertex.tan;
		bitan = vertex.bitan;

		return *this;
	}

	Vertex& operator=(Vertex&& vertex) noexcept {

		pos = std::move(vertex.pos);
		color = std::move(vertex.color);
		normal = std::move(vertex.normal);
		uv = std::move(vertex.uv);
		tan = std::move(vertex.tan);
		bitan = std::move(vertex.bitan);

		return *this;
	}
};