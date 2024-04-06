#pragma once

#include <iostream>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Vertex.h"

using VertexArray = std::vector<Vertex>;
using IndexArray = std::vector<unsigned int>;

class Mesh {
private:
	VertexArray vertices;
	IndexArray indices;
	glm::mat4 modelMatrix;
public:
	Mesh(const VertexArray& _vertices)
		: vertices(_vertices), modelMatrix(1.0) {
	}

	Mesh(const VertexArray& _vertices, const IndexArray& _indices)
		: vertices(_vertices), indices(_indices), modelMatrix(1.0) {
	}

	Mesh(const Mesh& mesh)
		: vertices(mesh.vertices), indices(mesh.indices), modelMatrix(mesh.modelMatrix) {
	}

	Mesh(Mesh&& mesh) noexcept
		: vertices(std::move(mesh.vertices)), indices(std::move(mesh.indices)),
		modelMatrix(std::move(mesh.modelMatrix)) {
	}

	Mesh() = default;
	~Mesh() = default;

	Mesh& operator=(const Mesh& mesh) {

		vertices = mesh.vertices;
		indices = mesh.indices;
		modelMatrix = mesh.modelMatrix;

		return *this;
	}

	Mesh& operator=(Mesh&& mesh) noexcept {

		vertices = std::move(mesh.vertices);
		indices = std::move(mesh.indices);
		modelMatrix = std::move(mesh.modelMatrix);

		return *this;
	}
public:
	inline void translate(const glm::vec3& v) { modelMatrix = glm::translate(modelMatrix, v); }
	inline void rotate(float degrees, const glm::vec3& axis) { modelMatrix = glm::rotate(modelMatrix, glm::radians(degrees), axis); }
	inline void scale(const glm::vec3& s) { modelMatrix = glm::scale(modelMatrix, s); }

	inline VertexArray& getVertices() { return vertices; }
	inline IndexArray& getIndices() { return indices; }

	inline void setModelMatrix(const glm::mat4& modelMatrix) { this->modelMatrix = modelMatrix; }
	inline glm::mat4& getModelMatrix() { return modelMatrix; }
};