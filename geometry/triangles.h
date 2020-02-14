#pragma once

#include "polygon_soup.h"

namespace fcpw {

class Triangle: public Primitive<3> {
public:
	// constructor
	Triangle(const Transform<float, 3, Affine>& transform_,
			 const std::shared_ptr<PolygonSoup<3>>& soup_, int index_);

	// returns bounding box
	BoundingBox<3> boundingBox() const;

	// returns centroid
	Vector3f centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// returns normal
	Vector3f normal(bool normalize=false) const;

	// returns barycentric coordinates
	Vector2f barycentricCoordinates(const Vector3f& p) const;

	// returns texture coordinates
	Vector2f textureCoordinates(const Vector2f& uv) const;

	// intersects with ray
	int intersect(const Ray<3>& r, std::vector<Interaction<3>>& is,
				  bool collectAll=false) const;

	// finds closest point to sphere center
	void findClosestPoint(const Vector3f& x, Interaction<3>& i) const;

	// members
	std::shared_ptr<PolygonSoup<3>> soup;
	const std::vector<int>& indices; /* a.k.a. vIndices */
	const std::vector<int>& eIndices;
	const std::vector<int>& tIndices;
	int index;

private:
	// returns normalized vertex or edge normal if available;
	// otherwise computes normalized triangle normal
	Vector3f normal(int vIndex, int eIndex) const;
};

// reads soup from obj file
std::shared_ptr<PolygonSoup<3>> readFromOBJFile(const std::string& filename,
												const Transform<float, 3, Affine>& transform);

// reads triangle soup from obj file
std::shared_ptr<PolygonSoup<3>> readFromOBJFile(const std::string& filename,
												const Transform<float, 3, Affine>& transform,
												std::vector<std::shared_ptr<Primitive<3>>>& triangles,
												bool computeWeightedNormals);

} // namespace fcpw
