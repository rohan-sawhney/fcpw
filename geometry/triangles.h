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
	int intersect(Ray<3>& r, std::vector<Interaction<3>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<3>& s, Interaction<3>& i) const;

	// splits primitive and returns the tight bounding boxes on either side of the split
	void split(const BoundingBox<3>& curBox, BoundingBox<3>& leftBox, BoundingBox<3>& rightBox, int splitDim, float splitLoc) const;

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
