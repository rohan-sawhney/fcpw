#pragma once

#include "polygon_soup.h"

namespace fcpw {

class Triangle: public GeometricPrimitive<3> {
public:
	// constructor
	Triangle(const std::shared_ptr<PolygonSoup<3>>& soup_, int index_);

	// returns bounding box
	BoundingBox<3> boundingBox() const;

	// returns centroid
	Vector3 centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// returns normal
	Vector3 normal(bool normalize=false) const;

	// returns the normalized normal based on the local parameterization
	Vector3 normal(const Vector2& uv) const;

	// returns barycentric coordinates
	Vector2 barycentricCoordinates(const Vector3& p) const;

	// returns texture coordinates
	Vector2 textureCoordinates(const Vector2& uv) const;

	// splits the triangle along the provided coordinate and axis
	void split(int dim, float splitCoord, BoundingBox<3>& boxLeft,
			   BoundingBox<3>& boxRight) const;

	// intersects with ray
	int intersect(Ray<3>& r, std::vector<Interaction<3>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<3>& s, Interaction<3>& i) const;

	// members
	std::shared_ptr<PolygonSoup<3>> soup;
	int index;

private:
	// returns normalized vertex or edge normal if available;
	// otherwise computes normalized triangle normal
	Vector3 normal(int vIndex, int eIndex) const;
};

// reads soup from obj file
std::shared_ptr<PolygonSoup<3>> readTriangleSoupFromOBJFile(const std::string& filename);

// reads triangle soup from obj file
std::shared_ptr<PolygonSoup<3>> readTriangleSoupFromOBJFile(const std::string& filename,
								   std::vector<std::shared_ptr<Primitive<3>>>& triangles,
								   bool computeWeightedNormals);

} // namespace fcpw
