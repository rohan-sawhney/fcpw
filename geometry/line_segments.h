#pragma once

#include "polygon_soup.h"

namespace fcpw {

class LineSegment: public GeometricPrimitive<3> {
public:
	// constructors
	LineSegment();
	LineSegment(const PolygonSoup<3> *soup_, int index_);

	// returns bounding box
	BoundingBox<3> boundingBox() const;

	// returns centroid
	Vector3 centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume; NOTE: specialized to flat line segment (z = 0)
	float signedVolume() const;

	// returns normal; NOTE: specialized to flat line segment (z = 0)
	Vector3 normal(bool normalize=false) const;

	// returns the normalized normal based on the local parameterization
	Vector3 normal(const Vector2& uv) const;

	// returns barycentric coordinates
	Vector2 barycentricCoordinates(const Vector3& p) const;

	// splits the line segment along the provided coordinate and axis
	void split(int dim, float splitCoord, BoundingBox<3>& boxLeft,
			   BoundingBox<3>& boxRight) const;

	// intersects with ray; NOTE: specialized to flat line segment (z = 0)
	int intersect(Ray<3>& r, std::vector<Interaction<3>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<3>& s, Interaction<3>& i) const;

	// members
	const PolygonSoup<3> *soup;
	int index;

private:
	// returns normalized vertex normal if available;
	// otherwise computes normalized segment normal
	Vector3 normal(int vIndex) const;
};

// reads soup from obj file
PolygonSoup<3>* readLineSegmentSoupFromOBJFile(const std::string& filename, bool& isFlat);

// reads line segment soup from obj file
PolygonSoup<3>* readLineSegmentSoupFromOBJFile(const std::string& filename, std::vector<LineSegment *>& lineSegments,
											   bool computeWeightedNormals);

} // namespace fcpw
