#pragma once

#include "polygon_soup.h"

namespace fcpw {

class LineSegment: public GeometricPrimitive<3> {
public:
	// constructor
	LineSegment(const std::shared_ptr<PolygonSoup<3>>& soup_, bool isFlat_, int index_);

	// returns bounding box
	BoundingBox<3> boundingBox() const;

	// returns centroid
	Vector3 centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume; NOTE: only defined for flat line segment (z = 0)
	float signedVolume() const;

	// returns normal; NOTE: only defined for flat line segment (z = 0)
	Vector3 normal(bool normalize=false) const;

	// returns the normalized normal based on the local parameterization
	Vector3 normal(const Vector2& uv) const;

	// returns barycentric coordinates
	Vector2 barycentricCoordinates(const Vector3& p) const;

	// splits the line segment along the provided coordinate and axis
	void split(int dim, float splitCoord, BoundingBox<3>& boxLeft,
			   BoundingBox<3>& boxRight) const;

	// intersects with ray; NOTE: only implemented for flat line segment (z = 0)
	int intersect(Ray<3>& r, std::vector<Interaction<3>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<3>& s, Interaction<3>& i) const;

	// members
	std::shared_ptr<PolygonSoup<3>> soup;
	bool isFlat;
	int index;

private:
	// returns normalized vertex normal if available;
	// otherwise computes normalized segment normal
	Vector3 normal(int vIndex) const;
};

// reads soup from obj file
std::shared_ptr<PolygonSoup<3>> readLineSegmentSoupFromOBJFile(const std::string& filename,
															   bool& isFlat);

// reads line segment soup from obj file
std::shared_ptr<PolygonSoup<3>> readLineSegmentSoupFromOBJFile(const std::string& filename,
								  std::vector<std::shared_ptr<Primitive<3>>>& lineSegments,
								  bool computeWeightedNormals);

} // namespace fcpw
