#pragma once

#include <fcpw/geometry/line_segments.h>

namespace fcpw {

class SilhouetteEdge: public SilhouettePrimitive<3> {
public:
	// constructor
	SilhouetteEdge();

	// returns bounding box
	BoundingBox<3> boundingBox() const;

	// returns centroid
	Vector3 centroid() const;

	// returns surface area
	float surfaceArea() const;

	// checks whether silhouette has adjacent face
	bool hasFace(int fIndex) const;

	// returns normal of adjacent face
	Vector3 normal(int fIndex, bool normalize=true) const;

	// returns normalized silhouette normal
	Vector3 normal() const;

	// finds closest silhouette point to sphere center
	bool findClosestSilhouettePoint(BoundingSphere<3>& s, Interaction<3>& i,
									bool flipNormalOrientation=false, float squaredMinRadius=0.0f,
									float precision=1e-3f, bool recordNormal=false) const;

	// members
	const PolygonSoup<3> *soup;
	int indices[4];
};

} // namespace fcpw

#include "edge_silhouettes.inl"
