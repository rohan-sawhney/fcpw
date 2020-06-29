#pragma once

#include "baseline.h"
#include "geometry/polygon_soup.h"
#include <include/embree3/rtcore.h>

namespace fcpw {

class EmbreeBvh: public Baseline<3, Triangle> {
public:
	// constructor
	EmbreeBvh(const std::vector<Triangle *>& triangles_,
			  const PolygonSoup<3> *soup_, bool printStats_=false);

	// destructor
	~EmbreeBvh();

	// returns bounding box
	BoundingBox<3> boundingBox() const;

	// returns centroid
	Vector3 centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	// NOTE: interactions are invalid when checkOcclusion is enabled
	int intersectFromNode(Ray<3>& r, std::vector<Interaction<3>>& is,
						  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
						  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	bool findClosestPointFromNode(BoundingSphere<3>& s, Interaction<3>& i,
								  int nodeStartIndex, int aggregateIndex,
								  const Vector3& boundaryHint, int& nodesVisited) const;

protected:
	// members
	RTCDevice device;
	RTCScene scene;
};

} // namespace fcpw

#include "embree_bvh.inl"
