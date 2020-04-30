#pragma once

#include "baseline.h"
#include "geometry/polygon_soup.h"
#include <include/embree3/rtcore.h>

namespace fcpw {

template <int DIM>
class EmbreeBvh: public Baseline<DIM> {
public:
	// constructor
	EmbreeBvh(const std::vector<std::shared_ptr<Primitive<DIM>>>& primitives_,
			  const std::shared_ptr<PolygonSoup<DIM>>& soup_);

	// destructor
	~EmbreeBvh();

	// returns bounding box
	BoundingBox<DIM> boundingBox() const;

	// returns centroid
	Vector<DIM> centroid() const;

	// returns surface area
	float surfaceArea() const;

	// returns signed volume
	float signedVolume() const;

	// intersects with ray, starting the traversal at the specified node;
	// use this for spatially/temporally coherent queries
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is, int startNodeIndex,
						  bool checkOcclusion=false, bool countHits=false) const;

	// intersects with ray
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkOcclusion=false, bool countHits=false) const;

	// finds closest point to sphere center, starting the traversal at the specified node;
	// use this for spatially/temporally coherent queries
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int startNodeIndex) const;

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

protected:
	// members
	std::shared_ptr<PolygonSoup<DIM>> soup;
	RTCDevice device;
	RTCScene scene;
};

} // namespace fcpw

#include "embree_bvh.inl"
