#pragma once

#include <fcpw/core/ray.h>
#include <fcpw/core/bounding_volumes.h>
#include <fcpw/core/interaction.h>

namespace fcpw {

template<size_t DIM>
class Primitive {
public:
	// constructor
	Primitive() {}

	// destructor
	virtual ~Primitive() {}

	// returns bounding box
	virtual BoundingBox<DIM> boundingBox() const = 0;

	// returns centroid
	virtual Vector<DIM> centroid() const = 0;

	// returns surface area
	virtual float surfaceArea() const = 0;

	// returns signed volume
	virtual float signedVolume() const = 0;

	// splits the primitive along the provided coordinate and axis
	virtual void split(int dim, float splitCoord, BoundingBox<DIM>& boxLeft,
					   BoundingBox<DIM>& boxRight) const = 0;

	// intersects with ray
	virtual int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  bool checkForOcclusion=false, bool recordAllHits=false) const = 0;

	// finds closest point to sphere center
	virtual bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const = 0;
};

template<size_t DIM>
class GeometricPrimitive: public Primitive<DIM> {
public:
	// returns normal
	virtual Vector<DIM> normal(bool normalize=false) const = 0;

	// returns the normalized normal based on the local parameterization
	virtual Vector<DIM> normal(const Vector<DIM - 1>& uv) const = 0;

	// returns barycentric coordinates
	virtual Vector<DIM - 1> barycentricCoordinates(const Vector<DIM>& p) const = 0;
};

template<size_t DIM>
class Aggregate: public Primitive<DIM> {
public:
	// splits the primitive along the provided coordinate and axis
	void split(int dim, float splitCoord, BoundingBox<DIM>& boxLeft,
			   BoundingBox<DIM>& boxRight) const {
		BoundingBox<DIM> box = this->boundingBox();

		if (box.pMin[dim] <= splitCoord) {
			boxLeft = box;
			boxLeft.pMax[dim] = splitCoord;
		}

		if (box.pMax[dim] >= splitCoord) {
			boxRight = box;
			boxRight.pMin[dim] = splitCoord;
		}
	}

	// intersects with ray
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkForOcclusion=false, bool recordAllHits=false) const {
		int nodesVisited = 0;
		return this->intersectFromNode(r, is, 0, this->index, nodesVisited, checkForOcclusion, recordAllHits);
	}

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const {
		int nodesVisited = 0;
		return this->findClosestPointFromNode(s, i, 0, this->index, Vector<DIM>::Zero(), nodesVisited);
	}

	// performs inside outside test for x
	// NOTE: assumes aggregate bounds watertight shape
	bool contains(const Vector<DIM>& x, bool useRayIntersection=true) const {
		if (useRayIntersection) {
			// do two intersection tests for robustness
			Vector<DIM> direction1 = Vector<DIM>::Zero();
			Vector<DIM> direction2 = Vector<DIM>::Zero();
			direction1[0] = 1;
			direction2[1] = 1;

			std::vector<Interaction<DIM>> is1;
			Ray<DIM> r1(x, direction1);
			int hits1 = this->intersect(r1, is1, false, true);

			std::vector<Interaction<DIM>> is2;
			Ray<DIM> r2(x, direction2);
			int hits2 = this->intersect(r2, is2, false, true);

			return hits1%2 == 1 && hits2%2 == 1;
		}

		Interaction<DIM> i;
		BoundingSphere<DIM> s(x, maxFloat);
		bool found = this->findClosestPoint(s, i);

		return i.signedDistance(x) < 0;
	}

	// checks whether there is a line of sight between xi and xj
	bool hasLineOfSight(const Vector<DIM>& xi, const Vector<DIM>& xj) const {
		Vector<DIM> direction = xj - xi;
		float dNorm = direction.norm();
		direction /= dNorm;

		std::vector<Interaction<DIM>> is;
		Ray<DIM> r(xi, direction, dNorm);
		int hits = this->intersect(r, is, true);

		return hits == 0;
	}

	// clamps x to the closest primitive this aggregate bounds
	void clampToBoundary(Vector<DIM>& x, float distanceUpperBound) const {
		Interaction<DIM> i;
		BoundingSphere<DIM> s(x, distanceUpperBound*distanceUpperBound);
		bool found = this->findClosestPoint(s, i);

		if (!found) {
			std::cerr << "Aggregate::clampToBoundary(): Cannot clamp to boundary since no "
					  << "closest point was found inside distance bound: " << distanceUpperBound
					  << std::endl;
		}

		if (i.distanceInfo == DistanceInfo::Bounded) {
			std::cerr << "Aggregate::clampToBoundary(): Cannot clamp to boundary since exact "
					  << "distance isn't available" << std::endl;
		}

		x = i.p;
	}

	// intersects with ray, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	// NOTE: interactions are invalid when checkForOcclusion is enabled
	virtual int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
								  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
								  bool checkForOcclusion=false, bool recordAllHits=false) const = 0;

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	virtual bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
										  int nodeStartIndex, int aggregateIndex,
										  const Vector<DIM>& boundaryHint, int& nodesVisited) const = 0;

	// members
	int index;
	bool computeNormals;
};

template<size_t DIM>
class TransformedAggregate: public Aggregate<DIM> {
public:
	// constructor
	TransformedAggregate(const std::shared_ptr<Aggregate<DIM>>& aggregate_,
						 const Transform<DIM>& transform_):
						 aggregate(aggregate_), t(transform_), tInv(t.inverse()),
						 det(t.matrix().determinant()), sqrtDet(std::sqrt(det)) {
		this->computeNormals = false;
	}

	// returns bounding box
	BoundingBox<DIM> boundingBox() const {
		return aggregate->boundingBox().transform(t);
	}

	// returns centroid
	Vector<DIM> centroid() const {
		return transformVector<DIM>(t, aggregate->centroid());
	}

	// returns surface area
	float surfaceArea() const {
		// NOTE: this is an approximate estimate
		return sqrtDet*aggregate->surfaceArea();
	}

	// returns signed volume
	float signedVolume() const {
		// NOTE: this is an approximate estimate
		return det*aggregate->signedVolume();
	}

	// intersects with ray, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	int intersectFromNode(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  int nodeStartIndex, int aggregateIndex, int& nodesVisited,
						  bool checkForOcclusion=false, bool recordAllHits=false) const {
		// apply inverse transform to ray
		Ray<DIM> rInv = r.transform(tInv);

		// intersect
		int hits = aggregate->intersectFromNode(rInv, is, nodeStartIndex, aggregateIndex,
												nodesVisited, checkForOcclusion, recordAllHits);

		// apply transform to ray and interactions
		r.tMax = rInv.transform(t).tMax;
		if (hits > 0) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].applyTransform(t, tInv, r.o);
			}
		}

		nodesVisited++;
		return hits;
	}

	// finds closest point to sphere center, starting the traversal at the specified node in an aggregate;
	// use this for spatially/temporally coherent queries
	bool findClosestPointFromNode(BoundingSphere<DIM>& s, Interaction<DIM>& i,
								  int nodeStartIndex, int aggregateIndex,
								  const Vector<DIM>& boundaryHint, int& nodesVisited) const {
		// apply inverse transform to sphere
		BoundingSphere<DIM> sInv = s.transform(tInv);

		// apply inverse transform to direction guess
		Vector<DIM> boundaryHintInv = boundaryHint;
		if (squaredNorm<DIM>(boundaryHint) > 0.0f) {
			boundaryHintInv = transformVector<DIM>(tInv, s.c + boundaryHint) - sInv.c;
			float hintNorm = boundaryHintInv.norm();
			boundaryHintInv /= hintNorm;
		}

		// find closest point
		bool found = aggregate->findClosestPointFromNode(sInv, i, nodeStartIndex, aggregateIndex,
														 boundaryHintInv, nodesVisited);

		// apply transform to sphere and interaction
		s.r2 = sInv.transform(t).r2;
		if (found) i.applyTransform(t, tInv, s.c);

		nodesVisited++;
		return found;
	}

	// performs inside outside test for x
	bool contains(const Vector<DIM>& x, bool useRayIntersection=true) const {
		return aggregate->contains(transformVector<DIM>(tInv, x), useRayIntersection);
	}

	// checks whether there is a line of sight between xi and xj
	bool hasLineOfSight(const Vector<DIM>& xi, const Vector<DIM>& xj) const {
		return aggregate->hasLineOfSight(transformVector<DIM>(tInv, xi),
										 transformVector<DIM>(tInv, xj));
	}

	// clamps x to the closest primitive this aggregate bounds
	void clampToBoundary(Vector<DIM>& x, float distanceUpperBound) const {
		// apply inverse transform to x and distance bound
		Vector<DIM> xInv = transformVector<DIM>(tInv, x);
		if (distanceUpperBound < maxFloat) {
			Vector<DIM> direction = Vector<DIM>::Zero();
			direction[0] = 1;
			distanceUpperBound = (transformVector<DIM>(tInv, x + direction*distanceUpperBound) - xInv).norm();
		}

		// clamp in object space and apply transform to x
		aggregate->clampToBoundary(xInv, distanceUpperBound);
		x = transformVector<DIM>(t, xInv);
	}

private:
	// members
	std::shared_ptr<Aggregate<DIM>> aggregate;
	Transform<DIM> t, tInv;
	float det, sqrtDet;
};

} // namespace fcpw
