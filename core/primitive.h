#pragma once

#include "ray.h"
#include "bounding_volumes.h"
#include "interaction.h"

namespace fcpw {

template <int DIM>
class Primitive {
public:
	// constructor
	Primitive(): swapHandedness(false) {}
	Primitive(bool swapHandedness_): swapHandedness(swapHandedness_) {}

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
	virtual void split(int dim, float splitCoord, BoundingBox<DIM>& bboxLeft,
					   BoundingBox<DIM>& bboxRight) const = 0;

	// intersects with ray
	virtual int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  bool checkOcclusion=false, bool countHits=false) const = 0;

	// finds closest point to sphere center
	virtual bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const = 0;

	// member
	bool swapHandedness;
};

template <int DIM>
class Aggregate: public Primitive<DIM> {
public:
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

		LOG_IF(FATAL, !found) << "Cannot clamp to boundary since no closest point was found inside distance bound: "
							  << distanceUpperBound;
		LOG_IF(FATAL, i.distanceInfo == DistanceInfo::Bounded)
							  << "Cannot clamp to boundary since exact distance isn't available";
		x = i.p;
	}

	// splits the primitive along the provided coordinate and axis
	void split(int dim, float splitCoord, BoundingBox<DIM>& bboxLeft,
			   BoundingBox<DIM>& bboxRight) const {
		BoundingBox<DIM> bbox = this->boundingBox();

		if (bbox.pMin[dim] <= splitCoord) {
			bboxLeft = bbox;
			bboxLeft.pMax[dim] = splitCoord;
		}

		if (bbox.pMax[dim] >= splitCoord) {
			bboxRight = bbox;
			bboxRight.pMin[dim] = splitCoord;
		}
	}
};

template <int DIM>
class TransformedAggregate: public Aggregate<DIM> {
public:
	// constructor
	TransformedAggregate(const std::shared_ptr<Aggregate<DIM>>& aggregate_, const Transform<DIM>& transform_):
						 aggregate(aggregate_), transform(transform_), transformInv(transform.inverse()),
						 determinant(transform.matrix().determinant()), sqrtDeterminant(determinant) {}

	// returns bounding box
	BoundingBox<DIM> boundingBox() const {
		return aggregate->boundingBox().transform(transform);
	}

	// returns centroid
	Vector<DIM> centroid() const {
		return transform*aggregate->centroid();
	}

	// returns surface area
	float surfaceArea() const {
		// NOTE: this is an approximate estimate
		return sqrtDeterminant*aggregate->surfaceArea();
	}

	// returns signed volume
	float signedVolume() const {
		// NOTE: this is an approximate estimate
		return determinant*aggregate->signedVolume();
	}

	// intersects with ray
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkOcclusion=false, bool countHits=false) const {
		// apply inverse transform to ray
		Ray<DIM> rInv = r.transform(transformInv);

		// intersect
		int hits = aggregate->intersect(rInv, is, checkOcclusion, countHits);

		// apply transform to ray and interactions
		r.tMax = rInv.transform(transform).tMax;
		if (hits > 0) {
			for (int i = 0; i < (int)is.size(); i++) {
				is[i].applyTransform(transform, transformInv, r.o);
			}
		}

		return hits;
	}

	// finds closest point to sphere center
	bool findClosestPoint(BoundingSphere<DIM>& s, Interaction<DIM>& i) const {
		// apply inverse transform to sphere
		BoundingSphere<DIM> sInv = s.transform(transformInv);

		// find closest point
		bool found = aggregate->findClosestPoint(sInv, i);

		// apply transform to sphere and interaction
		s.r2 = sInv.transform(transform).r2;
		if (found) i.applyTransform(transform, transformInv, s.c);

		return found;
	}

	// performs inside outside test for x
	bool contains(const Vector<DIM>& x, bool useRayIntersection=true) const {
		return aggregate->contains(transformInv*x, useRayIntersection);
	}

	// checks whether there is a line of sight between xi and xj
	bool hasLineOfSight(const Vector<DIM>& xi, const Vector<DIM>& xj) const {
		return aggregate->hasLineOfSight(transformInv*xi, transformInv*xj);
	}

	// clamps x to the closest primitive this aggregate bounds
	void clampToBoundary(Vector<DIM>& x, float distanceUpperBound) const {
		// apply inverse transform to x and distance bound
		Vector<DIM> xInv = transformInv*x;
		if (distanceUpperBound < maxFloat) {
			Vector<DIM> direction = Vector<DIM>::Zero();
			direction[0] = 1;
			distanceUpperBound = (transformInv*(x + distanceUpperBound*direction) - xInv).norm();
		}

		// clamp in object space and apply transform to x
		aggregate->clampToBoundary(xInv, distanceUpperBound);
		x = transform*xInv;
	}

private:
	// members
	std::shared_ptr<Aggregate<DIM>> aggregate;
	Transform<DIM> transform, transformInv;
	float determinant, sqrtDeterminant;
};

} // namespace fcpw
