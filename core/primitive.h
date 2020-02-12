#pragma once

#include "ray.h"
#include "bounding_volumes.h"
#include "interaction.h"

namespace fcpw {

template <int DIM>
class Primitive {
public:
	// constructor
	Primitive(bool swapHandedness_): swapHandedness(swapHandedness_) {}

	// destructor
	virtual ~Primitive() {}

	// updates internal state if soup positions are modified
	virtual void update() = 0;

	// returns bounding box
	virtual BoundingBox<DIM> boundingBox() const = 0;

	// returns centroid
	virtual Vector<DIM> centroid() const = 0;

	// returns surface area
	virtual float surfaceArea() const = 0;

	// returns signed volume
	virtual float signedVolume() const = 0;

	// returns normal
	virtual Vector<DIM> normal(bool normalize=false) const = 0;

	// returns texture coordinates
	virtual Vector<DIM - 1> textureCoordinates(const Vector<DIM - 1>& uv) const = 0;

	// intersects with ray
	virtual int intersect(const Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
						  bool collectAll=false) const = 0;

	// finds closest point to sphere center
	virtual void findClosestPoint(const Vector<DIM>& x, Interaction<DIM>& i) const = 0;

	// member
	bool swapHandedness;
};

template <int DIM>
class Aggregate: public Primitive<DIM> {
public:
	// performs inside outside test for x
	bool contains(const Vector<DIM>& x, bool useRayIntersection=true) const {
		if (useRayIntersection) {
			// do two intersection tests to ensure correctness of resultness
			Vector<DIM> direction1 = Vector<DIM>::Zero();
			Vector<DIM> direction2 = Vector<DIM>::Zero();
			direction1(0) = 1;
			direction2(1) = 1;

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
};

template <int DIM>
class TransformedAggregate: public Aggregate<DIM> {
public:
	// constructor
	TransformedAggregate(const std::shared_ptr<Aggregate<DIM>>& aggregate_,
						 const Transform<float, DIM, Affine>& transform_):
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
				  bool checkOcclusion=false, bool countHits=false,
				  bool collectAll=false) const {
		// apply inverse transform to ray
		Ray<DIM> rInv = r.transform(transformInv);

		// intersect
		int hits = aggregate->intersect(rInv, is, checkOcclusion, countHits, collectAll);

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
			direction(0) = 1;
			distanceUpperBound = (transformInv*(x + distanceUpperBound*direction) - xInv).norm();
		}

		// clamp in object space and apply transform to x
		aggregate->clampToBoundary(xInv, distanceUpperBound);
		x = transform*xInv;
	}

private:
	// members
	std::shared_ptr<Aggregate<DIM>> aggregate;
	Transform<float, DIM, Affine> transform, transformInv;
	float determinant, sqrtDeterminant;
};

} // namespace fcpw
