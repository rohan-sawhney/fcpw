#pragma once

#include <fcpw/core/core.h>
#include <fcpw/core/ray.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace fcpw {

template<size_t DIM>
struct BoundingBox;

template<size_t DIM>
struct BoundingSphere {

	BoundingSphere() : c(Vector<DIM>::Zero()), r2(-1.0f) {}

	// constructor
	BoundingSphere(const Vector<DIM>& c_, float r2_): c(c_), r2(r2_) {}

	// computes transformed sphere
	BoundingSphere<DIM> transform(const Transform<DIM>& t) const {
		Vector<DIM> tc = t*c;
		float tr2 = -1.0f;

		if (r2 >= 0.0f) {
			Vector<DIM> direction = Vector<DIM>::Zero();
			direction[0] = 1;
			tr2 = (t*(c + direction*std::sqrt(r2)) - tc).squaredNorm();
		}

		return BoundingSphere<DIM>(tc, tr2);
	}

	BoundingBox<DIM> box() const {
		float r = std::sqrt(r2);
		return BoundingBox<DIM>(c.array() - r, c.array() + r);
	}
	
	std::vector<Vector<DIM>> points() const {
		return box().points();
	}

	// checks for ray intersection
	bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
		
		Vector<DIM> rel_pos = r.o - c;
		float b = 2.0f * rel_pos.dot(r.d);
		float c = rel_pos.squaredNorm() - r2;
		float d = b * b - 4 * c;

		if(d <= 0.0f) return false;

		float sqd = std::sqrt(d);

		tMin = std::max((-b - sqd) / 2.0f, 0.0f);
		tMax = std::min((-b + sqd) / 2.0f, r.tMax);
		return true;
	}

	bool overlap(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {

		float center_dist = (s.c - c).norm();
		float r = std::sqrt(r2);
		float close = std::max(center_dist - r, 0.0f);
		float far = center_dist + r;
		d2Min = close * close;
		d2Max = far * far;
		return d2Min <= s.r2;
	}

	static constexpr float PI_F = 3.1415926535897f;
	
	float surfaceArea() const {
		return 4.0f * PI_F * r2;
	}
	float volume() const {
		return (4.0f / 3.0f) * PI_F * r2 * std::sqrt(r2);
	}

	bool isValid() const {
		return r2 >= 0.0f;
	}

	void fromPoints(const std::vector<Vector<DIM>>& points) {
		*this = BoundingSphere();
		for(const auto& p : points) expandToInclude(p);
	}
	void fromPoints(const std::vector<std::vector<Vector<DIM>>>& points) {
		*this = BoundingSphere();
		for(const auto& l : points)
			for(const auto& p : l) 
				expandToInclude(p);
	}

	BoundingSphere<DIM> intersect(const BoundingSphere<DIM>& b) const {
		float P = (b.c - c).squaredNorm();
		float Q = (r2 - b.r2 + P) / (2.0f * P);
		Vector<DIM> B = c + Q * (b.c - c);
		float R = r2 - (B - c).squaredNorm();
		return BoundingSphere(B, R);
	}

	Vector<DIM> c;
	float r2;

private:
	void expandToInclude(const Vector<DIM>& p) {
		if(r2 < 0.0f) {
			c = p;
			r2 = 0.0f;
		} else {
			r2 = std::max(r2, (p - c).squaredNorm());
		}
	}
};

template<size_t DIM>
struct OrientedBoundingBox {
	
	static_assert(DIM == 3);

	OrientedBoundingBox() : e(Vector3::Constant(-1.0f)) {}

	bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
		
		BoundingBox<DIM> local = local_box();
		Ray<DIM> rt = r.transform(T);
		return local.intersect(rt, tMin, tMax);
	}

	bool overlap(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {

		BoundingBox<DIM> local = local_box();
		BoundingSphere<DIM> st = s.transform(T);
		return local.overlap(st, d2Min, d2Max);
	}

	OrientedBoundingBox<DIM> intersect(const OrientedBoundingBox<DIM>& b) const {
		// std::cerr << "Warning: OBBxOBB->OBB intersection not supported" << std::endl;
		return OrientedBoundingBox();
	}

	// computes transformed sphere
	OrientedBoundingBox<DIM> transform(const Transform<DIM>& t) const {
		OrientedBoundingBox<DIM> ret;
		
		auto pt = points();
		for(auto& p : pt) p = t * p;

		ret.fromPoints(pt);
		return ret;
	}

	BoundingBox<DIM> box() const {
		auto pt = points();
		BoundingBox<DIM> ret;
		ret.fromPoints(pt);
		return ret;
	}

	BoundingBox<DIM> local_box() const {
		BoundingBox<DIM> ret;
		ret.pMin = -e;
		ret.pMax = e;
		return ret;
	}

	Eigen::Matrix3f rot_mat() const {
		return (T.matrix().template block<3,3>(0,0)).transpose();
	}

	Vector<DIM> center() const {
		Vector<DIM> v = T.matrix().template block<3,1>(0,3);
		return rot_mat() * -v;
	}
	
	std::vector<Vector<DIM>> points() const {
		
		Vector<DIM> c = -T.matrix().template block<3,1>(0,3);
		Vector<DIM> min = c-e;
		Vector<DIM> max = c+e;

		std::vector<Vector<DIM>> v;
		Eigen::Matrix3f u = rot_mat();
		v.push_back(u * Vector<DIM>(min.x(), min.y(), min.z()));
		v.push_back(u * Vector<DIM>(max.x(), min.y(), min.z()));
		v.push_back(u * Vector<DIM>(min.x(), max.y(), min.z()));
		v.push_back(u * Vector<DIM>(min.x(), min.y(), max.z()));
		v.push_back(u * Vector<DIM>(max.x(), max.y(), min.z()));
		v.push_back(u * Vector<DIM>(min.x(), max.y(), max.z()));
		v.push_back(u * Vector<DIM>(max.x(), min.y(), max.z()));
		v.push_back(u * Vector<DIM>(max.x(), max.y(), max.z()));

		return v;
	}

	float surfaceArea() const {
		Vector<DIM> ext = (2.0f * e).cwiseMax(1e-5f);
		return 2.0f*Vector<DIM>::Constant(ext.prod()).cwiseQuotient(ext).sum();
	}

	float volume() const {
		return (2.0f * e).prod();
	}

	bool isValid() const {
		return (e.array() >= Vector3::Constant(0.0f).array()).all();
	}

	void fromPoints(const std::vector<Vector<DIM>>& points) {
		*this = fitPCA(points);
	}
	void fromPoints(const std::vector<std::vector<Vector<DIM>>>& points) {
		std::vector<Vector<DIM>> flat;
		for(const auto& l : points) flat.insert(flat.end(), l.begin(), l.end());
		*this = fitPCA(flat);
	}

	static OrientedBoundingBox<DIM> fitPCA(const std::vector<Vector3>& points) {
		
		Vector3 center;
		center.setZero();
		for(const auto& v : points) {
			center += v;
		}
		center /= (float)points.size();
		
		// adjust for mean and compute covariance
		Eigen::Matrix3f covariance;
		covariance.setZero();
		for(const auto& v : points) {
			Vector3 pAdg = v - center;
			covariance += pAdg * pAdg.transpose();
		}
		covariance /= (float)points.size();

		// compute eigenvectors for the covariance matrix
		Eigen::EigenSolver<Eigen::Matrix3f> solver(covariance);
		Eigen::Matrix3f eigenVectors = solver.eigenvectors().real();

		// project min and max points on each principal axis
		float min1 = INFINITY, max1 = -INFINITY;
		float min2 = INFINITY, max2 = -INFINITY;
		float min3 = INFINITY, max3 = -INFINITY;
		float d = 0.0;
		eigenVectors.transposeInPlace();
		for(const auto& v : points) {
			d = eigenVectors.row(0).dot(v);
			if (min1 > d) min1 = d;
			if (max1 < d) max1 = d;
			
			d = eigenVectors.row(1).dot(v);
			if (min2 > d) min2 = d;
			if (max2 < d) max2 = d;
			
			d = eigenVectors.row(2).dot(v);
			if (min3 > d) min3 = d;
			if (max3 < d) max3 = d;
		}
		
		OrientedBoundingBox<DIM> ret;

		ret.e.x() = (max1 - min1) / 2.0f;
		ret.e.y() = (max2 - min2) / 2.0f;
		ret.e.z() = (max3 - min3) / 2.0f;

		Vector<DIM> c = (Vector3(min1,min2,min3) + Vector3(max1,max2,max3)) / 2.0f;

		Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
		T.block<3,3>(0,0) = eigenVectors;
		T.block<3,1>(0,3) = -c;
		ret.T.matrix() = T;
		return ret;
	}

	Vector3 e;
	Transform<DIM> T;
};

template<size_t DIM>
struct SphereSweptRect {
	
	static_assert(DIM == 3);

	SphereSweptRect() : r(-1.0f) {}

	bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
		return box().intersect(r, tMin, tMax);	
	}

	bool overlap(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {
		
		Vector3 p0 = c - e0 - e1;
		Vector3 p1 = c + e0 - e1;
		Vector3 p2 = c - e0 + e1;
		Vector3 p3 = c + e0 + e1;
		Vector3 d0, d1;
		Vector2 t0, t1;
		float c0 = findClosestPointTriangle(p0, p1, p2, s.c, d0, t0);
		float c1 = findClosestPointTriangle(p1, p2, p3, s.c, d1, t1);
		d2Min = std::min(c0, c1) - r - std::sqrt(s.r2);
		d2Max = FLT_MAX;
		return d2Min <= 0.0f;
	}

	SphereSweptRect<DIM> intersect(const SphereSweptRect<DIM>& b) const {
		// std::cerr << "Warning: RSSxRSS->RSS intersection not supported" << std::endl;
		return SphereSweptRect<DIM>();
	}

	// computes transformed sphere
	SphereSweptRect<DIM> transform(const Transform<DIM>& t) const {
		SphereSweptRect<DIM> ret;
		ret.c = t * c;
		ret.e0 = t * Vector<4>(e0, 0.0f);
		ret.e1 = t * Vector<4>(e1, 0.0f);
		auto v = Vector<4>(1.0f, 1.0f, 1.0f, 0.0f);
		float scale = (t * v).norm() / v.norm();
		ret.r = r * scale;
		return ret;
	}

	BoundingBox<DIM> box() const {
		BoundingBox<DIM> ret;
		ret.expandToInclude(c - e0 - e1);
		ret.expandToInclude(c - e0 + e1);
		ret.expandToInclude(c + e0 - e1);
		ret.expandToInclude(c + e0 + e1);
		ret.expandToInclude(ret.pMin - Vector3::Constant(r));
		ret.expandToInclude(ret.pMax + Vector3::Constant(r));
		return ret;
	}

	Vector<DIM> center() const {
		return c;
	}
	
	std::vector<Vector<DIM>> points() const {
		return box().points();
	}

	static constexpr float PI_F = 3.1415926535897f;

	float surfaceArea() const {
		float w = 2.0f * e0.norm();
		float h = 2.0f * e1.norm();
		return 2.0f*w*h + 2.0f*PI_F*r*w + 2.0f*PI_F*r*h + PI_F*r*r;
	}

	float volume() const {
		float w = 2.0f * e0.norm();
		float h = 2.0f * e1.norm();
		return 2.0f*r*w*h + PI_F*r*r*w + PI_F*r*r*h + (4.0f/3.0f)*PI_F*r*r*r;
	}

	bool isValid() const {
		return r >= 0.0f;
	}

	void fromPoints(const std::vector<Vector<DIM>>& points) {
		*this = fit(points);
	}
	void fromPoints(const std::vector<std::vector<Vector<DIM>>>& points) {
		std::vector<Vector<DIM>> flat;
		for(const auto& l : points) flat.insert(flat.end(), l.begin(), l.end());
		*this = fit(flat);
	}

	static SphereSweptRect<DIM> fit(const std::vector<Vector3>& points) {
		
		OrientedBoundingBox<DIM> obb = OrientedBoundingBox<DIM>::fitPCA(points);
		SphereSweptRect<DIM> ret;

		Eigen::Matrix3f u = obb.rot_mat();
		Vector3 c = obb.center();
		Vector3 e = obb.e;

		int a0, a1;
		e.maxCoeff(&a0);
		e[a0] = -1.0f;
		e.maxCoeff(&a1);

		Vector3 e0 = Vector3::Zero();
		Vector3 e1 = Vector3::Zero();
		e0[a0] = obb.e[a0];
		e1[a1] = obb.e[a1];

		ret.c = c;
		ret.e0 = u * e0;
		ret.e1 = u * e1;

		Vector3 norm = ret.e0.cross(ret.e1).normalized();
		float d = norm.dot(ret.c);

		ret.r = 0.0f;
		
		for(const auto& v : points) {
			float pd = norm.dot(v) - d;
			ret.r = std::max(ret.r, pd);
		}

		return ret;
	}

	Vector3 e0;
	Vector3 e1;
	Vector3 c;
	float r;
};

template<size_t DIM>
struct BoundingBox {
	// constructor
	BoundingBox(): pMin(Vector<DIM>::Constant(maxFloat)),
				   pMax(Vector<DIM>::Constant(minFloat)) {}

	// constructor
	BoundingBox(const Vector<DIM>& p) {
		Vector<DIM> epsilonVector = Vector<DIM>::Constant(epsilon);
		pMin = p - epsilonVector;
		pMax = p + epsilonVector;
	}

	BoundingBox(const Vector<DIM>& pMin, const Vector<DIM>& pMax) : pMin(pMin), pMax(pMax) {}

	BoundingBox box() const {
		return *this;
	}

	void fromPoints(const std::vector<Vector<DIM>>& points) {
		*this = BoundingBox();
		for(const auto& p : points) expandToInclude(p);
	}
	void fromPoints(const std::vector<std::vector<Vector<DIM>>>& points) {
		*this = BoundingBox();
		for(const auto& l : points)
			for(const auto& p : l)
				expandToInclude(p);
	}

	// returns box extent
	Vector<DIM> extent() const {
		return pMax - pMin;
	}

	// computes min and max squared distance to point;
	// min squared distance is 0 if point is inside box
	void computeSquaredDistance(const Vector<DIM>& p, float& d2Min, float& d2Max) const {
		Vector<DIM> u = pMin - p;
		Vector<DIM> v = p - pMax;
		d2Min = u.cwiseMax(v).cwiseMax(0.0f).squaredNorm();
		d2Max = u.cwiseMin(v).squaredNorm();
	}

	// checks whether box contains point
	bool contains(const Vector<DIM>& p) const {
		return (p.array() >= pMin.array()).all() &&
			   (p.array() <= pMax.array()).all();
	}

	// checks for overlap with sphere
	bool overlap(const BoundingSphere<DIM>& s, float& d2Min, float& d2Max) const {
		computeSquaredDistance(s.c, d2Min, d2Max);
		return d2Min <= s.r2;
	}

	// checks for overlap with bounding box
	bool overlap(const BoundingBox<DIM>& b) const {
		return (b.pMax.array() >= pMin.array()).all() &&
			   (b.pMin.array() <= pMax.array()).all();
	}

	// checks for ray intersection
	bool intersect(const Ray<DIM>& r, float& tMin, float& tMax) const {
		// slab test for ray box intersection
		// source: http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf
		Vector<DIM> t0 = (pMin - r.o).cwiseProduct(r.invD);
		Vector<DIM> t1 = (pMax - r.o).cwiseProduct(r.invD);
		Vector<DIM> tNear = t0.cwiseMin(t1);
		Vector<DIM> tFar = t0.cwiseMax(t1);

		float tNearMax = std::max(0.0f, tNear.maxCoeff());
		float tFarMin = std::min(r.tMax, tFar.minCoeff());
		if (tNearMax > tFarMin) return false;

		tMin = tNearMax;
		tMax = tFarMin;
		return true;
	}

	// checks whether bounding box is valid
	bool isValid() const {
		return (pMax.array() >= pMin.array()).all();
	}

	// returns max dimension
	int maxDimension() const {
		int index;
		float maxLength = (pMax - pMin).maxCoeff(&index);
		return index;
	}

	// returns centroid
	Vector<DIM> centroid() const {
		return (pMin + pMax)*0.5f;
	}

	// returns surface area
	float surfaceArea() const {
		Vector<DIM> e = extent().cwiseMax(1e-5f); // the 1e-5 is to prevent division by zero
		return 2.0f*Vector<DIM>::Constant(e.prod()).cwiseQuotient(e).sum();
	}

	// returns volume
	float volume() const {
		return extent().prod();
	}

	// computes transformed box
	BoundingBox<DIM> transform(const Transform<DIM>& t) const {
		BoundingBox<DIM> b;
		int nCorners = 1 << DIM;

		for (int i = 0; i < nCorners; i++) {
			Vector<DIM> p = Vector<DIM>::Zero();
			int temp = i;

			for (size_t j = 0; j < DIM; j++) {
				int idx = temp%2;
				p[j] = idx == 0 ? pMin[j] : pMax[j];
				temp /= 2;
			}

			b.expandToInclude(t*p);
		}

		return b;
	}

	std::vector<Vector<DIM>> points() const {

		std::vector<Vector<DIM>> pts;
		int nCorners = 1 << DIM;

		for (int i = 0; i < nCorners; i++) {
			Vector<DIM> p = Vector<DIM>::Zero();
			int temp = i;
			for (size_t j = 0; j < DIM; j++) {
				int idx = temp%2;
				p[j] = idx == 0 ? pMin[j] : pMax[j];
				temp /= 2;
			}
			pts.push_back(p);
		}

		return pts;
	}

	// returns the intersection of two bounding boxes
	BoundingBox<DIM> intersect(const BoundingBox<DIM>& b) const {
		BoundingBox<DIM> bIntersect;
		bIntersect.pMin = pMin.cwiseMax(b.pMin);
		bIntersect.pMax = pMax.cwiseMin(b.pMax);

		return bIntersect;
	}

	Vector<DIM> pMin, pMax;

	void expandToInclude(const Vector<DIM>& p) {
		Vector<DIM> epsilonVector = Vector<DIM>::Constant(epsilon);
		pMin = pMin.cwiseMin(p - epsilonVector);
		pMax = pMax.cwiseMax(p + epsilonVector);
	}
};


} // namespace fcpw
