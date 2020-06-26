#include "line_segments.h"

namespace fcpw {

LineSegment::LineSegment():
soup(nullptr),
pIndex(-1)
{

}

LineSegment::LineSegment(const PolygonSoup<3> *soup_, int pIndex_):
soup(soup_),
pIndex(pIndex_)
{

}

BoundingBox<3> LineSegment::boundingBox() const
{
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	BoundingBox<3> box(pa);
	box.expandToInclude(pb);

	return box;
}

Vector3 LineSegment::centroid() const
{
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	return (pa + pb)*0.5f;
}

float LineSegment::surfaceArea() const
{
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	return norm<3>(pb - pa);
}

float LineSegment::signedVolume() const
{
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	return 0.5f*cross(pa, pb)[2];
}

Vector3 LineSegment::normal(bool normalize) const
{
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	Vector3 s = pb - pa;
	Vector3 n(s[1], -s[0], 0);

	return normalize ? unit<3>(n) : n;
}

Vector3 LineSegment::normal(int vIndex) const
{
	if (soup->vNormals.size() > 0 && vIndex >= 0) {
		return soup->vNormals[soup->indices[pIndex + vIndex]];
	}

	return normal(true);
}

Vector3 LineSegment::normal(const Vector2& uv) const
{
	int vIndex = -1;
	if (uv[0] < epsilon) vIndex = 0;
	else if (uv[0] > oneMinusEpsilon) vIndex = 1;

	return normal(vIndex);
}

Vector2 LineSegment::barycentricCoordinates(const Vector3& p) const
{
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	return Vector2(norm<3>(p - pa)/norm<3>(pb - pa), 0.0f);
}

void LineSegment::split(int dim, float splitCoord, BoundingBox<3>& boxLeft,
						BoundingBox<3>& boxRight) const
{
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	if (pa[dim] <= splitCoord) {
		if (pb[dim] <= splitCoord) {
			boxLeft = BoundingBox<3>(pa);
			boxLeft.expandToInclude(pb);
			boxRight = BoundingBox<3>();

		} else {
			Vector3 u = pb - pa;
			float t = clamp((splitCoord - pa[dim])/u[dim], 0.0f, 1.0f);

			boxLeft = BoundingBox<3>(pa + u*t);
			boxRight = boxLeft;
			boxLeft.expandToInclude(pa);
			boxRight.expandToInclude(pb);
		}

	} else {
		if (pb[dim] >= splitCoord) {
			boxRight = BoundingBox<3>(pa);
			boxRight.expandToInclude(pb);
			boxLeft = BoundingBox<3>();

		} else {
			Vector3 u = pb - pa;
			float t = clamp((splitCoord - pa[dim])/u[dim], 0.0f, 1.0f);

			boxRight = BoundingBox<3>(pa + u*t);
			boxLeft = boxRight;
			boxRight.expandToInclude(pa);
			boxLeft.expandToInclude(pb);
		}
	}
}

int LineSegment::intersect(Ray<3>& r, std::vector<Interaction<3>>& is,
						   bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	is.clear();
	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	Vector3 u = pa - r.o;
	Vector3 v = pb - pa;

	// return if line segment and ray are parallel
	float dv = cross(r.d, v)[2];
	if (std::fabs(dv) < epsilon) return 0;

	// solve r.o + t*r.d = pa + s*(pb - pa) for t >= 0 && 0 <= s <= 1
	// s = (u x r.d)/(r.d x v)
	float ud = cross(u, r.d)[2];
	float s = ud/dv;

	if (s >= 0.0f && s <= 1.0f) {
		// t = (u x v)/(r.d x v)
		float uv = cross(u, v)[2];
		float t = uv/dv;

		if (t > epsilon && t <= r.tMax) {
			auto it = is.emplace(is.end(), Interaction<3>());
			it->d = t;
			it->p = r(t);
			it->uv[0] = s;
			it->uv[1] = -1;
			it->primitiveIndex = pIndex/2;

			return 1;
		}
	}

	return 0;
}

float findClosestPointLineSegment(const Vector3& pa, const Vector3& pb,
								  const Vector3& x, Vector3& pt, float& t)
{
	Vector3 u = pb - pa;
	Vector3 v = x - pa;

	float c1 = dot<3>(u, v);
	if (c1 <= 0.0f) {
		pt = pa;
		t = 0.0f;

		return norm<3>(x - pt);
	}

	float c2 = dot<3>(u, u);
	if (c2 <= c1) {
		pt = pb;
		t = 1.0f;

		return norm<3>(x - pt);
	}

	t = c1/c2;
	pt = pa + u*t;

	return norm<3>(x - pt);
}

bool LineSegment::findClosestPoint(BoundingSphere<3>& s, Interaction<3>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	const Vector3& pa = soup->positions[soup->indices[pIndex]];
	const Vector3& pb = soup->positions[soup->indices[pIndex + 1]];

	float d = findClosestPointLineSegment(pa, pb, s.c, i.p, i.uv[0]);

	if (d*d <= s.r2) {
		i.d = d;
		i.primitiveIndex = pIndex/2;
		i.uv[1] = -1;

		return true;
	}

	return false;
}

} // namespace fcpw
