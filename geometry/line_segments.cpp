#include "line_segments.h"
#include <fstream>
#include <sstream>
#include <map>

namespace fcpw {

LineSegment::LineSegment(const std::shared_ptr<PolygonSoup<3>>& soup_,
						 bool isFlat_, int index_):
Primitive<3>(),
soup(soup_),
indices(soup_->indices[index_]),
isFlat(isFlat_)
{

}

BoundingBox<3> LineSegment::boundingBox() const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];

	BoundingBox<3> box(pa);
	box.expandToInclude(pb);

	return box;
}

Vector3 LineSegment::centroid() const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];

	return (pa + pb)*0.5f;
}

float LineSegment::surfaceArea() const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];

	return norm<3>(pb - pa);
}

float LineSegment::signedVolume() const
{
	if (isFlat) {
		const Vector3& pa = soup->positions[indices[0]];
		const Vector3& pb = soup->positions[indices[1]];

		return 0.5f*cross(pa, pb)[2];
	}

	// signedVolume is undefined in 3d
	return 0.0f;
}

Vector3 LineSegment::normal(bool normalize) const
{
	if (isFlat) {
		const Vector3& pa = soup->positions[indices[0]];
		const Vector3& pb = soup->positions[indices[1]];

		Vector3 s = pb - pa;
		Vector3 n(s[1], -s[0], 0);

		return normalize ? unit<3>(n) : n;
	}

	// normal is undefined in 3d
	return Vector3();
}

Vector3 LineSegment::normal(int vIndex) const
{
	if (soup->vNormals.size() > 0 && vIndex >= 0) return soup->vNormals[indices[vIndex]];
	return normal(true);
}

Vector3 LineSegment::normal(const Vector2& uv) const
{
	int vIndex = -1;
	if (uv[0] < epsilon) vIndex = 0;
	else if (uv[0] > oneMinusEpsilon) vIndex = 1;

	return normal(vIndex);
}

float LineSegment::barycentricCoordinates(const Vector3& p) const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];

	return norm<3>(p - pa)/norm<3>(pb - pa);
}

void LineSegment::split(int dim, float splitCoord, BoundingBox<3>& boxLeft,
						BoundingBox<3>& boxRight) const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];

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

	if (isFlat) {
		const Vector3& pa = soup->positions[indices[0]];
		const Vector3& pb = soup->positions[indices[1]];

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
				it->primitive = this;

				return 1;
			}
		}

		return 0;
	}

	// not implemented for 3d
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

	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];

	float d = findClosestPointLineSegment(pa, pb, s.c, i.p, i.uv[0]);

	if (d*d <= s.r2) {
		i.d = d;
		i.primitive = this;
		i.uv[1] = -1;

		return true;
	}

	return false;
}

void computeWeightedLineSegmentNormals(const std::vector<std::shared_ptr<Primitive<3>>>& lineSegments,
									   std::shared_ptr<PolygonSoup<3>>& soup)
{
	int N = (int)soup->indices.size();
	int V = (int)soup->positions.size();
	soup->vNormals.resize(V, zeroVector<3>());

	for (int i = 0; i < N; i++) {
		Vector3 n = static_cast<const LineSegment *>(lineSegments[i].get())->normal(true);
		soup->vNormals[soup->indices[i][0]] += n;
		soup->vNormals[soup->indices[i][1]] += n;
	}

	for (int i = 0; i < V; i++) {
		soup->vNormals[i] = unit<3>(soup->vNormals[i]);
	}
}

std::shared_ptr<PolygonSoup<3>> readLineSegmentSoupFromOBJFile(const std::string& filename,
															   bool closeLoop, bool& isFlat)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// initialize
	std::shared_ptr<PolygonSoup<3>> soup = std::make_shared<PolygonSoup<3>>();
	std::ifstream in(filename);
	LOG_IF(FATAL, in.is_open() == false) << "Unable to open file: " << filename;

	// parse obj format
	std::string line;
	bool tokenIsL = false;
	isFlat = true;

	while (getline(in, line)) {
		std::stringstream ss(line);
		std::string token;
		ss >> token;

		if (token == "v") {
			float x, y, z;
			ss >> x >> y >> z;

			soup->positions.emplace_back(Vector3(x, y, z));
			if (std::fabs(z) > epsilon) isFlat = false;

		} else if (token == "f" || token == "l") {
			tokenIsL = token == "l";
			std::vector<int> indices;

			while (ss >> token) {
				Index index = parseFaceIndex(token);

				if (index.position < 0) {
					getline(in, line);
					size_t i = line.find_first_not_of("\t\n\v\f\r ");
					index = parseFaceIndex(line.substr(i));
				}

				indices.emplace_back(index.position);
			}

			if (tokenIsL) {
				soup->indices.emplace_back(indices);

			} else {
				int F = (int)indices.size();
				int N = closeLoop ? F : F - 1;
				for (int i = 0; i < N; i++) {
					int j = (i + 1)%F;
					soup->indices.emplace_back(std::vector<int>{indices[i], indices[j]});
				}
			}
		}
	}

	// close loop
	if (tokenIsL && closeLoop) {
		int F = (int)soup->indices.size();

		if (F > 0) {
			int startIndex = soup->indices[0][0];
			int endIndex = soup->indices[F - 1][1];

			if (startIndex != endIndex) {
				soup->indices.emplace_back(std::vector<int>{endIndex, startIndex});
			}
		}
	}

	// close
	in.close();

	return soup;
}

std::shared_ptr<PolygonSoup<3>> readLineSegmentSoupFromOBJFile(const std::string& filename,
								  std::vector<std::shared_ptr<Primitive<3>>>& lineSegments,
								  bool computeWeightedNormals, bool closeLoop)
{
	// read soup and initialize line segments
	bool isFlat = true;
	std::shared_ptr<PolygonSoup<3>> soup = readLineSegmentSoupFromOBJFile(filename,
																closeLoop, isFlat);
	int N = (int)soup->indices.size();
	lineSegments.clear();

	for (int i = 0; i < N; i++) {
		if (soup->indices[i].size() != 2) {
			LOG(FATAL) << "Soup has non line segment curves: " << filename;
		}

		lineSegments.emplace_back(std::make_shared<LineSegment>(soup, isFlat, i));
	}

	if (isFlat) {
		// swap indices if segments are oriented in clockwise order
		if (closeLoop) {
			float signedVolume = 0.0f;
			for (int i = 0; i < N; i++) {
				signedVolume += lineSegments[i]->signedVolume();
			}

			if (signedVolume < 0) {
				for (int i = 0; i < N; i++) {
					std::swap(soup->indices[i][0], soup->indices[i][1]);
				}
			}
		}

		// compute weighted normals if requested
		if (computeWeightedNormals) computeWeightedLineSegmentNormals(lineSegments, soup);
	}

	return soup;
}

} // namespace fcpw
