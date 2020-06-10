#include "triangles.h"
#include <fstream>
#include <sstream>
#include <map>

namespace fcpw {

Triangle::Triangle(const std::shared_ptr<PolygonSoup<3>>& soup_, int index_):
Primitive<3>(),
soup(soup_),
indices(soup_->indices[index_]),
eIndices(soup_->eIndices[index_]),
tIndices(soup_->tIndices[index_])
{

}

BoundingBox<3> Triangle::boundingBox() const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];
	const Vector3& pc = soup->positions[indices[2]];

	BoundingBox<3> box(pa);
	box.expandToInclude(pb);
	box.expandToInclude(pc);

	return box;
}

Vector3 Triangle::centroid() const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];
	const Vector3& pc = soup->positions[indices[2]];

	return (pa + pb + pc)/3.0f;
}

float Triangle::surfaceArea() const
{
	return 0.5f*norm<3>(normal());
}

float Triangle::signedVolume() const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];
	const Vector3& pc = soup->positions[indices[2]];

	return dot<3>(cross(pa, pb), pc)/6.0f;
}

Vector3 Triangle::normal(bool normalize) const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];
	const Vector3& pc = soup->positions[indices[2]];

	Vector3 v1 = pb - pa;
	Vector3 v2 = pc - pa;

	Vector3 n = cross(v1, v2);
	return normalize ? unit<3>(n) : n;
}

Vector3 Triangle::normal(int vIndex, int eIndex) const
{
	if (soup->vNormals.size() > 0 && vIndex >= 0) return soup->vNormals[indices[vIndex]];
	if (soup->eNormals.size() > 0 && eIndex >= 0) return soup->eNormals[eIndices[eIndex]];
	return normal(true);
}

Vector3 Triangle::normal(const Vector2& uv) const
{
	int vIndex = -1;
	if (uv[0] > oneMinusEpsilon && uv[1] < epsilon) vIndex = 0; // (1, 0, 0)
	else if (uv[0] < epsilon && uv[1] > oneMinusEpsilon) vIndex = 1; // (0, 1, 0)
	else if (uv[0] < epsilon && uv[1] < epsilon) vIndex = 2; // (0, 0, 1)

	int eIndex = -1;
	if (vIndex == -1) {
		if (uv[0] < epsilon) eIndex = 1; // (0, 1 - w, w)
		else if (uv[1] < epsilon) eIndex = 2; // (1 - w, 0, w)
		else if (uv[0] + uv[1] > oneMinusEpsilon) eIndex = 0; // (1 - v, v, 0)
	}

	return normal(vIndex, eIndex);
}

Vector2 Triangle::barycentricCoordinates(const Vector3& p) const
{
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];
	const Vector3& pc = soup->positions[indices[2]];

	Vector3 v1 = pb - pa;
	Vector3 v2 = pc - pa;
	Vector3 v3 = p - pa;

	float d11 = dot<3>(v1, v1);
	float d12 = dot<3>(v1, v2);
	float d22 = dot<3>(v2, v2);
	float d31 = dot<3>(v3, v1);
	float d32 = dot<3>(v3, v2);
	float denom = d11*d22 - d12*d12;
	float v = (d22*d31 - d12*d32)/denom;
	float w = (d11*d32 - d12*d31)/denom;

	return Vector2(1.0f - v - w, v);
}

Vector2 Triangle::textureCoordinates(const Vector2& uv) const
{
	if (tIndices.size() == 3) {
		const Vector2& pa = soup->textureCoordinates[tIndices[0]];
		const Vector2& pb = soup->textureCoordinates[tIndices[1]];
		const Vector2& pc = soup->textureCoordinates[tIndices[2]];

		float u = uv[0];
		float v = uv[1];
		float w = 1.0f - u - v;

		return pa*u + pb*v + pc*w;
	}

	return Vector2(-1, -1);
}

void Triangle::split(int dim, float splitCoord, BoundingBox<3>& boxLeft,
					 BoundingBox<3>& boxRight) const
{
	for (int i = 0; i < 3; i++) {
		const Vector3& pa = soup->positions[indices[i]];
		const Vector3& pb = soup->positions[indices[(i + 1)%3]];

		if (pa[dim] <= splitCoord && pb[dim] <= splitCoord) {
			const Vector3& pc = soup->positions[indices[(i + 2)%3]];

			if (pc[dim] <= splitCoord) {
				boxLeft = BoundingBox<3>(pa);
				boxLeft.expandToInclude(pb);
				boxLeft.expandToInclude(pc);
				boxRight = BoundingBox<3>();

			} else {
				Vector3 u = pa - pc;
				Vector3 v = pb - pc;
				float t = clamp((splitCoord - pc[dim])/u[dim], 0.0f, 1.0f);
				float s = clamp((splitCoord - pc[dim])/v[dim], 0.0f, 1.0f);

				boxLeft = BoundingBox<3>(pc + u*t);
				boxLeft.expandToInclude(pc + v*s);
				boxRight = boxLeft;
				boxLeft.expandToInclude(pa);
				boxLeft.expandToInclude(pb);
				boxRight.expandToInclude(pc);
			}

			break;

		} else if (pa[dim] >= splitCoord && pb[dim] >= splitCoord) {
			const Vector3& pc = soup->positions[indices[(i + 2)%3]];

			if (pc[dim] >= splitCoord) {
				boxRight = BoundingBox<3>(pa);
				boxRight.expandToInclude(pb);
				boxRight.expandToInclude(pc);
				boxLeft = BoundingBox<3>();

			} else {
				Vector3 u = pa - pc;
				Vector3 v = pb - pc;
				float t = clamp((splitCoord - pc[dim])/u[dim], 0.0f, 1.0f);
				float s = clamp((splitCoord - pc[dim])/v[dim], 0.0f, 1.0f);

				boxRight = BoundingBox<3>(pc + u*t);
				boxRight.expandToInclude(pc + v*s);
				boxLeft = boxRight;
				boxRight.expandToInclude(pa);
				boxRight.expandToInclude(pb);
				boxLeft.expandToInclude(pc);
			}

			break;
		}
	}
}

int Triangle::intersect(Ray<3>& r, std::vector<Interaction<3>>& is,
						bool checkOcclusion, bool countHits) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// Möller–Trumbore intersection algorithm
	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];
	const Vector3& pc = soup->positions[indices[2]];
	is.clear();

	Vector3 v1 = pb - pa;
	Vector3 v2 = pc - pa;
	Vector3 p = cross(r.d, v2);
	float det = dot<3>(v1, p);

	// ray and triangle are parallel if det is close to 0
	if (std::fabs(det) < epsilon) return 0;
	float invDet = 1.0f/det;

	Vector3 s = r.o - pa;
	float u = dot<3>(s, p)*invDet;
	if (u < 0 || u > 1) return 0;

	Vector3 q = cross(s, v1);
	float v = dot<3>(r.d, q)*invDet;
	if (v < 0 || u + v > 1) return 0;

	float t = dot<3>(v2, q)*invDet;
	if (t > epsilon && t <= r.tMax) {
		auto it = is.emplace(is.end(), Interaction<3>());
		it->d = t;
		it->p = r(t);
		it->uv[0] = u;
		it->uv[1] = v;
		it->primitive = this;

		return 1;
	}

	return 0;
}

float findClosestPointTriangle(const Vector3& pa, const Vector3& pb, const Vector3& pc,
							   const Vector3& x, Vector3& pt, Vector2& t)
{
	// source: real time collision detection
	// check if x in vertex region outside pa
	Vector3 ab = pb - pa;
	Vector3 ac = pc - pa;
	Vector3 ax = x - pa;
	float d1 = dot<3>(ab, ax);
	float d2 = dot<3>(ac, ax);
	if (d1 <= 0.0f && d2 <= 0.0f) {
		// barycentric coordinates (1, 0, 0)
		t[0] = 1.0f;
		t[1] = 0.0f;
		pt = pa;
		return norm<3>(x - pt);
	}

	// check if x in vertex region outside pb
	Vector3 bx = x - pb;
	float d3 = dot<3>(ab, bx);
	float d4 = dot<3>(ac, bx);
	if (d3 >= 0.0f && d4 <= d3) {
		// barycentric coordinates (0, 1, 0)
		t[0] = 0.0f;
		t[1] = 1.0f;
		pt = pb;
		return norm<3>(x - pt);
	}

	// check if x in edge region of ab, if so return projection of x onto ab
	float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		// barycentric coordinates (1 - v, v, 0)
		float v = d1/(d1 - d3);
		t[0] = 1.0f - v;
		t[1] = v;
		pt = pa + ab*v;
		return norm<3>(x - pt);
	}

	// check if x in vertex region outside pc
	Vector3 cx = x - pc;
	float d5 = dot<3>(ab, cx);
	float d6 = dot<3>(ac, cx);
	if (d6 >= 0.0f && d5 <= d6) {
		// barycentric coordinates (0, 0, 1)
		t[0] = 0.0f;
		t[1] = 0.0f;
		pt = pc;
		return norm<3>(x - pt);
	}

	// check if x in edge region of ac, if so return projection of x onto ac
	float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		// barycentric coordinates (1 - w, 0, w)
		float w = d2/(d2 - d6);
		t[0] = 1.0f - w;
		t[1] = 0.0f;
		pt = pa + ac*w;
		return norm<3>(x - pt);
	}

	// check if x in edge region of bc, if so return projection of x onto bc
	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		// barycentric coordinates (0, 1 - w, w)
		float w = (d4 - d3)/((d4 - d3) + (d5 - d6));
		t[0] = 0.0f;
		t[1] = 1.0f - w;
		pt = pb + (pc - pb)*w;
		return norm<3>(x - pt);
	}

	// x inside face region. Compute pt through its barycentric coordinates (u, v, w)
	float denom = 1.0f/(va + vb + vc);
	float v = vb*denom;
	float w = vc*denom;
	t[0] = 1.0f - v - w;
	t[1] = v;

	pt = pa + ab*v + ac*w; //= u*a + v*b + w*c, u = va*denom = 1.0f - v - w
	return norm<3>(x - pt);
}

bool Triangle::findClosestPoint(BoundingSphere<3>& s, Interaction<3>& i) const
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	const Vector3& pa = soup->positions[indices[0]];
	const Vector3& pb = soup->positions[indices[1]];
	const Vector3& pc = soup->positions[indices[2]];

	float d = findClosestPointTriangle(pa, pb, pc, s.c, i.p, i.uv);

	if (d*d <= s.r2) {
		i.d = d;
		i.primitive = this;

		return true;
	}

	return false;
}

void computeWeightedTriangleNormals(const std::vector<std::shared_ptr<Primitive<3>>>& triangles,
									std::shared_ptr<PolygonSoup<3>>& soup)
{
	// set edge indices
	int E = 0;
	int N = (int)soup->indices.size();
	std::map<std::pair<int, int>, int> indexMap;

	for (int i = 0; i < N; i++) {
		const std::vector<int>& index = soup->indices[i];

		for (int j = 0; j < 3; j++) {
			int k = (j + 1)%3;
			int I = index[j];
			int J = index[k];
			if (I > J) std::swap(I, J);
			std::pair<int, int> e(I, J);

			if (indexMap.find(e) == indexMap.end()) indexMap[e] = E++;
			soup->eIndices[i].emplace_back(indexMap[e]);
		}
	}

	// compute normals
	int V = (int)soup->positions.size();
	soup->vNormals.resize(V, zeroVector<3>());
	soup->eNormals.resize(E, zeroVector<3>());

	for (int i = 0; i < N; i++) {
		Vector3 n = static_cast<const Triangle *>(triangles[i].get())->normal(true);
		for (int j = 0; j < 3; j++) {
			soup->vNormals[soup->indices[i][j]] += n;
			soup->eNormals[soup->eIndices[i][j]] += n;
		}
	}

	for (int i = 0; i < V; i++) soup->vNormals[i] = unit<3>(soup->vNormals[i]);
	for (int i = 0; i < E; i++) soup->eNormals[i] = unit<3>(soup->eNormals[i]);
}

std::shared_ptr<PolygonSoup<3>> readTriangleSoupFromOBJFile(const std::string& filename)
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
	while (getline(in, line)) {
		std::stringstream ss(line);
		std::string token;
		ss >> token;

		if (token == "v") {
			float x, y, z;
			ss >> x >> y >> z;

			soup->positions.emplace_back(Vector3(x, y, z));

		} else if (token == "vt") {
			float u, v;
			ss >> u >> v;

			soup->textureCoordinates.emplace_back(Vector2(u, v));

		} else if (token == "f") {
			std::vector<int> indices, tIndices;

			while (ss >> token) {
				Index index = parseFaceIndex(token);

				if (index.position < 0) {
					getline(in, line);
					size_t i = line.find_first_not_of("\t\n\v\f\r ");
					index = parseFaceIndex(line.substr(i));
				}

				indices.emplace_back(index.position);
				tIndices.emplace_back(index.uv);
			}

			soup->indices.emplace_back(indices);
			soup->tIndices.emplace_back(tIndices);
		}
	}

	// close
	in.close();

	LOG_IF(INFO, soup->textureCoordinates.size() == 0) << "Model does not contain uvs";
	return soup;
}

std::shared_ptr<PolygonSoup<3>> readTriangleSoupFromOBJFile(const std::string& filename,
								  std::vector<std::shared_ptr<Primitive<3>>>& triangles,
								  bool computeWeightedNormals)
{
	// read soup and initialize triangles
	std::shared_ptr<PolygonSoup<3>> soup = readTriangleSoupFromOBJFile(filename);
	int N = (int)soup->indices.size();
	soup->eIndices.resize(N); // entries will be set if vertex and edge normals are requested
	triangles.clear();

	for (int i = 0; i < N; i++) {
		if (soup->indices[i].size() != 3) {
			LOG(FATAL) << "Soup has non-triangular polygons: " << filename;
		}

		triangles.emplace_back(std::make_shared<Triangle>(soup, i));
	}

	// compute weighted normals if requested
	if (computeWeightedNormals) computeWeightedTriangleNormals(triangles, soup);

	return soup;
}

} // namespace fcpw
