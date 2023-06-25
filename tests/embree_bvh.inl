#include <common/math/vec2.h>
#include <common/math/vec3.h>

namespace fcpw {

void errorFunction(void *userPtr, enum RTCError error, const char *str)
{
	if (error == RTC_ERROR_NONE) return;

	std::string code = "";
	switch (error) {
		case RTC_ERROR_UNKNOWN          : code = "RTC_ERROR_UNKNOWN"; break;
		case RTC_ERROR_INVALID_ARGUMENT : code = "RTC_ERROR_INVALID_ARGUMENT"; break;
		case RTC_ERROR_INVALID_OPERATION: code = "RTC_ERROR_INVALID_OPERATION"; break;
		case RTC_ERROR_OUT_OF_MEMORY    : code = "RTC_ERROR_OUT_OF_MEMORY"; break;
		case RTC_ERROR_UNSUPPORTED_CPU  : code = "RTC_ERROR_UNSUPPORTED_CPU"; break;
		case RTC_ERROR_CANCELLED        : code = "RTC_ERROR_CANCELLED"; break;
		default                         : code = "invalid error code"; break;
	}

	std::cerr << "Embree error code: " << code << " msg: " << str << std::endl;
	exit(EXIT_FAILURE);
}

struct IntersectContext {
	// constructor
	IntersectContext(std::vector<Interaction<3>>& is_): is(is_) {}

	// members
	RTCIntersectContext context;
	std::vector<Interaction<3>>& is;
};

void triangleIntersectionCallback(const struct RTCFilterFunctionNArguments *args)
{
	// get required information from args
	RTCRay *ray = (RTCRay *)args->ray;
	RTCHit *hit = (RTCHit *)args->hit;
	IntersectContext *context = (IntersectContext *)args->context;
	std::vector<Interaction<3>>& is = context->is;
	args->valid[0] = 0; // ignore all hits

	// add interaction
	auto it = is.emplace(is.end(), Interaction<3>());
	it->d = ray->tfar;
	it->p = Vector3(ray->org_x, ray->org_y, ray->org_z) +
			Vector3(ray->dir_x, ray->dir_y, ray->dir_z)*it->d;
	it->uv[0] = hit->u;
	it->uv[1] = hit->v;
	it->primitiveIndex = hit->primID;
}

embree::Vec3fa closestPointTriangle(embree::Vec3fa const& p, embree::Vec3fa const& a,
									embree::Vec3fa const& b, embree::Vec3fa const& c)
{
	const embree::Vec3fa ab = b - a;
	const embree::Vec3fa ac = c - a;
	const embree::Vec3fa ap = p - a;

	const float d1 = dot(ab, ap);
	const float d2 = dot(ac, ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a;

	const embree::Vec3fa bp = p - b;
	const float d3 = dot(ab, bp);
	const float d4 = dot(ac, bp);
	if (d3 >= 0.0f && d4 <= d3) return b;

	const embree::Vec3fa cp = p - c;
	const float d5 = dot(ab, cp);
	const float d6 = dot(ac, cp);
	if (d6 >= 0.0f && d5 <= d6) return c;

	const float vc = d1*d4 - d3*d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		const float v = d1/(d1 - d3);
		return a + v*ab;
	}

	const float vb = d5*d2 - d1*d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		const float v = d2/(d2 - d6);
		return a + v*ac;
	}

	const float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		const float v = (d4 - d3)/((d4 - d3) + (d5 - d6));
		return b + v*(c - b);
	}

	const float denom = 1.0f/(va + vb + vc);
	const float v = vb*denom;
	const float w = vc*denom;
	return a + v*ab + w*ac;
}

struct ClosestPointResult {
	// constructor
	ClosestPointResult(): primID(RTC_INVALID_GEOMETRY_ID), geomID(RTC_INVALID_GEOMETRY_ID) {}

	// members
	embree::Vec3fa p;
	unsigned int primID;
	unsigned int geomID;
};

// NOTE: this global variable is created with great sadness; alternatives
// such as declaring closestPointTriangleCallback as a member function of Embree Bvh
// and using that function as the callback, or using std::bind to wrap the soup that
// needs to be passed to closestPointTriangleCallback don't work since their function
// signatures don't match those of the callback.
static const PolygonSoup<3> *callbackSoup = nullptr;

bool closestPointTriangleCallback(RTCPointQueryFunctionArguments *args)
{
	// get required information from args
	const unsigned int primID = args->primID;
	const unsigned int geomID = args->geomID;
	embree::Vec3fa q(args->query->x, args->query->y, args->query->z);

	// determine distance to closest point on triangle
	const Vector3& pa = callbackSoup->positions[callbackSoup->indices[3*primID]];
	const Vector3& pb = callbackSoup->positions[callbackSoup->indices[3*primID + 1]];
	const Vector3& pc = callbackSoup->positions[callbackSoup->indices[3*primID + 2]];
	embree::Vec3fa v1(pa[0], pa[1], pa[2]);
	embree::Vec3fa v2(pb[0], pb[1], pb[2]);
	embree::Vec3fa v3(pc[0], pc[1], pc[2]);

	const embree::Vec3fa p = closestPointTriangle(q, v1, v2, v3);
	float d = distance(q, p);

	// store result in userPtr and update the query radius if we found a point
	// closer to the query position
	if (d < args->query->radius) {
		args->query->radius = d;
		ClosestPointResult *result = (ClosestPointResult *)args->userPtr;
		result->p = p;
		result->primID = primID;
		result->geomID = geomID;

		return true;
	}

	return false;
}

inline EmbreeBvh::EmbreeBvh(const std::vector<Triangle *>& triangles_,
							const PolygonSoup<3> *soup_, bool printStats_):
Baseline<3, Triangle>(triangles_)
{
	using namespace std::chrono;
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	// initialize device
	device = rtcNewDevice(NULL); // specify flags e.g. threads, isa, verbose, tri_accel=bvh4.triangle4v if required
	if (!device) {
		std::cerr << "EmbreeBvh(): Unable to create device: " << rtcGetDeviceError(NULL) << std::endl;
		exit(EXIT_FAILURE);
	}

	// register error callback
	rtcSetDeviceErrorFunction(device, errorFunction, NULL);

	// initialize scene
	scene = rtcNewScene(device);
	rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
	rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

	// initialize geometry; NOTE: specialized to triangle meshes for now
	RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);

	// register closest point callback
	callbackSoup = soup_;
	rtcSetGeometryPointQueryFunction(geometry, closestPointTriangleCallback);

	// load geometry
	float *vertices = (float *)rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0,
													   RTC_FORMAT_FLOAT3, 3*sizeof(float),
													   soup_->positions.size());
	unsigned int *indices = (unsigned int *)rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0,
																	RTC_FORMAT_UINT3, 3*sizeof(unsigned int),
																	soup_->indices.size()/3);

	if (vertices && indices) {
		for (int i = 0; i < (int)soup_->positions.size(); i++) {
			for (int j = 0; j < 3; j++) {
				vertices[3*i + j] = soup_->positions[i][j];
			}
		}

		for (int i = 0; i < (int)soup_->indices.size(); i++) {
			indices[i] = soup_->indices[i];
		}
	}

	// commit, attach and release geometry
	rtcCommitGeometry(geometry);
	rtcAttachGeometry(scene, geometry);
	rtcReleaseGeometry(geometry);

	// commit scene
	rtcCommitScene(scene);

	// print stats
	if (printStats_) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
		std::cout << "Built Embree Bvh with "
				  << this->primitives.size() << " triangles in "
				  << timeSpan.count() << " seconds" << std::endl;
	}
}

inline EmbreeBvh::~EmbreeBvh()
{
	rtcReleaseScene(scene);
	rtcReleaseDevice(device);
}

inline BoundingBox<3> EmbreeBvh::boundingBox() const
{
	return Baseline<3, Triangle>::boundingBox();
}

inline Vector3 EmbreeBvh::centroid() const
{
	return Baseline<3, Triangle>::centroid();
}

inline float EmbreeBvh::surfaceArea() const
{
	return Baseline<3, Triangle>::surfaceArea();
}

inline float EmbreeBvh::signedVolume() const
{
	return Baseline<3, Triangle>::signedVolume();
}

inline int EmbreeBvh::intersectFromNode(Ray<3>& r, std::vector<Interaction<3>>& is,
										int nodeStartIndex, int aggregateIndex, int& nodesVisited,
										bool checkForOcclusion, bool recordAllHits) const
{
	// initialize intersect context (RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT is enabled by default)
	IntersectContext context(is);
	rtcInitIntersectContext(&context.context);
	nodesVisited++;

	// initialize rayhit structure
	RTCRayHit rayhit;
	rayhit.ray.org_x = r.o[0];
	rayhit.ray.org_y = r.o[1];
	rayhit.ray.org_z = r.o[2];
	rayhit.ray.dir_x = r.d[0];
	rayhit.ray.dir_y = r.d[1];
	rayhit.ray.dir_z = r.d[2];
	rayhit.ray.tnear = 0.0f;
	rayhit.ray.tfar = r.tMax;
	rayhit.ray.mask = 0;
	rayhit.ray.flags = 0;
	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	is.clear();

	if (checkForOcclusion) {
		// test for occlusion
		rtcOccluded1(scene, &context.context, &rayhit.ray);
		return rayhit.ray.tfar >= 0.0f ? 0 : 1;
	}

	// set filter function to collect all hits if requested
	if (recordAllHits) context.context.filter = triangleIntersectionCallback;

	// intersect single ray with the scene
	rtcIntersect1(scene, &context.context, &rayhit);
	int hits = 0;

	if (is.size() > 0) {
		// sort interactions
		std::sort(is.begin(), is.end(), compareInteractions<3>);
		is = removeDuplicates<3>(is);
		hits = (int)is.size();

	} else {
		if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
			// record closest interaction
			auto it = is.emplace(is.end(), Interaction<3>());
			it->d = rayhit.ray.tfar;
			it->p = r(it->d);
			it->uv[0] = rayhit.hit.u;
			it->uv[1] = rayhit.hit.v;
			it->primitiveIndex = rayhit.hit.primID;
			r.tMax = it->d;
			hits = 1;
		}
	}

	// compute normals and set aggregate index
	for (int i = 0; i < (int)is.size(); i++) {
		is[i].computeNormal(this->primitives[is[i].primitiveIndex]);
		is[i].objectIndex = this->index;
	}

	return hits;
}

inline bool EmbreeBvh::findClosestPointFromNode(BoundingSphere<3>& s, Interaction<3>& i,
												int nodeStartIndex, int aggregateIndex,
												int& nodesVisited, bool recordNormal) const
{
	// initialize closest point context
	RTCPointQueryContext context;
	rtcInitPointQueryContext(&context);
	nodesVisited++;

	// initialize point query
	RTCPointQuery query;
	query.x = s.c[0];
	query.y = s.c[1];
	query.z = s.c[2];
	query.radius = std::sqrt(s.r2);
	query.time = 0.0f;

	// perform query
	ClosestPointResult result;
	rtcPointQuery(scene, &query, &context, nullptr, (void*)&result);

	if (result.geomID != RTC_INVALID_GEOMETRY_ID) {
		// record result
		i.p[0] = result.p.x;
		i.p[1] = result.p.y;
		i.p[2] = result.p.z;
		i.d = (i.p - s.c).norm();
		const Triangle *triangle = this->primitives[result.primID];
		i.uv = triangle->barycentricCoordinates(i.p);
		if (recordNormal) i.computeNormal(triangle);
		i.primitiveIndex = result.primID;
		i.objectIndex = this->index;
		s.r2 = i.d*i.d;

		return true;
	}

	return false;
}

} // namespace fcpw
