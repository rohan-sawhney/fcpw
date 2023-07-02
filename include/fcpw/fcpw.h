#pragma once

#include <fcpw/utilities/scene_data.h>

namespace fcpw {

enum class PrimitiveType {
	LineSegment,
	Triangle
};

template<size_t DIM>
class Scene {
public:
	// constructor
	Scene();

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// API to specify scene geometry

	// sets the PrimitiveType for each object in the scene;
	// e.g., {{LineSegment}, {Triangle}, {LineSegment, Triangle}} specifies a scene
	// with 3 objects, the 1st containing line segments, the 2nd containing triangles
	// and the 3rd containing a mix of line segments and triangles; each call
	// to this function resets the scene data
	void setObjectTypes(const std::vector<std::vector<PrimitiveType>>& objectTypes);

	// sets the number of vertices in an object
	void setObjectVertexCount(int nVertices, int objectIndex);

	// sets the number of line segments in an object
	void setObjectLineSegmentCount(int nLineSegments, int objectIndex);

	// sets the number of triangles in an object
	void setObjectTriangleCount(int nTriangles, int objectIndex);

	// sets the position of a vertex in an object
	void setObjectVertex(const Vector<DIM>& position, int vertexIndex, int objectIndex);

	// sets the vertex indices of a line segment in an object
	void setObjectLineSegment(const int *indices, int lineSegmentIndex, int objectIndex);

	// sets the vertex indices of a triangle in an object
	void setObjectTriangle(const int *indices, int triangleIndex, int objectIndex);

	// sets the vertex indices of a primitive in an object; primitiveIndex must lie in the range
	// [0, nLineSegments) or [0, nTriangles) based on the primitive type; internally, the line
	// segments are stored before the triangles; NOTE: use this function only if an object contains
	// mixed primitive types, otherwise use one of the two functions above
	void setObjectPrimitive(const int *indices, const PrimitiveType& primitiveType,
							int primitiveIndex, int objectIndex);

	// sets the instance transforms for an object
	void setObjectInstanceTransforms(const std::vector<Transform<DIM>>& transforms, int objectIndex);

	// sets the data for a node in the csg tree; NOTE: the root node of the csg tree must have index 0
	void setCsgTreeNode(const CsgTreeNode& csgTreeNode, int nodeIndex);

	// computes silhouette information for all primitives in the scene to perform closest silhouette
	// point queries; the optional ignoreSilhouette callback allows the user to specify which
	// interior vertices/edges in the line segment/triangle geometry to ignore for silhouette tests
	// (arguments: vertex/edge dihedral angle, index of an adjacent line segment/triangle)
	// NOTE: does not currently support mixed PrimitiveTypes or non-manifold geometry
	void computeSilhouettes(const std::function<bool(float, int)>& ignoreSilhouette={});

	// precomputes vertex and edge normals for an object with a single PrimitiveType; if normals are required
	// for objects with mixed primitive types, they can be computed outside of fcpw after performing
	// the query by using the "primitiveIndex" member in the "Interaction" class. NOTE: enabling normal
	// computation for non-planar line segments produces gargabe results since normals are not well defined
	void computeObjectNormals(int objectIndex, bool computeWeighted=false);

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// API to build the scene aggregate/accelerator

	// builds a (possibly vectorized) aggregate/accelerator for the scene; each call to this
	// function rebuilds the aggregate/accelerator for the scene from the specified geometry
	// (except when reduceMemoryFootprint is set to true which results in undefined behavior);
	// it is recommended to set vectorize to false for primitives that do not implement
	// vectorized intersection and closest point queries; set reduceMemoryFootprint to true
	// to reduce the memory footprint of fcpw when constructing an aggregate, however if you
	// plan to access the scene data let it remain false.
	void build(const AggregateType& aggregateType, bool vectorize,
			   bool printStats=false, bool reduceMemoryFootprint=false);

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// API to perform ray intersection and closest point queries on the scene, among others

	// intersects the scene with the given ray and returns the number of hits;
	// by default, returns the closest interaction if it exists;
	// if checkForOcclusion is enabled, the interactions vector is not populated;
	// if recordAllHits is enabled, sorts interactions by distance to the ray origin
	int intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
				  bool checkForOcclusion=false, bool recordAllHits=false) const;

	// intersects the scene with the given sphere and returns the number of primitives
	// inside the sphere; interactions contain the primitive indices; if recordOneHit
	// is set to true, randomly selects one geometric primitive inside the sphere
	// (one for each aggregate in the hierarchy) using the user specified weight function
	// (function argument is the squared distance between the sphere and primitive centers)
	// and writes the selection pdf value to Interaction<DIM>::d along with the primitive index
	int intersect(const BoundingSphere<DIM>& s,
				  std::vector<Interaction<DIM>>& is, bool recordOneHit=false,
				  const std::function<float(float)>& primitiveWeight={}) const;

	// intersects the scene with the given sphere; this method is faster than the one
	// above, since it does not visit all primitives inside the sphere during traversal--
	// the primitives visited are chosen stochastically; this method randomly selects one
	// geometric primitive inside the sphere (for each aggregate in the hierarchy) using
	// the user specified weight function (function argument is the squared distance
	// between the sphere and box/primitive centers) and samples a random point on that
	// primitive (written to Interaction<DIM>::p) using the random numbers randNums[DIM];
	// the selection pdf value is written to Interaction<DIM>::d along with the primitive index
	int intersectStochastic(const BoundingSphere<DIM>& s,
							std::vector<Interaction<DIM>>& is, float *randNums,
							const std::function<float(float)>& traversalWeight={},
							const std::function<float(float)>& primitiveWeight={}) const;

	// checks whether a point is contained inside a scene; NOTE: the scene must be watertight
	bool contains(const Vector<DIM>& x) const;

	// checks whether there is a line of sight between between two points in the scene
	bool hasLineOfSight(const Vector<DIM>& xi, const Vector<DIM>& xj) const;

	// finds the closest point in the scene to a query point; optionally specify a conservative
	// radius guess around the query point inside which the search is performed
	bool findClosestPoint(const Vector<DIM>& x, Interaction<DIM>& i,
						  float squaredRadius=maxFloat, bool recordNormal=false) const;

	// finds the closest point on the visibility silhouette in the scene to a query point;
	// optionally specify a minimum radius to stop the closest silhouette search,
	// a conservative maximum radius guess around the query point inside which the
	// search is performed, as well as a precision parameter to help classify silhouettes
	// when the query point lies on the scene geometry
	bool findClosestSilhouettePoint(const Vector<DIM>& x, Interaction<DIM>& i,
									bool flipNormalOrientation=false, float squaredMinRadius=0.0f,
									float squaredMaxRadius=maxFloat, float precision=1e-3f,
									bool recordNormal=false) const;

	// returns a pointer to the underlying scene data; use at your own risk...
	SceneData<DIM>* getSceneData();

private:
	// member
	std::unique_ptr<SceneData<DIM>> sceneData;
};

} // namespace fcpw

#include "fcpw.inl"
