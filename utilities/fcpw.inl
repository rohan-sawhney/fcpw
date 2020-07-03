#include "accelerators/baseline.h"
#include "accelerators/sbvh.h"
#ifdef BUILD_ENOKI
	#include "accelerators/mbvh.h"
#endif
#include <map>

namespace fcpw {

template<size_t DIM>
inline Scene<DIM>::Scene():
sceneData(new SceneData<DIM>())
{

}

template<size_t DIM>
inline void Scene<DIM>::setObjectTypes(const std::vector<std::vector<PrimitiveType>>& objectTypes)
{
	// clear old data
	sceneData->clearAggregateData();
	sceneData->clearObjectData();

	// initialize soup and object vectors
	int nObjects = (int)objectTypes.size();
	int nLineSegmentObjects = 0;
	int nTriangleObjects = 0;
	sceneData->soups.resize(nObjects);
	sceneData->instanceTransforms.resize(nObjects);

	for (int i = 0; i < nObjects; i++) {
		for (int j = 0; j < (int)objectTypes[i].size(); j++) {
			if (objectTypes[i][j] == PrimitiveType::LineSegment) {
				sceneData->soupToObjectsMap[i].emplace_back(std::make_pair(ObjectType::LineSegments,
																		   nLineSegmentObjects));
				nLineSegmentObjects++;

			} else if (objectTypes[i][j] == PrimitiveType::Triangle) {
				sceneData->soupToObjectsMap[i].emplace_back(std::make_pair(ObjectType::Triangles,
																		   nTriangleObjects));
				nTriangleObjects++;
			}
		}
	}

	sceneData->lineSegmentObjects.resize(nLineSegmentObjects);
	sceneData->triangleObjects.resize(nTriangleObjects);
}

template<size_t DIM>
inline void Scene<DIM>::setObjectVertexCount(int nVertices, int objectIndex)
{
	sceneData->soups[objectIndex].positions.resize(nVertices);
}

template<size_t DIM>
inline void Scene<DIM>::setObjectLineSegmentCount(int nLineSegments, int objectIndex)
{
	// resize soup indices
	PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];
	int nIndices = (int)soup.indices.size();
	soup.indices.resize(nIndices + 2*nLineSegments);

	// allocate line segments
	const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[objectIndex];
	for (int i = 0; i < (int)objectsMap.size(); i++) {
		if (objectsMap[i].first == ObjectType::LineSegments) {
			int lineSegmentObjectIndex = objectsMap[i].second;
			sceneData->lineSegmentObjects[lineSegmentObjectIndex] =
					std::unique_ptr<std::vector<LineSegment>>(new std::vector<LineSegment>(nLineSegments));
			break;
		}
	}
}

template<size_t DIM>
inline void Scene<DIM>::setObjectTriangleCount(int nTriangles, int objectIndex)
{
	// resize soup indices
	PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];
	int nIndices = (int)soup.indices.size();
	soup.indices.resize(nIndices + 3*nTriangles);

	// allocate triangles
	const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[objectIndex];
	for (int i = 0; i < (int)objectsMap.size(); i++) {
		if (objectsMap[i].first == ObjectType::Triangles) {
			int triangleObjectIndex = objectsMap[i].second;
			sceneData->triangleObjects[triangleObjectIndex] =
					std::unique_ptr<std::vector<Triangle>>(new std::vector<Triangle>(nTriangles));
			break;
		}
	}
}

template<size_t DIM>
inline void Scene<DIM>::setObjectVertex(const Vector<DIM>& position, int vertexIndex, int objectIndex)
{
	sceneData->soups[objectIndex].positions[vertexIndex] = position;
}

template<size_t DIM>
inline void Scene<DIM>::setObjectLineSegment(const std::vector<int>& indices, int lineSegmentIndex, int objectIndex)
{
	// update soup indices
	PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];
	soup.indices[2*lineSegmentIndex + 0] = indices[0];
	soup.indices[2*lineSegmentIndex + 1] = indices[1];

	// update line segment indices
	int lineSegmentObjectIndex = sceneData->soupToObjectsMap[objectIndex][0].second;
	LineSegment& lineSegment = (*sceneData->lineSegmentObjects[lineSegmentObjectIndex])[lineSegmentIndex];
	lineSegment.soup = &soup;
	lineSegment.indices[0] = indices[0];
	lineSegment.indices[1] = indices[1];
	lineSegment.pIndex = lineSegmentIndex;
}

template<size_t DIM>
inline void Scene<DIM>::setObjectTriangle(const std::vector<int>& indices, int triangleIndex, int objectIndex)
{
	// update soup indices
	PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];
	soup.indices[3*triangleIndex + 0] = indices[0];
	soup.indices[3*triangleIndex + 1] = indices[1];
	soup.indices[3*triangleIndex + 2] = indices[2];

	// update triangle indices
	int triangleObjectIndex = sceneData->soupToObjectsMap[objectIndex][0].second;
	Triangle& triangle = (*sceneData->triangleObjects[triangleObjectIndex])[triangleIndex];
	triangle.soup = &soup;
	triangle.indices[0] = indices[0];
	triangle.indices[1] = indices[1];
	triangle.indices[2] = indices[2];
	triangle.pIndex = triangleIndex;
}

template<size_t DIM>
inline void Scene<DIM>::setObjectPrimitive(const std::vector<int>& indices, const PrimitiveType& primitiveType,
										   int primitiveIndex, int objectIndex)
{
	// count line segments and triangles
	const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[objectIndex];
	PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];
	int lineSegmentObjectIndex = -1;
	int nLineSegments = 0;
	int triangleObjectIndex = -1;
	int nTriangles = 0;

	for (int i = 0; i < (int)objectsMap.size(); i++) {
		if (objectsMap[i].first == ObjectType::LineSegments) {
			lineSegmentObjectIndex = objectsMap[i].second;
			nLineSegments += (int)sceneData->lineSegmentObjects[lineSegmentObjectIndex]->size();

		} else if (objectsMap[i].first == ObjectType::Triangles) {
			triangleObjectIndex = objectsMap[i].second;
			nTriangles += (int)sceneData->triangleObjects[triangleObjectIndex]->size();
		}
	}

	// update indices
	if (primitiveType == PrimitiveType::LineSegment) {
		soup.indices[2*primitiveIndex + 0] = indices[0];
		soup.indices[2*primitiveIndex + 1] = indices[1];

		LineSegment& lineSegment = (*sceneData->lineSegmentObjects[lineSegmentObjectIndex])[primitiveIndex];
		lineSegment.soup = &soup;
		lineSegment.indices[0] = indices[0];
		lineSegment.indices[1] = indices[1];
		lineSegment.pIndex = primitiveIndex;

	} else if (primitiveType == PrimitiveType::Triangle) {
		int offset = 2*nLineSegments;
		soup.indices[offset + 3*primitiveIndex + 0] = indices[0];
		soup.indices[offset + 3*primitiveIndex + 1] = indices[1];
		soup.indices[offset + 3*primitiveIndex + 2] = indices[2];

		Triangle& triangle = (*sceneData->triangleObjects[triangleObjectIndex])[primitiveIndex];
		triangle.soup = &soup;
		triangle.indices[0] = indices[0];
		triangle.indices[1] = indices[1];
		triangle.indices[2] = indices[2];
		triangle.pIndex = nLineSegments + primitiveIndex;
	}
}

template<size_t DIM>
inline void Scene<DIM>::setObjectInstanceTransforms(const std::vector<Transform<DIM>>& transforms, int objectIndex)
{
	std::vector<Transform<DIM>>& objectTransforms = sceneData->instanceTransforms[objectIndex];
	objectTransforms.insert(objectTransforms.end(), transforms.begin(), transforms.end());
}

template<size_t DIM>
inline void Scene<DIM>::setCsgTreeNode(const CsgTreeNode& csgTreeNode, int nodeIndex)
{
	sceneData->csgTree[nodeIndex] = csgTreeNode;
}

template<size_t DIM, typename PrimitiveType>
inline void computeWeightedNormals(const std::vector<PrimitiveType>& primitives, PolygonSoup<DIM>& soup)
{
	// do nothing
}

template<>
inline void computeWeightedNormals<3, LineSegment>(const std::vector<LineSegment>& lineSegments,
												   PolygonSoup<3>& soup)
{
	int N = (int)lineSegments.size();
	int V = (int)soup.positions.size();
	soup.vNormals.resize(V, zeroVector<3>());

	for (int i = 0; i < N; i++) {
		Vector3 n = lineSegments[i].normal(true);
		soup.vNormals[lineSegments[i].indices[0]] += n;
		soup.vNormals[lineSegments[i].indices[1]] += n;
	}

	for (int i = 0; i < V; i++) {
		soup.vNormals[i] = unit<3>(soup.vNormals[i]);
	}
}

template<>
inline void computeWeightedNormals<3, Triangle>(const std::vector<Triangle>& triangles,
												PolygonSoup<3>& soup)
{
	// set edge indices
	int E = 0;
	int N = (int)triangles.size();
	std::map<std::pair<int, int>, int> indexMap;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < 3; j++) {
			int k = (j + 1)%3;
			int I = triangles[i].indices[j];
			int J = triangles[i].indices[k];
			if (I > J) std::swap(I, J);
			std::pair<int, int> e(I, J);

			if (indexMap.find(e) == indexMap.end()) indexMap[e] = E++;
			soup.eIndices.emplace_back(indexMap[e]);
		}
	}

	// compute normals
	int V = (int)soup.positions.size();
	soup.vNormals.resize(V, zeroVector<3>());
	soup.eNormals.resize(E, zeroVector<3>());

	for (int i = 0; i < N; i++) {
		Vector3 n = triangles[i].normal(true);

		for (int j = 0; j < 3; j++) {
			soup.vNormals[triangles[i].indices[j]] += n;
			soup.eNormals[soup.eIndices[3*triangles[i].pIndex + j]] += n;
		}
	}

	for (int i = 0; i < V; i++) soup.vNormals[i] = unit<3>(soup.vNormals[i]);
	for (int i = 0; i < E; i++) soup.eNormals[i] = unit<3>(soup.eNormals[i]);
}

template<size_t DIM>
inline void Scene<DIM>::computeObjectNormals(int objectIndex)
{
	const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[objectIndex];
	PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];

	if (objectsMap[0].first == ObjectType::LineSegments) {
		int lineSegmentObjectIndex = objectsMap[0].second;
		computeWeightedNormals<DIM, LineSegment>(*sceneData->lineSegmentObjects[lineSegmentObjectIndex], soup);

	} else if (objectsMap[0].first == ObjectType::Triangles) {
		int triangleObjectIndex = objectsMap[0].second;
		computeWeightedNormals<DIM, Triangle>(*sceneData->triangleObjects[triangleObjectIndex], soup);
	}
}

template<size_t DIM, typename PrimitiveType>
inline void sortSoupPositions(const std::vector<SbvhNode<DIM>>& flatTree,
							  std::vector<PrimitiveType *>& primitives,
							  PolygonSoup<DIM>& soup)
{
	// do nothing
}

template<>
inline void sortSoupPositions<3, LineSegment>(const std::vector<SbvhNode<3>>& flatTree,
											  std::vector<LineSegment *>& lineSegments,
											  PolygonSoup<3>& soup)
{
	int V = (int)soup.positions.size();
	std::vector<Vector<3>> sortedPositions(V), sortedVertexNormals(V);
	std::vector<int> indexMap(V, -1);
	int v = 0;

	// collect sorted positions, updating line segment and soup indices
	for (int i = 0; i < (int)flatTree.size(); i++) {
		const SbvhNode<3>& node(flatTree[i]);

		for (int j = 0; j < node.nReferences; j++) { // leaf node if nReferences > 0
			int referenceIndex = node.referenceOffset + j;
			LineSegment *lineSegment = lineSegments[referenceIndex];

			for (int k = 0; k < 2; k++) {
				int vIndex = lineSegment->indices[k];

				if (indexMap[vIndex] == -1) {
					sortedPositions[v] = soup.positions[vIndex];
					if (soup.vNormals.size() > 0) sortedVertexNormals[v] = soup.vNormals[vIndex];
					indexMap[vIndex] = v++;
				}

				soup.indices[2*lineSegment->pIndex + k] = indexMap[vIndex];
				lineSegment->indices[k] = indexMap[vIndex];
			}
		}
	}

	// update to sorted positions
	soup.positions = std::move(sortedPositions);
	if (soup.vNormals.size() > 0) soup.vNormals = std::move(sortedVertexNormals);
}

template<>
inline void sortSoupPositions<3, Triangle>(const std::vector<SbvhNode<3>>& flatTree,
										   std::vector<Triangle *>& triangles,
										   PolygonSoup<3>& soup)
{
	int V = (int)soup.positions.size();
	std::vector<Vector<3>> sortedPositions(V), sortedVertexNormals(V);
	std::vector<int> indexMap(V, -1);
	int v = 0;

	// collect sorted positions, updating triangle and soup indices
	for (int i = 0; i < (int)flatTree.size(); i++) {
		const SbvhNode<3>& node(flatTree[i]);

		for (int j = 0; j < node.nReferences; j++) { // leaf node if nReferences > 0
			int referenceIndex = node.referenceOffset + j;
			Triangle *triangle = triangles[referenceIndex];

			for (int k = 0; k < 3; k++) {
				int vIndex = triangle->indices[k];

				if (indexMap[vIndex] == -1) {
					sortedPositions[v] = soup.positions[vIndex];
					if (soup.vNormals.size() > 0) sortedVertexNormals[v] = soup.vNormals[vIndex];
					indexMap[vIndex] = v++;
				}

				soup.indices[3*triangle->pIndex + k] = indexMap[vIndex];
				triangle->indices[k] = indexMap[vIndex];
			}
		}
	}

	// update to sorted positions
	soup.positions = std::move(sortedPositions);
	if (soup.vNormals.size() > 0) soup.vNormals = std::move(sortedVertexNormals);
}

// TODO: going to get errors if DIM & PrimitiveType are incompatible
template<size_t DIM, typename PrimitiveType>
inline std::unique_ptr<Aggregate<DIM>> makeAggregate(const AggregateType& aggregateType,
													 std::vector<PrimitiveType *>& primitives,
													 bool vectorize, bool printStats,
													 SortPositionsFunc<DIM, PrimitiveType> sortPositions={})
{
	std::unique_ptr<Sbvh<DIM, PrimitiveType>> sbvh = nullptr;
	bool packLeaves = false;
	int leafSize = 4;

#ifdef BUILD_ENOKI
	if (vectorize) {
		packLeaves = true;
		leafSize = SIMD_WIDTH;
	}
#endif

	if (aggregateType == AggregateType::Bvh_LongestAxisCenter) {
		sbvh = std::unique_ptr<Sbvh<DIM, PrimitiveType>>(new Sbvh<DIM, PrimitiveType>(
				CostHeuristic::LongestAxisCenter, primitives, sortPositions, printStats, false, leafSize));

	} else if (aggregateType == AggregateType::Bvh_SurfaceArea) {
		sbvh = std::unique_ptr<Sbvh<DIM, PrimitiveType>>(new Sbvh<DIM, PrimitiveType>(
				CostHeuristic::SurfaceArea, primitives, sortPositions, printStats, packLeaves, leafSize));

	} else if (aggregateType == AggregateType::Bvh_OverlapSurfaceArea) {
		sbvh = std::unique_ptr<Sbvh<DIM, PrimitiveType>>(new Sbvh<DIM, PrimitiveType>(
				CostHeuristic::OverlapSurfaceArea, primitives, sortPositions, printStats, packLeaves, leafSize));

	} else if (aggregateType == AggregateType::Bvh_Volume) {
		sbvh = std::unique_ptr<Sbvh<DIM, PrimitiveType>>(new Sbvh<DIM, PrimitiveType>(
				CostHeuristic::Volume, primitives, sortPositions, printStats, packLeaves, leafSize));

	} else if (aggregateType == AggregateType::Bvh_OverlapVolume) {
		sbvh = std::unique_ptr<Sbvh<DIM, PrimitiveType>>(new Sbvh<DIM, PrimitiveType>(
				CostHeuristic::OverlapVolume, primitives, sortPositions, printStats, packLeaves, leafSize));

	} else {
		return std::unique_ptr<Baseline<DIM, PrimitiveType>>(new Baseline<DIM, PrimitiveType>(primitives));
	}

#ifdef BUILD_ENOKI
	if (vectorize) {
		return std::unique_ptr<Mbvh<SIMD_WIDTH, DIM, PrimitiveType>>(
				new Mbvh<SIMD_WIDTH, DIM, PrimitiveType>(sbvh.get(), printStats));
	}
#endif

	return sbvh;
}

template<size_t DIM>
inline std::unique_ptr<Aggregate<DIM>> buildCsgAggregateRecursive(
										int nodeIndex, std::unordered_map<int, CsgTreeNode>& csgTree,
										std::vector<std::unique_ptr<Aggregate<DIM>>>& aggregateInstances,
										int& nAggregates)
{
	const CsgTreeNode& node = csgTree[nodeIndex];
	std::unique_ptr<Aggregate<DIM>> instance1 = nullptr;
	std::unique_ptr<Aggregate<DIM>> instance2 = nullptr;

	if (node.isLeafChild1) {
		instance1 = std::move(aggregateInstances[node.child1]);

	} else {
		instance1 = buildCsgAggregateRecursive<DIM>(node.child1, csgTree, aggregateInstances, nAggregates);
		instance1->index = nAggregates++;
	}

	if (node.isLeafChild2) {
		instance2 = std::move(aggregateInstances[node.child2]);

	} else {
		instance2 = buildCsgAggregateRecursive<DIM>(node.child2, csgTree, aggregateInstances, nAggregates);
		instance2->index = nAggregates++;
	}

	return std::unique_ptr<CsgNode<DIM, Aggregate<DIM>, Aggregate<DIM>>>(
		new CsgNode<DIM, Aggregate<DIM>, Aggregate<DIM>>(std::move(instance1), std::move(instance2), node.operation));
}

template<size_t DIM>
inline void Scene<DIM>::build(const AggregateType& aggregateType, bool vectorize, bool printStats)
{
	// clear old aggregate data
	sceneData->clearAggregateData();

	// allocate space for line segment, triangle & mixed object ptrs
	int nObjects = (int)sceneData->soups.size();
	int nLineSegmentObjectPtrs = 0;
	int nTriangleObjectPtrs = 0;
	int nMixedObjectPtrs = 0;

	for (int i = 0; i < nObjects; i++) {
		const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[i];

		if (objectsMap.size() > 1) nMixedObjectPtrs++;
		else if (objectsMap[0].first == ObjectType::LineSegments) nLineSegmentObjectPtrs++;
		else if (objectsMap[0].first == ObjectType::Triangles) nTriangleObjectPtrs++;
	}

	sceneData->lineSegmentObjectPtrs.resize(nLineSegmentObjectPtrs);
	sceneData->triangleObjectPtrs.resize(nTriangleObjectPtrs);
	sceneData->mixedObjectPtrs.resize(nMixedObjectPtrs);

	// populate line segment, triangle & mixed object ptrs, and make their aggregates
	nLineSegmentObjectPtrs = 0;
	nTriangleObjectPtrs = 0;
	nMixedObjectPtrs = 0;
	int nAggregates = 0;
	std::vector<std::unique_ptr<Aggregate<DIM>>> objectAggregates(nObjects);
	using SortLineSegmentPositionsFunc = std::function<void(const std::vector<SbvhNode<DIM>>&, std::vector<LineSegment *>&)>;
	using SortTrianglePositionsFunc = std::function<void(const std::vector<SbvhNode<DIM>>&, std::vector<Triangle *>&)>;

	for (int i = 0; i < nObjects; i++) {
		const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[i];

		if (objectsMap.size() > 1) {
			// TODO: might be problematic for DIM != 3, compile time errors maybe?
			// soup contains mixed primitives, set mixed object ptrs
			std::vector<GeometricPrimitive<DIM> *>& mixedObjectPtr = sceneData->mixedObjectPtrs[nMixedObjectPtrs];

			for (int j = 0; j < (int)objectsMap.size(); j++) {
				if (objectsMap[j].first == ObjectType::LineSegments) {
					int lineSegmentObjectIndex = objectsMap[j].second;
					std::vector<LineSegment>& lineSegmentObject = *sceneData->lineSegmentObjects[lineSegmentObjectIndex];

					for (int k = 0; k < (int)lineSegmentObject.size(); k++) {
						mixedObjectPtr.emplace_back(&lineSegmentObject[k]);
					}

				} else if (objectsMap[j].first == ObjectType::Triangles) {
					int triangleObjectIndex = objectsMap[j].second;
					std::vector<Triangle>& triangleObject = *sceneData->triangleObjects[triangleObjectIndex];

					for (int k = 0; k < (int)triangleObject.size(); k++) {
						mixedObjectPtr.emplace_back(&triangleObject[k]);
					}
				}
			}

			objectAggregates[i] = makeAggregate<DIM, GeometricPrimitive<DIM>>(aggregateType, mixedObjectPtr,
																			  vectorize, printStats);
			nMixedObjectPtrs++;

		} else if (objectsMap[0].first == ObjectType::LineSegments) {
			// soup contains line segments, set line segment object ptrs
			int lineSegmentObjectIndex = objectsMap[0].second;
			std::vector<LineSegment>& lineSegmentObject = *sceneData->lineSegmentObjects[lineSegmentObjectIndex];
			std::vector<LineSegment *>& lineSegmentObjectPtr = sceneData->lineSegmentObjectPtrs[nLineSegmentObjectPtrs];

			for (int j = 0; j < (int)lineSegmentObject.size(); j++) {
				lineSegmentObjectPtr.emplace_back(&lineSegmentObject[j]);
			}

			// make aggregate
			SortLineSegmentPositionsFunc sortLineSegmentPositions = {};
			if (!vectorize) {
				sortLineSegmentPositions = std::bind(&sortSoupPositions<DIM, LineSegment>,
													 std::placeholders::_1, std::placeholders::_2,
													 std::ref(sceneData->soups[i]));
			}

			objectAggregates[i] = makeAggregate<DIM, LineSegment>(aggregateType, lineSegmentObjectPtr, vectorize,
																  printStats, sortLineSegmentPositions);
			nLineSegmentObjectPtrs++;

		} else if (objectsMap[0].first == ObjectType::Triangles) {
			// soup contains triangles, set triangle object ptrs
			int triangleObjectIndex = objectsMap[0].second;
			std::vector<Triangle>& triangleObject = *sceneData->triangleObjects[triangleObjectIndex];
			std::vector<Triangle *>& triangleObjectPtr = sceneData->triangleObjectPtrs[nTriangleObjectPtrs];

			for (int j = 0; j < (int)triangleObject.size(); j++) {
				triangleObjectPtr.emplace_back(&triangleObject[j]);
			}

			// make aggregate
			SortTrianglePositionsFunc sortTrianglePositions = {};
			if (!vectorize) {
				sortTrianglePositions = std::bind(&sortSoupPositions<DIM, Triangle>,
												  std::placeholders::_1, std::placeholders::_2,
												  std::ref(sceneData->soups[i]));
			}

			objectAggregates[i] = makeAggregate<DIM, Triangle>(aggregateType, triangleObjectPtr, vectorize,
															   printStats, sortTrianglePositions);
			nTriangleObjectPtrs++;
		}

		objectAggregates[i]->index = nAggregates++;
		objectAggregates[i]->computeNormals = sceneData->soups[i].vNormals.size() > 0;
	}

	// set aggregate instances and instance ptrs
	for (int i = 0; i < nObjects; i++) {
		int nObjectInstances = (int)sceneData->instanceTransforms[i].size();

		if (nObjectInstances == 0) {
			sceneData->aggregateInstancePtrs.emplace_back(objectAggregates[i].get());
			sceneData->aggregateInstances.emplace_back(std::move(objectAggregates[i]));

		} else {
			std::shared_ptr<Aggregate<DIM>> aggregate = std::move(objectAggregates[i]);
			for (int j = 0; j < nObjectInstances; j++) {
				std::unique_ptr<TransformedAggregate<DIM>> transformedAggregate(
					new TransformedAggregate<DIM>(aggregate, sceneData->instanceTransforms[i][j]));
				transformedAggregate->index = nAggregates++;

				sceneData->aggregateInstancePtrs.emplace_back(transformedAggregate.get());
				sceneData->aggregateInstances.emplace_back(std::move(transformedAggregate));
			}
		}
	}

	// set aggregate
	if (sceneData->aggregateInstances.size() == 1) {
		// clear the vectors of aggregate instances if there is only a single aggregate
		sceneData->aggregate = std::move(sceneData->aggregateInstances[0]);
		sceneData->aggregateInstancePtrs.clear();
		sceneData->aggregateInstances.clear();

	} else if (sceneData->csgTree.size() > 0) {
		// build csg tree
		sceneData->aggregate = buildCsgAggregateRecursive<DIM>(0, sceneData->csgTree,
															   sceneData->aggregateInstances, nAggregates);
		sceneData->aggregate->index = nAggregates++;
		sceneData->aggregateInstancePtrs.clear();
		sceneData->aggregateInstances.clear();

	} else {
		// make aggregate
		sceneData->aggregate = makeAggregate<DIM, Aggregate<DIM>>(aggregateType, sceneData->aggregateInstancePtrs,
																  false, printStats);
		sceneData->aggregate->index = nAggregates++;
	}
}

template<size_t DIM>
inline int Scene<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is, bool checkForOcclusion,
								 bool recordAllHits) const
{
	return sceneData->aggregate->intersect(r, is, checkForOcclusion, recordAllHits);
}

template<size_t DIM>
inline bool Scene<DIM>::contains(const Vector<DIM>& x) const
{
	return sceneData->aggregate->contains(x);
}

template<size_t DIM>
inline bool Scene<DIM>::hasLineOfSight(const Vector<DIM>& xi, const Vector<DIM>& xj) const
{
	return sceneData->aggregate->hasLineOfSight(xi, xj);
}

template<size_t DIM>
inline bool Scene<DIM>::findClosestPoint(const Vector<DIM>& x, Interaction<DIM>& i, float squaredRadius) const
{
	BoundingSphere<DIM> s(x, squaredRadius);
	return sceneData->aggregate->findClosestPoint(s, i);
}

template<size_t DIM>
inline SceneData<DIM>* Scene<DIM>::getSceneData()
{
	return sceneData.get();
}

} // namespace fcpw
