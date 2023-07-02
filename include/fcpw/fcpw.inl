#include <fcpw/aggregates/baseline.h>
#include <fcpw/aggregates/sbvh.h>
#ifdef FCPW_USE_ENOKI
	#include <fcpw/aggregates/mbvh.h>
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
inline void Scene<DIM>::setObjectLineSegment(const int *indices, int lineSegmentIndex, int objectIndex)
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
inline void Scene<DIM>::setObjectTriangle(const int *indices, int triangleIndex, int objectIndex)
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
inline void Scene<DIM>::setObjectPrimitive(const int *indices, const PrimitiveType& primitiveType,
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

inline int assignEdgeIndices(const std::vector<Triangle>& triangles, PolygonSoup<3>& soup)
{
	int E = 0;
	int N = (int)triangles.size();
	std::map<std::pair<int, int>, int> indexMap;
	soup.eIndices.clear();

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

	return E;
}

template<size_t DIM>
inline void Scene<DIM>::computeSilhouettes(const std::function<bool(float, int)>& ignoreSilhouette)
{
	int nLineSegmentObjects = (int)sceneData->lineSegmentObjects.size();
	int nTriangleObjects = (int)sceneData->triangleObjects.size();
	sceneData->silhouetteVertexObjects.resize(nLineSegmentObjects);
	sceneData->silhouetteEdgeObjects.resize(nTriangleObjects);
	sceneData->ignoreSilhouette = ignoreSilhouette;

	for (auto& kv: sceneData->soupToObjectsMap) {
		int objectIndex = kv.first;
		const std::vector<std::pair<ObjectType, int>>& objectsMap = kv.second;
		PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];

		if (objectsMap[0].first == ObjectType::LineSegments) {
			// allocate silhouette vertices
			int lineSegmentObjectIndex = objectsMap[0].second;
			int nLineSegments = (int)soup.indices.size()/2;
			int nSilhouetteVertices = (int)soup.positions.size();
			sceneData->silhouetteVertexObjects[lineSegmentObjectIndex] =
				std::unique_ptr<std::vector<SilhouetteVertex>>(new std::vector<SilhouetteVertex>(nSilhouetteVertices));

			// assign soup and indices to silhouette vertices
			for (int i = 0; i < nLineSegments; i++) {
				const LineSegment& lineSegment = (*sceneData->lineSegmentObjects[lineSegmentObjectIndex])[i];

				SilhouetteVertex& silhouetteVertex1 = (*sceneData->silhouetteVertexObjects[lineSegmentObjectIndex])[lineSegment.indices[0]];
				silhouetteVertex1.soup = &soup;
				silhouetteVertex1.indices[1] = lineSegment.indices[0];
				silhouetteVertex1.indices[2] = lineSegment.indices[1];
				silhouetteVertex1.pIndex = lineSegment.indices[0];

				SilhouetteVertex& silhouetteVertex2 = (*sceneData->silhouetteVertexObjects[lineSegmentObjectIndex])[lineSegment.indices[1]];
				silhouetteVertex2.soup = &soup;
				silhouetteVertex2.indices[0] = lineSegment.indices[0];
				silhouetteVertex2.indices[1] = lineSegment.indices[1];
				silhouetteVertex2.pIndex = lineSegment.indices[1];
			}

		} else if (objectsMap[0].first == ObjectType::Triangles) {
			// allocate silhouette edges
			int triangleObjectIndex = objectsMap[0].second;
			int nTriangles = (int)soup.indices.size()/3;
			int nSilhouetteEdges = assignEdgeIndices(*sceneData->triangleObjects[triangleObjectIndex], soup);
			sceneData->silhouetteEdgeObjects[triangleObjectIndex] =
				std::unique_ptr<std::vector<SilhouetteEdge>>(new std::vector<SilhouetteEdge>(nSilhouetteEdges));

			// assign soup and indices to silhouette edges
			for (int i = 0; i < nTriangles; i++) {
				const Triangle& triangle = (*sceneData->triangleObjects[triangleObjectIndex])[i];

				for (int j = 0; j < 3; j++) {
					int I = j - 1 < 0 ? 2 : j - 1;
					int J = j + 0;
					int K = j + 1 > 2 ? 0 : j + 1;
					int eIndex = soup.eIndices[3*triangle.pIndex + j];

					float orientation = 1;
					if (triangle.indices[J] > triangle.indices[K]) {
						std::swap(J, K);
						orientation *= -1;
					}

					SilhouetteEdge& silhouetteEdge = (*sceneData->silhouetteEdgeObjects[triangleObjectIndex])[eIndex];
					silhouetteEdge.soup = &soup;
					silhouetteEdge.indices[orientation == 1 ? 0 : 3] = triangle.indices[I];
					silhouetteEdge.indices[1] = triangle.indices[J];
					silhouetteEdge.indices[2] = triangle.indices[K];
					silhouetteEdge.pIndex = eIndex;
				}
			}
		}
	}
}

template<size_t DIM, typename PrimitiveType>
inline void computeNormals(const std::vector<PrimitiveType>& primitives,
						   PolygonSoup<DIM>& soup, bool computeWeighted)
{
	// do nothing
}

template<>
inline void computeNormals<3, LineSegment>(const std::vector<LineSegment>& lineSegments,
										   PolygonSoup<3>& soup, bool computeWeighted)
{
	int N = (int)lineSegments.size();
	int V = (int)soup.positions.size();
	soup.vNormals.clear();
	soup.vNormals.resize(V, Vector<3>::Zero());

	for (int i = 0; i < N; i++) {
		Vector3 n = lineSegments[i].normal(true);
		float a = computeWeighted ? lineSegments[i].surfaceArea() : 1.0f;

		soup.vNormals[lineSegments[i].indices[0]] += a*n;
		soup.vNormals[lineSegments[i].indices[1]] += a*n;
	}

	for (int i = 0; i < V; i++) {
		soup.vNormals[i].normalize();
	}
}

template<>
inline void computeNormals<3, Triangle>(const std::vector<Triangle>& triangles,
										PolygonSoup<3>& soup, bool computeWeighted)
{
	int N = (int)triangles.size();
	int V = (int)soup.positions.size();
	int E = assignEdgeIndices(triangles, soup);
	soup.vNormals.clear();
	soup.eNormals.clear();
	soup.vNormals.resize(V, Vector<3>::Zero());
	soup.eNormals.resize(E, Vector<3>::Zero());

	for (int i = 0; i < N; i++) {
		Vector3 n = triangles[i].normal(true);
		float area = triangles[i].surfaceArea();

		for (int j = 0; j < 3; j++) {
			float angle = computeWeighted ? triangles[i].angle(j) : 1.0f;

			soup.vNormals[triangles[i].indices[j]] += angle*n;
			soup.eNormals[soup.eIndices[3*triangles[i].pIndex + j]] += area*n;
		}
	}

	for (int i = 0; i < V; i++) soup.vNormals[i].normalize();
	for (int i = 0; i < E; i++) soup.eNormals[i].normalize();
}

template<size_t DIM>
inline void Scene<DIM>::computeObjectNormals(int objectIndex, bool computeWeighted)
{
	const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[objectIndex];
	PolygonSoup<DIM>& soup = sceneData->soups[objectIndex];

	if (objectsMap[0].first == ObjectType::LineSegments) {
		int lineSegmentObjectIndex = objectsMap[0].second;
		computeNormals<DIM, LineSegment>(*sceneData->lineSegmentObjects[lineSegmentObjectIndex],
										 soup, computeWeighted);

	} else if (objectsMap[0].first == ObjectType::Triangles) {
		int triangleObjectIndex = objectsMap[0].second;
		computeNormals<DIM, Triangle>(*sceneData->triangleObjects[triangleObjectIndex],
									  soup, computeWeighted);
	}
}

template<bool CONEDATA>
inline void sortLineSegmentSoupPositions(const std::vector<SbvhNode<3, CONEDATA>>& flatTree,
										 std::vector<LineSegment *>& lineSegments,
										 PolygonSoup<3>& soup, std::vector<int>& indexMap)
{
	int V = (int)soup.positions.size();
	std::vector<Vector<3>> sortedPositions(V), sortedVertexNormals(V);
	indexMap.resize(V, -1);
	int v = 0;

	// collect sorted positions, updating line segment and soup indices
	for (int i = 0; i < (int)flatTree.size(); i++) {
		const SbvhNode<3, CONEDATA>& node(flatTree[i]);

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

template<bool CONEDATA>
inline void sortTriangleSoupPositions(const std::vector<SbvhNode<3, CONEDATA>>& flatTree,
									  std::vector<Triangle *>& triangles,
									  PolygonSoup<3>& soup, std::vector<int>& indexMap)
{
	int V = (int)soup.positions.size();
	std::vector<Vector<3>> sortedPositions(V), sortedVertexNormals(V);
	indexMap.resize(V, -1);
	int v = 0;

	// collect sorted positions, updating triangle and soup indices
	for (int i = 0; i < (int)flatTree.size(); i++) {
		const SbvhNode<3, CONEDATA>& node(flatTree[i]);

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

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline void sortSoupPositions(const std::vector<SbvhNode<DIM, CONEDATA>>& flatTree,
							  std::vector<PrimitiveType *>& primitives,
							  std::vector<SilhouetteType *>& silhouettes,
							  PolygonSoup<DIM>& soup)
{
	// do nothing
}

template<>
inline void sortSoupPositions<3, false, LineSegment, SilhouettePrimitive<3>>(const std::vector<SbvhNode<3, false>>& flatTree,
																			 std::vector<LineSegment *>& lineSegments,
																			 std::vector<SilhouettePrimitive<3> *>& silhouettes,
																			 PolygonSoup<3>& soup)
{
	std::vector<int> indexMap;
	sortLineSegmentSoupPositions<false>(flatTree, lineSegments, soup, indexMap);
}

template<>
inline void sortSoupPositions<3, true, LineSegment, SilhouetteVertex>(const std::vector<SbvhNode<3, true>>& flatTree,
																	  std::vector<LineSegment *>& lineSegments,
																	  std::vector<SilhouetteVertex *>& silhouetteVertices,
																	  PolygonSoup<3>& soup)
{
	std::vector<int> indexMap;
	sortLineSegmentSoupPositions<true>(flatTree, lineSegments, soup, indexMap);

	for (int i = 0; i < (int)lineSegments.size(); i++) {
		int index1 = lineSegments[i]->indices[0];
		int index2 = lineSegments[i]->indices[1];

		SilhouetteVertex *silhouetteVertex1 = silhouetteVertices[index1];
		silhouetteVertex1->indices[1] = index1;
		silhouetteVertex1->indices[2] = index2;
		silhouetteVertex1->pIndex = index1;

		SilhouetteVertex *silhouetteVertex2 = silhouetteVertices[index2];
		silhouetteVertex2->indices[0] = index1;
		silhouetteVertex2->indices[1] = index2;
		silhouetteVertex2->pIndex = index2;
	}
}

template<>
inline void sortSoupPositions<3, false, Triangle, SilhouettePrimitive<3>>(const std::vector<SbvhNode<3, false>>& flatTree,
																		  std::vector<Triangle *>& triangles,
																		  std::vector<SilhouettePrimitive<3> *>& silhouettes,
																		  PolygonSoup<3>& soup)
{
	std::vector<int> indexMap;
	sortTriangleSoupPositions<false>(flatTree, triangles, soup, indexMap);
}

template<>
inline void sortSoupPositions<3, true, Triangle, SilhouetteEdge>(const std::vector<SbvhNode<3, true>>& flatTree,
																 std::vector<Triangle *>& triangles,
																 std::vector<SilhouetteEdge *>& silhouetteEdges,
																 PolygonSoup<3>& soup)
{
	std::vector<int> indexMap;
	sortTriangleSoupPositions<true>(flatTree, triangles, soup, indexMap);

	for (int i = 0; i < silhouetteEdges.size(); i++) {
		SilhouetteEdge *silhouetteEdge = silhouetteEdges[i];

		for (int j = 0; j < 4; j++) {
			int vIndex = silhouetteEdge->indices[j];
			if (vIndex != -1) silhouetteEdge->indices[j] = indexMap[vIndex];
		}
	}
}

template<size_t DIM, bool CONEDATA, typename PrimitiveType, typename SilhouetteType>
inline std::unique_ptr<Aggregate<DIM>> makeAggregate(const AggregateType& aggregateType,
													 std::vector<PrimitiveType *>& primitives,
													 std::vector<SilhouetteType *>& silhouettes,
													 bool vectorize, bool printStats,
													 SortPositionsFunc<DIM, CONEDATA, PrimitiveType, SilhouetteType> sortPositions={},
													 const std::function<bool(float, int)>& ignoreSilhouette={})
{
	std::unique_ptr<Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>> sbvh = nullptr;
	bool packLeaves = false;
	int leafSize = 4;

#ifdef FCPW_USE_ENOKI
	if (vectorize) {
		packLeaves = true;
		leafSize = FCPW_SIMD_WIDTH;
	}
#endif

	if (aggregateType == AggregateType::Bvh_LongestAxisCenter) {
		sbvh = std::unique_ptr<Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>>(new Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>(
				CostHeuristic::LongestAxisCenter, primitives, silhouettes, sortPositions, ignoreSilhouette, printStats, false, leafSize));

	} else if (aggregateType == AggregateType::Bvh_SurfaceArea) {
		sbvh = std::unique_ptr<Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>>(new Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>(
				CostHeuristic::SurfaceArea, primitives, silhouettes, sortPositions, ignoreSilhouette, printStats, packLeaves, leafSize));

	} else if (aggregateType == AggregateType::Bvh_OverlapSurfaceArea) {
		sbvh = std::unique_ptr<Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>>(new Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>(
				CostHeuristic::OverlapSurfaceArea, primitives, silhouettes, sortPositions, ignoreSilhouette, printStats, packLeaves, leafSize));

	} else if (aggregateType == AggregateType::Bvh_Volume) {
		sbvh = std::unique_ptr<Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>>(new Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>(
				CostHeuristic::Volume, primitives, silhouettes, sortPositions, ignoreSilhouette, printStats, packLeaves, leafSize));

	} else if (aggregateType == AggregateType::Bvh_OverlapVolume) {
		sbvh = std::unique_ptr<Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>>(new Sbvh<DIM, CONEDATA, PrimitiveType, SilhouetteType>(
				CostHeuristic::OverlapVolume, primitives, silhouettes, sortPositions, ignoreSilhouette, printStats, packLeaves, leafSize));

	} else {
		return std::unique_ptr<Baseline<DIM, PrimitiveType, SilhouetteType>>(
				new Baseline<DIM, PrimitiveType, SilhouetteType>(primitives, silhouettes));
	}

#ifdef FCPW_USE_ENOKI
	if (vectorize) {
		return std::unique_ptr<Mbvh<FCPW_SIMD_WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>>(
				new Mbvh<FCPW_SIMD_WIDTH, DIM, CONEDATA, PrimitiveType, SilhouetteType>(sbvh.get(), printStats));
	}
#endif

	return sbvh;
}

template<size_t DIM>
inline void buildGeometricAggregates(const AggregateType& aggregateType, bool vectorize, bool printStats,
									 const std::function<bool(float, int)>& ignoreSilhouette,
									 std::unique_ptr<SceneData<DIM>>& sceneData,
									 std::vector<std::unique_ptr<Aggregate<DIM>>>& objectAggregates)
{
	std::cerr << "buildGeometricAggregates(): DIM: " << DIM << std::endl;
	exit(EXIT_FAILURE);
}

template<>
inline void buildGeometricAggregates<3>(const AggregateType& aggregateType, bool vectorize, bool printStats,
										const std::function<bool(float, int)>& ignoreSilhouette,
										std::unique_ptr<SceneData<3>>& sceneData,
										std::vector<std::unique_ptr<Aggregate<3>>>& objectAggregates)
{
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

	objectAggregates.resize(nObjects);
	sceneData->lineSegmentObjectPtrs.resize(nLineSegmentObjectPtrs);
	sceneData->triangleObjectPtrs.resize(nTriangleObjectPtrs);
	sceneData->mixedObjectPtrs.resize(nMixedObjectPtrs);
	sceneData->silhouetteVertexObjectPtrs.resize(nLineSegmentObjectPtrs);
	sceneData->silhouetteEdgeObjectPtrs.resize(nTriangleObjectPtrs);

	// populate the object ptrs and make their aggregates
	int nAggregates = 0;
	nLineSegmentObjectPtrs = 0;
	nTriangleObjectPtrs = 0;
	nMixedObjectPtrs = 0;

	for (int i = 0; i < nObjects; i++) {
		const std::vector<std::pair<ObjectType, int>>& objectsMap = sceneData->soupToObjectsMap[i];

		if (objectsMap.size() > 1) {
			// soup contains mixed primitives, set mixed object ptrs
			std::vector<GeometricPrimitive<3> *>& mixedObjectPtr = sceneData->mixedObjectPtrs[nMixedObjectPtrs];

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

			objectAggregates[i] = makeAggregate<3, false, GeometricPrimitive<3>, SilhouettePrimitive<3>>(aggregateType, mixedObjectPtr,
																										 sceneData->silhouetteObjectPtrStub,
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

			if (sceneData->silhouetteVertexObjects.size() > 0) {
				// soup contains silhouette vertices, set silhouette vertex object ptrs
				std::vector<SilhouetteVertex>& silhouetteVertexObject = *sceneData->silhouetteVertexObjects[lineSegmentObjectIndex];
				std::vector<SilhouetteVertex *>& silhouetteVertexObjectPtr = sceneData->silhouetteVertexObjectPtrs[nLineSegmentObjectPtrs];

				for (int j = 0; j < (int)silhouetteVertexObject.size(); j++) {
					silhouetteVertexObjectPtr.emplace_back(&silhouetteVertexObject[j]);
				}

				using SortLineSegmentPositionsFunc = std::function<void(const std::vector<SbvhNode<3, true>>&,
																		std::vector<LineSegment *>&,
																		std::vector<SilhouetteVertex *>&)>;
				SortLineSegmentPositionsFunc sortLineSegmentPositions = std::bind(&sortSoupPositions<3, true, LineSegment, SilhouetteVertex>,
																				  std::placeholders::_1, std::placeholders::_2,
																				  std::placeholders::_3, std::ref(sceneData->soups[i]));
				objectAggregates[i] = makeAggregate<3, true, LineSegment, SilhouetteVertex>(aggregateType, lineSegmentObjectPtr,
																							silhouetteVertexObjectPtr, vectorize,
																							printStats, sortLineSegmentPositions,
																							ignoreSilhouette);

			} else {
				using SortLineSegmentPositionsFunc = std::function<void(const std::vector<SbvhNode<3, false>>&,
																		std::vector<LineSegment *>&,
																		std::vector<SilhouettePrimitive<3> *>&)>;
				SortLineSegmentPositionsFunc sortLineSegmentPositions = std::bind(&sortSoupPositions<3, false, LineSegment, SilhouettePrimitive<3>>,
																				  std::placeholders::_1, std::placeholders::_2,
																				  std::placeholders::_3, std::ref(sceneData->soups[i]));
				objectAggregates[i] = makeAggregate<3, false, LineSegment, SilhouettePrimitive<3>>(aggregateType, lineSegmentObjectPtr,
																								   sceneData->silhouetteObjectPtrStub, vectorize,
																								   printStats, sortLineSegmentPositions);
			}

			nLineSegmentObjectPtrs++;

		} else if (objectsMap[0].first == ObjectType::Triangles) {
			// soup contains triangles, set triangle object ptrs
			int triangleObjectIndex = objectsMap[0].second;
			std::vector<Triangle>& triangleObject = *sceneData->triangleObjects[triangleObjectIndex];
			std::vector<Triangle *>& triangleObjectPtr = sceneData->triangleObjectPtrs[nTriangleObjectPtrs];

			for (int j = 0; j < (int)triangleObject.size(); j++) {
				triangleObjectPtr.emplace_back(&triangleObject[j]);
			}

			if (sceneData->silhouetteEdgeObjects.size() > 0) {
				// soup contains silhouette edges, set silhouette edge object ptrs
				std::vector<SilhouetteEdge>& silhouetteEdgeObject = *sceneData->silhouetteEdgeObjects[triangleObjectIndex];
				std::vector<SilhouetteEdge *>& silhouetteEdgeObjectPtr = sceneData->silhouetteEdgeObjectPtrs[nTriangleObjectPtrs];

				for (int j = 0; j < (int)silhouetteEdgeObject.size(); j++) {
					silhouetteEdgeObjectPtr.emplace_back(&silhouetteEdgeObject[j]);
				}

				using SortTrianglePositionsFunc = std::function<void(const std::vector<SbvhNode<3, true>>&,
																	 std::vector<Triangle *>&,
																	 std::vector<SilhouetteEdge *>&)>;
				SortTrianglePositionsFunc sortTrianglePositions = std::bind(&sortSoupPositions<3, true, Triangle, SilhouetteEdge>,
																			std::placeholders::_1, std::placeholders::_2,
																			std::placeholders::_3, std::ref(sceneData->soups[i]));
				objectAggregates[i] = makeAggregate<3, true, Triangle, SilhouetteEdge>(aggregateType, triangleObjectPtr,
																					   silhouetteEdgeObjectPtr, vectorize,
																					   printStats, sortTrianglePositions,
																					   ignoreSilhouette);

			} else {
				using SortTrianglePositionsFunc = std::function<void(const std::vector<SbvhNode<3, false>>&,
																	 std::vector<Triangle *>&,
																	 std::vector<SilhouettePrimitive<3> *>&)>;
				SortTrianglePositionsFunc sortTrianglePositions = std::bind(&sortSoupPositions<3, false, Triangle, SilhouettePrimitive<3>>,
																			std::placeholders::_1, std::placeholders::_2,
																			std::placeholders::_3, std::ref(sceneData->soups[i]));
				objectAggregates[i] = makeAggregate<3, false, Triangle, SilhouettePrimitive<3>>(aggregateType, triangleObjectPtr,
																								sceneData->silhouetteObjectPtrStub, vectorize,
																								printStats, sortTrianglePositions);
			}

			nTriangleObjectPtrs++;
		}

		objectAggregates[i]->index = nAggregates++;
	}
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
inline void Scene<DIM>::build(const AggregateType& aggregateType, bool vectorize,
							  bool printStats, bool reduceMemoryFootprint)
{
	// clear old aggregate data
	sceneData->clearAggregateData();

	// build geometric aggregates
	std::vector<std::unique_ptr<Aggregate<DIM>>> objectAggregates;
	buildGeometricAggregates<DIM>(aggregateType, vectorize, printStats,
								  sceneData->ignoreSilhouette,
								  sceneData, objectAggregates);
	int nAggregates = (int)objectAggregates.size();

	// build aggregate instances and instance ptrs
	for (int i = 0; i < (int)sceneData->soups.size(); i++) {
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

	// build root aggregate
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
		// make aggregate of aggregates
		if (sceneData->silhouetteVertexObjects.size() > 0 || sceneData->silhouetteEdgeObjects.size() > 0) {
			sceneData->aggregate = makeAggregate<DIM, true, Aggregate<DIM>, SilhouettePrimitive<DIM>>(aggregateType, sceneData->aggregateInstancePtrs,
																									  sceneData->silhouetteObjectPtrStub, false, printStats);

		} else {
			sceneData->aggregate = makeAggregate<DIM, false, Aggregate<DIM>, SilhouettePrimitive<DIM>>(aggregateType, sceneData->aggregateInstancePtrs,
																									   sceneData->silhouetteObjectPtrStub, false, printStats);
		}

		sceneData->aggregate->index = nAggregates++;
	}

	// reduce memory footprint of aggregate
	if (reduceMemoryFootprint) {
		sceneData->soupToObjectsMap.clear();
		sceneData->instanceTransforms.clear();
		sceneData->csgTree.clear();

		for (int i = 0; i < (int)sceneData->soups.size(); i++) {
			PolygonSoup<DIM>& soup = sceneData->soups[i];
			soup.indices.clear();
			if (vectorize && sceneData->mixedObjectPtrs.size() == 0 && soup.vNormals.size() == 0) {
				soup.positions.clear();
			}
		}
	}
}

template<size_t DIM>
inline int Scene<DIM>::intersect(Ray<DIM>& r, std::vector<Interaction<DIM>>& is, bool checkForOcclusion,
								 bool recordAllHits) const
{
	return sceneData->aggregate->intersect(r, is, checkForOcclusion, recordAllHits);
}

template<size_t DIM>
inline int Scene<DIM>::intersect(const BoundingSphere<DIM>& s,
								 std::vector<Interaction<DIM>>& is, bool recordOneHit,
								 const std::function<float(float)>& primitiveWeight) const
{
	return sceneData->aggregate->intersect(s, is, recordOneHit, primitiveWeight);
}

template<size_t DIM>
inline int Scene<DIM>::intersectStochastic(const BoundingSphere<DIM>& s,
										   std::vector<Interaction<DIM>>& is, float *randNums,
										   const std::function<float(float)>& traversalWeight,
										   const std::function<float(float)>& primitiveWeight) const
{
	return sceneData->aggregate->intersectStochastic(s, is, randNums, traversalWeight, primitiveWeight);
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
inline bool Scene<DIM>::findClosestPoint(const Vector<DIM>& x, Interaction<DIM>& i,
										 float squaredRadius, bool recordNormal) const
{
	BoundingSphere<DIM> s(x, squaredRadius);
	return sceneData->aggregate->findClosestPoint(s, i, recordNormal);
}

template<size_t DIM>
inline bool Scene<DIM>::findClosestSilhouettePoint(const Vector<DIM>& x, Interaction<DIM>& i,
												   bool flipNormalOrientation, float squaredMinRadius,
												   float squaredMaxRadius, float precision, bool recordNormal) const
{
	BoundingSphere<DIM> s(x, squaredMaxRadius);
	return sceneData->aggregate->findClosestSilhouettePoint(s, i, flipNormalOrientation, squaredMinRadius,
															precision, recordNormal);
}

template<size_t DIM>
inline SceneData<DIM>* Scene<DIM>::getSceneData()
{
	return sceneData.get();
}

} // namespace fcpw
