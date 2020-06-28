#include "accelerators/baseline.h"
#include "accelerators/sbvh.h"
#ifdef BUILD_ENOKI
	#include "accelerators/mbvh.h"
#endif
#ifdef BENCHMARK_EMBREE
	#include "accelerators/embree_bvh.h"
#endif

namespace fcpw {

template<size_t DIM>
inline Scene<DIM>::Scene(bool computeNormals_):
aggregate(nullptr),
computeNormals(computeNormals_)
{

}

template<size_t DIM>
inline void Scene<DIM>::clearData()
{
	// clear line segments
	for (int i = 0; i < (int)lineSegmentObjects.size(); i++) {
		for (int j = 0; j < (int)lineSegmentObjects[i].size(); j++) {
			if (lineSegmentObjects[i][j]->pIndex == 0) {
				delete[] lineSegmentObjects[i][j];
				break;
			}
		}
	}

	// clear triangles
	for (int i = 0; i < (int)triangleObjects.size(); i++) {
		for (int j = 0; j < (int)triangleObjects[i].size(); j++) {
			if (triangleObjects[i][j]->pIndex == 0) {
				delete[] triangleObjects[i][j];
				break;
			}
		}
	}

	// clear mixed objects
	for (int i = 0; i < (int)mixedObjects.size(); i++) {
		for (int j = 0; j < (int)mixedObjects[i].size(); j++) {
			delete mixedObjects[i][j];
		}
	}

	// clear vectors
	soups.clear();
	lineSegmentObjects.clear();
	triangleObjects.clear();
	mixedObjects.clear();
	objectTypes.clear();
	instanceTransforms.clear();
	csgTree.clear();
}

template<size_t DIM>
inline void Scene<DIM>::clearAggregate()
{
	// clear object instances
	for (int i = 0; i < (int)objectInstances.size(); i++) {
		delete objectInstances[i];
	}

	// delete aggregate
	objectInstances.clear();
	if (aggregate) {
		delete aggregate;
		aggregate = nullptr;
	}
}

template<size_t DIM>
inline Scene<DIM>::~Scene()
{
	clearData();
	clearAggregate();
}

template<size_t DIM, typename PrimitiveType>
inline void readSoupFromFile(const std::string& filename, const LoadingOption& loadingOption,
							 bool computeNormals, PolygonSoup<DIM>& soup,
							 std::vector<PrimitiveType *>& primitives)
{
	std::cerr << "readSoupFromFile<DIM, PrimitiveType>(): Not supported" << std::endl;
	exit(EXIT_FAILURE);
}

template<>
inline void readSoupFromFile<3, LineSegment>(const std::string& filename, const LoadingOption& loadingOption,
											 bool computeNormals, PolygonSoup<3>& soup,
											 std::vector<LineSegment *>& lineSegments)
{
	if (loadingOption == LoadingOption::ObjLineSegments) {
		readLineSegmentSoupFromOBJFile(filename, soup);
		buildLineSegments(soup, lineSegments);
		if (computeNormals) computeWeightedLineSegmentNormals(lineSegments, soup);

	} else {
		std::cerr << "readSoupFromFile<3, LineSegment>(): Invalid loading option" << std::endl;
		exit(EXIT_FAILURE);
	}
}

template<>
inline void readSoupFromFile<3, Triangle>(const std::string& filename, const LoadingOption& loadingOption,
										  bool computeNormals, PolygonSoup<3>& soup,
										  std::vector<Triangle *>& triangles)
{
	if (loadingOption == LoadingOption::ObjTriangles) {
		readTriangleSoupFromOBJFile(filename, soup);
		buildTriangles(soup, triangles);
		if (computeNormals) computeWeightedTriangleNormals(triangles, soup);

	} else {
		std::cerr << "readSoupFromFile<3, Triangle>(): Invalid loading option" << std::endl;
		exit(EXIT_FAILURE);
	}
}

template<size_t DIM>
inline void Scene<DIM>::loadFiles()
{
	// compute the number of line segment and triangle files
	clearData();
	int nFiles = (int)files.size();
	int nLineSegmentFiles = 0;
	int nTriangleFiles = 0;
	soups.resize(nFiles);
	objectTypes.resize(nFiles);
	instanceTransforms.resize(nFiles);
	if (!csgFilename.empty() && !computeNormals) {
		std::cout << "Scene::loadFiles(): Turning normal computation now, required for distance queries to csg"
				  << std::endl;
		computeNormals = true;
	}

	for (int i = 0; i < nFiles; i++) {
		if (files[i].second == LoadingOption::ObjLineSegments) {
			objectTypes[i] = ObjectType::LineSegments;
			nLineSegmentFiles++;

		} else if (files[i].second == LoadingOption::ObjTriangles) {
			objectTypes[i] = ObjectType::Triangles;
			nTriangleFiles++;
		}
	}

	lineSegmentObjects.resize(nLineSegmentFiles);
	triangleObjects.resize(nTriangleFiles);

	// load soups
	nLineSegmentFiles = 0;
	nTriangleFiles = 0;

	for (int i = 0; i < nFiles; i++) {
		if (objectTypes[i] == ObjectType::LineSegments) {
			readSoupFromFile<3, LineSegment>(files[i].first, files[i].second, computeNormals,
											 soups[i], lineSegmentObjects[nLineSegmentFiles]);
			nLineSegmentFiles++;

		} else if (objectTypes[i] == ObjectType::Triangles) {
			readSoupFromFile<3, Triangle>(files[i].first, files[i].second, computeNormals,
										  soups[i], triangleObjects[nTriangleFiles]);
			nTriangleFiles++;
		}
	}

	// load instance transforms
	if (!instanceFilename.empty()) {
		loadInstanceTransforms<DIM>(instanceFilename, instanceTransforms);
	}

	// load csg tree
	if (!csgFilename.empty()) {
		loadCsgTree(csgFilename, csgTree);
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

template<size_t DIM, typename PrimitiveType>
inline Aggregate<DIM>* makeAggregate(const AggregateType& aggregateType,
									 std::vector<PrimitiveType *>& primitives,
									 bool printStats, bool vectorize,
									 SortPositionsFunc<DIM, PrimitiveType> sortPositions={})
{
	Sbvh<DIM, PrimitiveType> *sbvh = nullptr;
	int leafSize = 4;
	bool packLeaves = false;

#ifdef BUILD_ENOKI
	if (vectorize) {
		leafSize = SIMD_WIDTH;
		packLeaves = true;
	}
#endif

	if (aggregateType == AggregateType::Bvh_LongestAxisCenter) {
		sbvh = new Sbvh<DIM, PrimitiveType>(CostHeuristic::LongestAxisCenter, primitives,
											sortPositions, printStats, false, leafSize);

	} else if (aggregateType == AggregateType::Bvh_SurfaceArea) {
		sbvh = new Sbvh<DIM, PrimitiveType>(CostHeuristic::SurfaceArea, primitives,
											sortPositions, printStats, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Bvh_OverlapSurfaceArea) {
		sbvh = new Sbvh<DIM, PrimitiveType>(CostHeuristic::OverlapSurfaceArea, primitives,
			 								sortPositions, printStats, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Bvh_Volume) {
		sbvh = new Sbvh<DIM, PrimitiveType>(CostHeuristic::Volume, primitives,
											sortPositions, printStats, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Bvh_OverlapVolume) {
		sbvh = new Sbvh<DIM, PrimitiveType>(CostHeuristic::OverlapVolume, primitives,
											sortPositions, printStats, packLeaves, leafSize);

	} else {
		return new Baseline<DIM, PrimitiveType>(primitives);
	}

#ifdef BUILD_ENOKI
	if (vectorize) {
		Mbvh<SIMD_WIDTH, DIM, PrimitiveType> *mbvh = new Mbvh<SIMD_WIDTH, DIM, PrimitiveType>(sbvh, printStats);
		delete sbvh;

		return mbvh;
	}
#endif

	return sbvh;
}

template<size_t DIM>
inline Aggregate<DIM>* buildCsgAggregateRecursive(int nodeIndex, std::unordered_map<int, CsgTreeNode>& csgTree,
												  std::vector<Aggregate<DIM> *>& objectInstances)
{
	const CsgTreeNode& node = csgTree[nodeIndex];
	std::unique_ptr<Aggregate<DIM>> instance1 = nullptr;
	std::unique_ptr<Aggregate<DIM>> instance2 = nullptr;

	if (node.isLeafChild1) instance1 = std::unique_ptr<Aggregate<DIM>>(objectInstances[node.child1]);
	else instance1 = std::unique_ptr<Aggregate<DIM>>(buildCsgAggregateRecursive(node.child1, csgTree, objectInstances));

	if (node.isLeafChild2) instance2 = std::unique_ptr<Aggregate<DIM>>(objectInstances[node.child2]);
	else instance2 = std::unique_ptr<Aggregate<DIM>>(buildCsgAggregateRecursive(node.child2, csgTree, objectInstances));

	return new CsgNode<DIM, Aggregate<DIM>, Aggregate<DIM>>(std::move(instance1), std::move(instance2), node.operation);
}

using SortLineSegmentPositionsFunc = std::function<void(const std::vector<SbvhNode<3>>&, std::vector<LineSegment *>&)>;
using SortTrianglePositionsFunc = std::function<void(const std::vector<SbvhNode<3>>&, std::vector<Triangle *>&)>;

template<size_t DIM>
inline void Scene<DIM>::buildAggregate(const AggregateType& aggregateType, bool printStats, bool vectorize)
{
	// build object aggregates
	clearAggregate();
	int nObjects = (int)soups.size();
	std::vector<Aggregate<DIM> *> objectAggregates(nObjects);
	int nLineSegmentObjects = 0;
	int nTriangleObjects = 0;
	int nMixedObjects = 0;
	SortLineSegmentPositionsFunc sortLineSegmentPositions = {};
	SortTrianglePositionsFunc sortTrianglePositionsFunc = {};

	for (int i = 0; i < nObjects; i++) {
		if (objectTypes[i] == ObjectType::LineSegments) {
			if (!vectorize) {
				sortLineSegmentPositions = std::bind(&sortSoupPositions<3, LineSegment>,
													 std::placeholders::_1, std::placeholders::_2,
													 std::ref(soups[i]));
			}

			objectAggregates[i] = makeAggregate<DIM, LineSegment>(aggregateType, lineSegmentObjects[nLineSegmentObjects],
																  printStats, vectorize, sortLineSegmentPositions);
			nLineSegmentObjects++;

		} else if (objectTypes[i] == ObjectType::Triangles) {
			if (!vectorize) {
				sortTrianglePositionsFunc = std::bind(&sortSoupPositions<3, Triangle>,
													  std::placeholders::_1, std::placeholders::_2,
													  std::ref(soups[i]));
			}

			objectAggregates[i] = makeAggregate<DIM, Triangle>(aggregateType, triangleObjects[nTriangleObjects],
															   printStats, vectorize, sortTrianglePositionsFunc);
			nTriangleObjects++;

		} else if (objectTypes[i] == ObjectType::Mixed) {
			objectAggregates[i] = makeAggregate<DIM, GeometricPrimitive<DIM>>(aggregateType, mixedObjects[nMixedObjects],
																			  printStats, vectorize);
			nMixedObjects++;
		}

		objectAggregates[i]->computeNormals = computeNormals;
	}

	// build object instances
	for (int i = 0; i < nObjects; i++) {
		int nObjectInstances = (int)instanceTransforms[i].size();

		if (nObjectInstances == 0) {
			objectInstances.emplace_back(objectAggregates[i]);

		} else {
			std::shared_ptr<Aggregate<DIM>> aggregate(objectAggregates[i]);
			for (int j = 0; j < nObjectInstances; j++) {
				objectInstances.emplace_back(new TransformedAggregate<DIM>(aggregate, instanceTransforms[i][j]));
			}
		}
	}

	// set aggregate
	if (objectInstances.size() == 1) {
		// set to object aggregate if there is only a single object instance in the scene
		aggregate = objectInstances[0];
		objectInstances.clear();

	} else if (csgTree.size() > 0) {
		// build csg aggregate if csg tree is specified
		aggregate = buildCsgAggregateRecursive<DIM>(0, csgTree, objectInstances);
		objectInstances.clear();

	} else {
		// make aggregate
		aggregate = makeAggregate<DIM, Aggregate<DIM>>(aggregateType, objectInstances, printStats, vectorize);
	}
}

#ifdef BENCHMARK_EMBREE
template<size_t DIM>
inline bool Scene<DIM>::buildEmbreeAggregate(bool printStats)
{
	clearAggregate();
	if (triangleObjects.size() != 1) {
		std::cout << "Scene::buildEmbreeAggregate(): Only a single triangle object is supported at the moment"
				  << std::endl;
		return false;
	}

	for (int i = 0; i < (int)soups.size(); i++) {
		if (objectTypes[i] == ObjectType::Triangles) {
			aggregate = new EmbreeBvh(triangleObjects[0], &soups[i], printStats);
			return true;
		}
	}

	std::cout << "Scene::buildEmbreeAggregate(): Only triangles supported at the moment" << std::endl;
	return false;
}
#endif

} // namespace fcpw
