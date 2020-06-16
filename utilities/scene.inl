#include "accelerators/baseline.h"
#include "accelerators/sbvh.h"
#ifdef BUILD_ENOKI
	#include "accelerators/mbvh.h"
#endif
#ifdef BENCHMARK_EMBREE
	#include "accelerators/embree_bvh.h"
#endif
#include <fstream>
#include <sstream>

namespace fcpw {

template<size_t DIM, typename PrimitiveType>
inline std::shared_ptr<PolygonSoup<DIM>> readSoupFromFile(const std::string& filename,
					  const LoadingOption& loadingOption, bool computeWeightedNormals,
					  std::vector<std::shared_ptr<PrimitiveType>>& primitives)
{
	LOG(FATAL) << "readSoupFromFile<DIM, PrimitiveType>(): Not supported";
	return nullptr;
}

template<>
inline std::shared_ptr<PolygonSoup<3>> readSoupFromFile<3, LineSegment>(const std::string& filename,
									const LoadingOption& loadingOption, bool computeWeightedNormals,
									std::vector<std::shared_ptr<LineSegment>>& lineSegments)
{
	if (loadingOption == LoadingOption::ObjLineSegments) {
		return readLineSegmentSoupFromOBJFile(filename, lineSegments, computeWeightedNormals);
	}

	LOG(FATAL) << "readSoupFromFile<3, LineSegment>(): Invalid loading option";
	return nullptr;
}

template<>
inline std::shared_ptr<PolygonSoup<3>> readSoupFromFile<3, Triangle>(const std::string& filename,
								 const LoadingOption& loadingOption, bool computeWeightedNormals,
								 std::vector<std::shared_ptr<Triangle>>& triangles)
{
	if (loadingOption == LoadingOption::ObjTriangles) {
		return readTriangleSoupFromOBJFile(filename, triangles, computeWeightedNormals);
	}

	LOG(FATAL) << "readSoupFromFile<3, Triangle>(): Invalid loading option";
	return nullptr;
}

template<size_t DIM>
inline void loadInstanceTransforms(std::vector<std::vector<Transform<DIM>>>& instanceTransforms)
{
	// load file
	std::ifstream in(instanceFilename);
	LOG_IF(FATAL, in.is_open() == false) << "Unable to open file: " << instanceFilename;

	// parse transforms
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		int object;
		ss >> object;

		int nTransform = (int)instanceTransforms[object].size();
		instanceTransforms[object].emplace_back(Transform<DIM>());

		for (int i = 0; i <= DIM; i++) {
			for (int j = 0; j <= DIM; j++) {
				ss >> instanceTransforms[object][nTransform].matrix()(i, j);
			}
		}
	}

	// close file
	in.close();
}

inline void loadCsgTree(std::unordered_map<int, CsgTreeNode>& csgTree)
{
	// load scene
	std::ifstream in(csgFilename);
	LOG_IF(FATAL, in.is_open() == false) << "Unable to open file: " << csgFilename;

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		int node;
		ss >> node;

		std::string operationStr, child1Str, child2Str;
		ss >> operationStr >> child1Str >> child2Str;

		std::size_t found1 = child1Str.find_last_of("_");
		std::size_t found2 = child2Str.find_last_of("_");
		csgTree[node].child1 = std::stoi(child1Str.substr(found1 + 1));
		csgTree[node].child2 = std::stoi(child2Str.substr(found2 + 1));
		csgTree[node].isLeafChild1 = child1Str.find("node_") == std::string::npos;
		csgTree[node].isLeafChild2 = child2Str.find("node_") == std::string::npos;
		csgTree[node].operation = operationStr == "Union" ? BooleanOperation::Union :
								 (operationStr == "Intersection" ? BooleanOperation::Intersection :
								 (operationStr == "Difference" ? BooleanOperation::Difference : BooleanOperation::None));
	}

	// close file
	in.close();
}

template<size_t DIM>
inline void Scene<DIM>::loadFiles(bool computeWeightedNormals)
{
	// compute the number of line segment and triangle files, and map their
	// indices to the global files vector
	int nFiles = (int)files.size();
	int nLineSegmentFiles = 0;
	int nTriangleFiles = 0;
	int nMixedFiles = 0;
	lineSegmentObjectMap.clear();
	triangleObjectMap.clear();
	mixedObjectMap.clear();

	for (int i = 0; i < nFiles; i++) {
		if (files[i].second == LoadingOption::ObjLineSegments) {
			lineSegmentObjectMap[nLineSegmentFiles++] = i;

		} else if (files[i].second == LoadingOption::ObjTriangles) {
			triangleObjectMap[nTriangleFiles++] = i;
		}
	}

	soups.resize(nFiles);
	lineSegmentObjects.resize(nLineSegmentFiles);
	triangleObjects.resize(nTriangleFiles);
	mixedObjects.resize(nMixedFiles);
	instanceTransforms.resize(nFiles);

	// load line segment soups
	for (int i = 0; i < nLineSegmentFiles; i++) {
		int I = lineSegmentObjectMap[i];
		soups[I] = readSoupFromFile<3, LineSegment>(files[I].first, files[I].second,
									  computeWeightedNormals, lineSegmentObjects[i]);
	}

	// load triangle soups
	for (int i = 0; i < nTriangleFiles; i++) {
		int I = triangleObjectMap[i];
		soups[I] = readSoupFromFile<3, Triangle>(files[I].first, files[I].second,
									  computeWeightedNormals, triangleObjects[i]);
	}

	// load instance transforms
	if (!instanceFilename.empty()) loadInstanceTransforms<DIM>(instanceTransforms);

	// load csg tree
	if (!csgFilename.empty()) loadCsgTree(csgTree);
}

template<size_t DIM, typename PrimitiveType>
inline std::shared_ptr<Aggregate<DIM>> makeAggregate(const AggregateType& aggregateType, bool vectorize,
													 std::vector<std::shared_ptr<PrimitiveType>>& primitives)
{
	std::shared_ptr<Sbvh<DIM, PrimitiveType>> sbvh = nullptr;
	int leafSize = 4;
	bool packLeaves = false;

#ifdef BUILD_ENOKI
	if (vectorize) {
		leafSize = SIMD_WIDTH;
		packLeaves = true;
	}
#endif

	if (aggregateType == AggregateType::Bvh_LongestAxisCenter) {
		sbvh = std::make_shared<Sbvh<DIM, PrimitiveType>>(primitives, CostHeuristic::LongestAxisCenter,
														  1.0f, false, leafSize);

	} else if (aggregateType == AggregateType::Bvh_SurfaceArea) {
		sbvh = std::make_shared<Sbvh<DIM, PrimitiveType>>(primitives, CostHeuristic::SurfaceArea,
														  1.0f, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Bvh_OverlapSurfaceArea) {
		sbvh = std::make_shared<Sbvh<DIM, PrimitiveType>>(primitives, CostHeuristic::OverlapSurfaceArea,
														  1.0f, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Bvh_Volume) {
		sbvh = std::make_shared<Sbvh<DIM, PrimitiveType>>(primitives, CostHeuristic::Volume,
														  1.0f, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Bvh_OverlapVolume) {
		sbvh = std::make_shared<Sbvh<DIM, PrimitiveType>>(primitives, CostHeuristic::OverlapVolume,
														  1.0f, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Sbvh_SurfaceArea) {
		sbvh = std::make_shared<Sbvh<DIM, PrimitiveType>>(primitives, CostHeuristic::SurfaceArea,
														  1e-5, packLeaves, leafSize);

	} else if (aggregateType == AggregateType::Sbvh_Volume) {
		sbvh = std::make_shared<Sbvh<DIM, PrimitiveType>>(primitives, CostHeuristic::Volume,
														  1e-5, packLeaves, leafSize);

	} else {
		return std::make_shared<Baseline<DIM, PrimitiveType>>(primitives);
	}

#ifdef BUILD_ENOKI
	if (vectorize) return std::make_shared<Mbvh<SIMD_WIDTH, DIM, PrimitiveType>>(sbvh);
#endif

	return sbvh;
}

template<size_t DIM>
inline std::shared_ptr<Aggregate<DIM>> buildCsgAggregateRecursive(
				int nodeIndex, std::unordered_map<int, CsgTreeNode>& csgTree,
				std::vector<std::shared_ptr<Primitive<DIM>>>& objectInstances)
{
	const CsgTreeNode& node = csgTree[nodeIndex];
	std::shared_ptr<Primitive<DIM>> instance1, instance2;

	if (node.isLeafChild1) instance1 = objectInstances[node.child1];
	else instance1 = buildCsgAggregateRecursive(node.child1, csgTree, objectInstances);

	if (node.isLeafChild2) instance2 = objectInstances[node.child2];
	else instance2 = buildCsgAggregateRecursive(node.child2, csgTree, objectInstances);

	return std::make_shared<CsgNode<DIM>>(instance1, instance2, node.operation);
}

template<size_t DIM>
inline void Scene<DIM>::buildAggregate(const AggregateType& aggregateType, bool vectorize)
{
	// initialize instances and aggregate
	aggregate = nullptr;
	objectInstances.clear();

	// build object aggregates
	int nLineSegmentObjects = (int)lineSegmentObjects.size();
	LOG_IF(FATAL, nLineSegmentObjects != lineSegmentObjectMap.size()) << "Line segment objects and objectMap not equal in size";

	int nTriangleObjects = (int)triangleObjects.size();
	LOG_IF(FATAL, nTriangleObjects != triangleObjectMap.size()) << "Triangle objects and objectMap not equal in size";

	int nMixedObjects = (int)mixedObjects.size();
	LOG_IF(FATAL, nMixedObjects != mixedObjectMap.size()) << "Mixed objects and objectMap not equal in size";

	int nObjects = nLineSegmentObjects + nTriangleObjects + nMixedObjects;
	std::vector<std::shared_ptr<Aggregate<DIM>>> objectAggregates(nObjects);

	for (int i = 0; i < nLineSegmentObjects; i++) {
		int I = lineSegmentObjectMap[i];
		objectAggregates[I] = makeAggregate<DIM, LineSegment>(aggregateType, vectorize, lineSegmentObjects[i]);
	}

	for (int i = 0; i < nTriangleObjects; i++) {
		int I = triangleObjectMap[i];
		objectAggregates[I] = makeAggregate<DIM, Triangle>(aggregateType, vectorize, triangleObjects[i]);
	}

	for (int i = 0; i < nMixedObjects; i++) {
		int I = mixedObjectMap[i];
		objectAggregates[I] = makeAggregate<DIM, GeometricPrimitive<DIM>>(aggregateType, vectorize, mixedObjects[i]);
	}

	// build object instances
	for (int i = 0; i < nObjects; i++) {
		int nObjectInstances = (int)instanceTransforms[i].size();

		if (nObjectInstances == 0) {
			objectInstances.emplace_back(objectAggregates[i]);

		} else {
			for (int j = 0; j < nObjectInstances; j++) {
				objectInstances.emplace_back(std::make_shared<TransformedAggregate<DIM>>(objectAggregates[i],
																				  instanceTransforms[i][j]));
			}
		}
	}

	// set aggregate
	if (objectInstances.size() == 1) {
		// set to object aggregate if there is only a single object instance in the scene
		aggregate = objectAggregates[0];
		objectInstances.clear();

	} else if (csgTree.size() > 0) {
		// build csg aggregate if csg tree is specified
		aggregate = buildCsgAggregateRecursive<DIM>(0, csgTree, objectInstances);

	} else {
		// make aggregate
		aggregate = makeAggregate<DIM, Primitive<DIM>>(aggregateType, vectorize, objectInstances);
	}
}

#ifdef BENCHMARK_EMBREE
template<size_t DIM>
inline bool Scene<DIM>::buildEmbreeAggregate()
{
	if (triangleObjects.size() != 1 && triangleObjectMap.size() != 1) {
		LOG(INFO) << "Scene::buildEmbreeAggregate(): Only a single triangle object is supported at the moment";
		return false;
	}

	int index = triangleObjectMap[0];
	aggregate = std::make_shared<EmbreeBvh>(triangleObjects[0], soups[index]);
	objectInstances.clear();

	return true;
}
#endif

} // namespace fcpw
