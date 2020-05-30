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

template <int DIM>
inline std::shared_ptr<PolygonSoup<DIM>> readSoupFromFile(
			const std::string& filename, const LoadingOption& loadingOption, bool computeWeightedNormals,
			std::vector<std::shared_ptr<Primitive<DIM>>>& primitives, ObjectType& objectType)
{
	LOG(FATAL) << "readSoupFromFile<DIM>(): Not implemented";
	return nullptr;
}

template <>
inline std::shared_ptr<PolygonSoup<3>> readSoupFromFile(
			const std::string& filename, const LoadingOption& loadingOption, bool computeWeightedNormals,
			std::vector<std::shared_ptr<Primitive<3>>>& primitives, ObjectType& objectType)
{
	if (loadingOption == LoadingOption::ObjTriangles) {
		objectType = ObjectType::Triangles;
		return readFromOBJFile(filename, primitives, computeWeightedNormals);
	}

	LOG(FATAL) << "readSoupFromFile<3>(): Invalid loading option";
	return nullptr;
}

template <int DIM>
inline void loadInstanceTransforms(
				std::vector<std::vector<Transform<DIM>>>& instanceTransforms)
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

template <int DIM>
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

template <int DIM>
inline void Scene<DIM>::loadFiles(bool computeWeightedNormals)
{
	int nFiles = (int)files.size();
	soups.resize(nFiles);
	objects.resize(nFiles);
	instanceTransforms.resize(nFiles);
	objectTypes.resize(nFiles);

	// load soups and primitives
	for (int i = 0; i < nFiles; i++) {
		soups[i] = readSoupFromFile<DIM>(files[i].first, files[i].second,
										 computeWeightedNormals, objects[i], objectTypes[i]);
	}

	// load instance transforms
	if (!instanceFilename.empty()) loadInstanceTransforms<DIM>(instanceTransforms);

	// load csg tree
	if (!csgFilename.empty()) loadCsgTree<DIM>(csgTree);
}

template <int DIM>
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

template <int DIM>
inline std::shared_ptr<Aggregate<DIM>> makeAggregate(const AggregateType& aggregateType, bool vectorize,
													 std::vector<std::shared_ptr<Primitive<DIM>>>& primitives)
{
	std::shared_ptr<Sbvh<DIM>> sbvh = nullptr;
	int leafSize = 4;

#ifdef BUILD_ENOKI
	if (vectorize) leafSize = SIMD_WIDTH;
#endif

	if (aggregateType == AggregateType::Bvh_LongestAxisCenter) {
		sbvh = std::make_shared<Sbvh<DIM>>(primitives, CostHeuristic::LongestAxisCenter, 1.0f, leafSize);

	} else if (aggregateType == AggregateType::Bvh_SurfaceArea) {
		sbvh = std::make_shared<Sbvh<DIM>>(primitives, CostHeuristic::SurfaceArea, 1.0f, leafSize);

	} else if (aggregateType == AggregateType::Bvh_OverlapSurfaceArea) {
		sbvh = std::make_shared<Sbvh<DIM>>(primitives, CostHeuristic::OverlapSurfaceArea, 1.0f, leafSize);

	} else if (aggregateType == AggregateType::Bvh_Volume) {
		sbvh = std::make_shared<Sbvh<DIM>>(primitives, CostHeuristic::Volume, 1.0f, leafSize);

	} else if (aggregateType == AggregateType::Bvh_OverlapVolume) {
		sbvh = std::make_shared<Sbvh<DIM>>(primitives, CostHeuristic::OverlapVolume, 1.0f, leafSize);

	} else if (aggregateType == AggregateType::Sbvh_SurfaceArea) {
		sbvh = std::make_shared<Sbvh<DIM>>(primitives, CostHeuristic::SurfaceArea, 1e-5, leafSize);

	} else if (aggregateType == AggregateType::Sbvh_Volume) {
		sbvh = std::make_shared<Sbvh<DIM>>(primitives, CostHeuristic::Volume, 1e-5, leafSize);

	} else {
		return std::make_shared<Baseline<DIM>>(primitives);
	}

#ifdef BUILD_ENOKI
	if (vectorize) return std::make_shared<Mbvh<SIMD_WIDTH, DIM>>(sbvh);
#endif

	return sbvh;
}

template <int DIM>
inline void Scene<DIM>::buildAggregate(const AggregateType& aggregateType, bool vectorize)
{
	// initialize instances and aggregate
	aggregate = nullptr;
	objectInstances.clear();

	// build object aggregates
	int nObjects = (int)objects.size();
	std::vector<std::shared_ptr<Aggregate<DIM>>> objectAggregates(nObjects);

	for (int i = 0; i < nObjects; i++) {
		objectAggregates[i] = makeAggregate<DIM>(aggregateType, vectorize, objects[i]);
	}

	// build object instances
	for (int i = 0; i < nObjects; i++) {
		int nObjectInstances = (int)instanceTransforms[i].size();

		if (nObjectInstances == 0) {
			objectInstances.emplace_back(objectAggregates[i]);

		} else {
			for (int j = 0; j < nObjectInstances; j++) {
				objectInstances.emplace_back(std::make_shared<TransformedAggregate<DIM>>(
											 objectAggregates[i], instanceTransforms[i][j]));
			}
		}
	}

	// set aggregate
	if (objectInstances.size() == 1) {
		// set to object aggregate if there is only a single object instance in the scene
		aggregate = objectAggregates[0];

	} else if (csgTree.size() > 0) {
		// build csg aggregate if csg tree is specified
		aggregate = buildCsgAggregateRecursive<DIM>(0, csgTree, objectInstances);

	} else {
		// make aggregate
		aggregate = makeAggregate<DIM>(aggregateType, vectorize, objectInstances);
	}
}

#ifdef BENCHMARK_EMBREE
template <int DIM>
inline void Scene<DIM>::buildEmbreeAggregate()
{
	int nObjects = (int)objects.size();
	if (nObjects > 1) LOG(FATAL) << "Scene::buildEmbreeAggregate(): Not supported for multiple objects";

	aggregate = std::make_shared<EmbreeBvh<DIM>>(objects[0], soups[0]);
	objectInstances.clear();
}
#endif

} // namespace fcpw
