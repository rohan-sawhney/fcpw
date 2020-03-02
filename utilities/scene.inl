#include "csg_node.h"
#include "accelerators/baseline.h"
#include "accelerators/bvh.h"
#include "accelerators/sbvh.h"
#include "accelerators/bvh_simd.h"
#ifdef BENCHMARK_EMBREE
	#include "accelerators/embree_bvh.h"
#endif
#include <fstream>
#include <sstream>

namespace fcpw {

template <int DIM>
inline std::shared_ptr<PolygonSoup<DIM>> readSoupFromFile(
				const std::string& filename, const LoadingOption& loadingOption,
				const Transform<float, DIM, Affine>& transform, bool computeWeightedNormals,
				std::vector<std::shared_ptr<Primitive<DIM>>>& primitives, ObjectType& objectType)
{
	LOG(FATAL) << "readSoupFromFile<DIM>(): Not implemented";
	return nullptr;
}

template <>
inline std::shared_ptr<PolygonSoup<3>> readSoupFromFile(
				const std::string& filename, const LoadingOption& loadingOption,
				const Transform<float, 3, Affine>& transform, bool computeWeightedNormals,
				std::vector<std::shared_ptr<Primitive<3>>>& primitives, ObjectType& objectType)
{
	if (loadingOption == LoadingOption::ObjTriangles) {
		objectType = ObjectType::Triangles;
		return readFromOBJFile(filename, transform, primitives, computeWeightedNormals);
	}

	LOG(FATAL) << "readSoupFromFile<3>(): Invalid loading option";
	return nullptr;
}

template <int DIM>
inline void loadInstanceTransforms(
				std::vector<std::vector<Transform<float, DIM, Affine>>>& instanceTransforms)
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

		Matrix<float, DIM + 1, DIM + 1> transform;
		for (int i = 0; i <= DIM; i++) {
			for (int j = 0; j <= DIM; j++) {
				ss >> transform(i, j);
			}
		}

		instanceTransforms[object].emplace_back(Transform<float, DIM, Affine>(transform));
	}

	// close file
	in.close();
}

struct CsgTreeNode {
	int child1, child2;
	bool isLeafChild1, isLeafChild2;
	BooleanOperation operation;
};

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
inline void Scene<DIM>::loadFiles(bool computeWeightedNormals, bool randomizeObjectTransforms)
{
	int nFiles = (int)files.size();
	soups.resize(nFiles);
	objects.resize(nFiles);
	instanceTransforms.resize(nFiles);
	objectTypes.resize(nFiles);

	// generate object space transforms
	Transform<float, DIM, Affine> Id = Transform<float, DIM, Affine>::Identity();
	std::vector<Transform<float, DIM, Affine>> objectTransforms(nFiles, Id);
	if (randomizeObjectTransforms) {
		for (int i = 0; i < nFiles; i++) {
			objectTransforms[i].prescale(uniformRealRandomNumber(0.1f, 1.0f))
							   .pretranslate(uniformRealRandomVector<DIM>());
		}
	}

	// load soups and primitives
	for (int i = 0; i < nFiles; i++) {
		soups[i] = readSoupFromFile<DIM>(files[i].first, files[i].second, objectTransforms[i],
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
inline void Scene<DIM>::buildAggregate(const AggregateType& aggregateType)
{
	// initialize instances and aggregate
	aggregate = nullptr;
	objectInstances.clear();

	// build object aggregates
	int nObjects = (int)objects.size();
	std::vector<std::shared_ptr<Aggregate<DIM>>> objectAggregates(nObjects);

	for (int i = 0; i < nObjects; i++) {
		switch(aggregateType){
			case AggregateType::Bvh:
				objectAggregates[i] = std::make_shared<Bvh<DIM>>(objects[i]);
				break;
			// heuristics
			case AggregateType::Bvh_SAH:
				objectAggregates[i] = std::make_shared<Bvh<DIM>>(objects[i], 4, 1);
				break;
			case AggregateType::Bvh_Vol:
				objectAggregates[i] = std::make_shared<Bvh<DIM>>(objects[i], 4, 2);
				break;
			case AggregateType::Bvh_Overlap_SAH:
				objectAggregates[i] = std::make_shared<Bvh<DIM>>(objects[i], 4, 3);
				break;
			case AggregateType::Bvh_Overlap_Vol:
				objectAggregates[i] = std::make_shared<Bvh<DIM>>(objects[i], 4, 4);
				break;
			// sbvh
			case AggregateType::Sbvh:
				objectAggregates[i] = std::make_shared<Sbvh<DIM>>(objects[i], 4, 0, 32, false);
				break;
			// SIMD parallelism
			case AggregateType::SSEBvh:
				(std::make_shared<Bvh<DIM>>(objects[i], 4, 4, 32, true))->convert(4, objectAggregates[i]);
				break;
			case AggregateType::AVXBvh:
				(std::make_shared<Bvh<DIM>>(objects[i], 8))->convert(8, objectAggregates[i]);
				break;
			case AggregateType::AVX512Bvh:
				(std::make_shared<Bvh<DIM>>(objects[i], 16))->convert(16, objectAggregates[i]);
				break;
			default:
				objectAggregates[i] = std::make_shared<Baseline<DIM>>(objects[i]);
				break;		
		}
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

	} else if (aggregateType == AggregateType::Bvh) {
		// build bvh aggregate
		aggregate = std::make_shared<Bvh<DIM>>(objectInstances);

	} else if (aggregateType == AggregateType::Sbvh) {
		// build sbvh aggregate
		aggregate = std::make_shared<Sbvh<DIM>>(objectInstances);
	} else {
		// build baseline aggregate
		aggregate = std::make_shared<Baseline<DIM>>(objectInstances);
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
