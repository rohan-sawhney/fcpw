#pragma once

#include "geometry/triangles.h"
#include "accelerators/bvh.h"
#include "accelerators/baseline.h"
#include <unordered_map>
#include <fstream>
#include <sstream>

// TODO: need more sophisticated file loading to prescribe transforms
namespace fcpw {

std::vector<std::string> triangleFilenames;
std::vector<std::string> lineSegmentFilenames;
std::vector<std::string> bezierFilenames;
std::vector<std::string> implicitFilenames;
std::unordered_map<int, bool> isBezierSoup;
std::unordered_map<int, std::string> isImplicit;

std::string csgFilename;
std::string instanceFilename;
std::string samplingStrategy;
std::string integratorStrategy;
std::string interpolationStrategy;

float epsilonShell = 1e-3;
float russianRouletteProb = 0.0f;
float bezierClosestPointPrecision = 1e-6;

enum class loadingOption {
	trianglesObj,
	lineSegmentsObj,
	lineSegmentsFromTrianglesObj,
	beziersSvg,
	implicits
};

template <int DIM>
std::shared_ptr<PolygonSoup<DIM>> readSoupFromFile(const std::string& filename, const loadingOption& option,
												   const Transform<float, DIM, Affine>& transform, bool computeSingularNormals,
												   std::vector<std::shared_ptr<Shape<DIM>>>& shapes, int soupIndex)
{
	LOG(FATAL) << "readSoupFromFile<DIM>(): Not implemented";
	return nullptr;
}

template <>
std::shared_ptr<PolygonSoup<2>> readSoupFromFile<2>(const std::string& filename, const loadingOption& option,
													const Transform<float, 2, Affine>& transform, bool computeVertexNormals,
													std::vector<std::shared_ptr<Shape<2>>>& shapes, int soupIndex)
{
	if (option == loadingOption::lineSegmentsObj) {
		return readFromOBJFile(filename, transform, shapes, computeVertexNormals);
	}

	if (option == loadingOption::lineSegmentsFromTrianglesObj) {
		return buildFromPolygonSoup(readFromOBJFile(filename, Transform<float, 3, Affine>::Identity()),
									transform, shapes, computeVertexNormals);
	}

	if (option == loadingOption::beziersSvg) {
		isBezierSoup[soupIndex] = true;
		return readFromSVGFile(filename, transform, shapes, bezierClosestPointPrecision, computeVertexNormals);
	}

	if (option == loadingOption::implicits) {
		return readFromImplicitsFile<2>(filename, transform, shapes, isImplicit[soupIndex]);
	}

	LOG(FATAL) << "readSoupFromFile<2>(): Invalid loading option";
	return nullptr;
}

template <>
std::shared_ptr<PolygonSoup<3>> readSoupFromFile<3>(const std::string& filename, const loadingOption& option,
													const Transform<float, 3, Affine>& transform, bool computeVertexEdgeNormals,
													std::vector<std::shared_ptr<Shape<3>>>& shapes, int soupIndex)
{
	if (option == loadingOption::trianglesObj) {
		return readFromOBJFile(filename, transform, shapes, computeVertexEdgeNormals);
	}

	if (option == loadingOption::implicits) {
		return readFromImplicitsFile<3>(filename, transform, shapes, isImplicit[soupIndex]);
	}

	LOG(FATAL) << "readSoupFromFile<3>(): Invalid loading option";
	return nullptr;
}

template <int DIM, typename T>
void readSoupsFromFiles(std::vector<std::shared_ptr<PolygonSoup<DIM>>>& soups,
						std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
						const std::shared_ptr<DirichletBoundaryCondition<DIM, T>>& dirichlet,
						const std::shared_ptr<DirichletBoundaryCondition<DIM, T>>& nestedDirichlet,
						const std::vector<Transform<float, DIM, Affine>>& transforms,
						const std::vector<std::string>& filenames,
						const loadingOption& option, bool computeSingularNormals)
{
	for (int i = 0; i < (int)filenames.size(); i++) {
		std::vector<std::shared_ptr<Shape<DIM>>> shapes;
		soups.emplace_back(readSoupFromFile<DIM>(filenames[i], option, transforms[i],
												 computeSingularNormals, shapes, soups.size()));
		primitives.emplace_back(std::vector<std::shared_ptr<Primitive<DIM>>>(shapes.size(), nullptr));

		for (int j = 0; j < (int)shapes.size(); j++) {
			primitives[soups.size() - 1][j] = std::make_shared<GeometricPrimitive<DIM, T>>(
															shapes[j], dirichlet, nestedDirichlet);
		}
	}
}

template <int DIM, typename T>
void buildSoupScene(std::vector<std::shared_ptr<PolygonSoup<DIM>>>& soups,
					std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
					const std::shared_ptr<DirichletBoundaryCondition<DIM, T>>& dirichlet,
					const std::shared_ptr<DirichletBoundaryCondition<DIM, T>>& nestedDirichlet=nullptr,
					bool randomizeTransforms=false, bool computeSingularNormals=false)
{
	int nT = (int)triangleFilenames.size();
	int nL = (int)lineSegmentFilenames.size();
	int nB = (int)bezierFilenames.size();
	int nI = (int)implicitFilenames.size();
	int nTotal = nT + nL + nB + nI;
	soups.reserve(nTotal);
	primitives.reserve(nTotal);
	Transform<float, DIM, Affine> Id = Transform<float, DIM, Affine>::Identity();

	// lambda to randomize transforms
	auto randomize = [](std::vector<Transform<float, DIM, Affine>>& transforms) -> void {
		for (int i = 0; i < (int)transforms.size(); i++) {
			transforms[i].prescale(uniformRealRandomNumber(0.1f, 1.0f))
						 .pretranslate(uniformRealRandomVector<DIM>());
		}
	};

	if (DIM == 2) {
		// load line segment soups from obj files
		std::vector<Transform<float, DIM, Affine>> transforms(nL, Id);
		if (randomizeTransforms) randomize(transforms);
		readSoupsFromFiles<DIM, T>(soups, primitives, dirichlet, nestedDirichlet, transforms, lineSegmentFilenames,
								   loadingOption::lineSegmentsObj, computeSingularNormals);

		// build line segment soups from triangle soups
		transforms.resize(nT, Id);
		if (randomizeTransforms) randomize(transforms);
		readSoupsFromFiles<DIM, T>(soups, primitives, dirichlet, nestedDirichlet, transforms, triangleFilenames,
								   loadingOption::lineSegmentsFromTrianglesObj, computeSingularNormals);

		// load bezier soups from svg files
		transforms.resize(nB, Id);
		if (randomizeTransforms) randomize(transforms);
		readSoupsFromFiles<DIM, T>(soups, primitives, dirichlet, nestedDirichlet, transforms, bezierFilenames,
								   loadingOption::beziersSvg, computeSingularNormals);

		// load implicits from files
		transforms.resize(nI, Id);
		if (randomizeTransforms) randomize(transforms);
		readSoupsFromFiles<DIM, T>(soups, primitives, dirichlet, nestedDirichlet, transforms, implicitFilenames,
								   loadingOption::implicits, computeSingularNormals);

	} else if (DIM == 3) {
		// load triangle soups from obj files
		std::vector<Transform<float, DIM, Affine>> transforms(nT, Id);
		if (randomizeTransforms) randomize(transforms);
		readSoupsFromFiles<DIM, T>(soups, primitives, dirichlet, nestedDirichlet, transforms, triangleFilenames,
								   loadingOption::trianglesObj, computeSingularNormals);

		// load implicits from files
		transforms.resize(nI, Id);
		if (randomizeTransforms) randomize(transforms);
		readSoupsFromFiles<DIM, T>(soups, primitives, dirichlet, nestedDirichlet, transforms, implicitFilenames,
								   loadingOption::implicits, computeSingularNormals);
	}
}

template <typename T>
std::shared_ptr<PolygonSoup<3>> tesselateBeziers(const std::vector<std::shared_ptr<Primitive<3>>>& primitives,
												 int granularity)
{
	LOG(FATAL) << "tesselateBeziers<T>(): Not implemented for dimension: 3";
	return nullptr;
}

template <typename T>
std::shared_ptr<PolygonSoup<2>> tesselateBeziers(const std::vector<std::shared_ptr<Primitive<2>>>& primitives,
												 int granularity)
{
	std::vector<std::shared_ptr<Shape<2>>> beziers, lineSegments;

	for (int i = 0; i < (int)primitives.size(); i++) {
		GeometricPrimitive<2, T> *geometricPrim = static_cast<GeometricPrimitive<2, T> *>(primitives[i].get());
		std::shared_ptr<Shape<2>> shape = geometricPrim->getShape();
		const Bezier *bezier = dynamic_cast<const Bezier *>(shape.get());

		if (bezier) beziers.emplace_back(shape);
	}

	return tesselateBeziers(beziers, lineSegments, granularity);
}

template <int DIM>
void loadInstanceTransforms(int nPrimitives, std::vector<std::vector<Transform<float, DIM, Affine>>>& instanceTransforms)
{
	// load scene
	std::ifstream in(instanceFilename);
	LOG_IF(FATAL, in.is_open() == false) << "Unable to open file: " << instanceFilename;
	instanceTransforms.resize(nPrimitives);

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		int primitive;
		ss >> primitive;

		Matrix<float, DIM + 1, DIM + 1> transform;
		for (int i = 0; i <= DIM; i++) {
			for (int j = 0; j <= DIM; j++) {
				ss >> transform(i, j);
			}
		}

		instanceTransforms[primitive].emplace_back(Transform<float, DIM, Affine>(transform));
	}

	// close file
	in.close();
}

template <int DIM>
void scaleImplicit(int index, float scale, std::shared_ptr<Shape<DIM>>& shape)
{
	LOG(FATAL) << "scaleImplicit(): not implemented for dimension: " << DIM;
}

template <>
void scaleImplicit<2>(int index, float scale, std::shared_ptr<Shape<2>>& shape)
{
	if (isImplicit[index] == "sphere") {
		static_cast<Sphere<2> *>(shape.get())->radius /= scale;
	}
}

template <>
void scaleImplicit<3>(int index, float scale, std::shared_ptr<Shape<3>>& shape)
{
	if (isImplicit[index] == "sphere") {
		static_cast<Sphere<3> *>(shape.get())->radius /= scale;

	} else if (isImplicit[index] == "double-torus") {
		static_cast<DoubleTorus *>(shape.get())->minorRadius /= scale;
		static_cast<DoubleTorus *>(shape.get())->majorRadius /= scale;
		static_cast<DoubleTorus *>(shape.get())->smoothing /= scale;
	}
}

template <int DIM, typename T>
void normalizeScene(std::vector<std::shared_ptr<PolygonSoup<DIM>>>& soups,
					const std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
					VectorXf& center, float& radius, bool rescaleToUnitRadius)
{
	// compute bounding box
	BoundingBox<DIM> bb;
	for (int i = 0; i < (int)primitives.size(); i++) {
		for (int j = 0; j < (int)primitives[i].size(); j++) {
			bb.expandToInclude(primitives[i][j]->boundingBox());
		}
	}

	// translate to origin
	center = bb.centroid();
	for (int i = 0; i < (int)soups.size(); i++) {
		for (int j = 0; j < (int)soups[i]->positions.size(); j++) {
			soups[i]->positions[j] -= center;
		}
	}

	if (rescaleToUnitRadius) {
		// recale to unit radius
		radius = bb.extent().norm()/2.0f;
		for (int i = 0; i < (int)soups.size(); i++) {
			for (int j = 0; j < (int)soups[i]->positions.size(); j++) {
				soups[i]->positions[j] /= radius;
			}
		}

	} else {
		radius = 1.0f;
	}

	// update primitive shapes
	for (int i = 0; i < (int)primitives.size(); i++) {
		for (int j = 0; j < (int)primitives[i].size(); j++) {
			GeometricPrimitive<DIM, T> *geometricPrim = static_cast<GeometricPrimitive<DIM, T> *>(primitives[i][j].get());
			std::shared_ptr<Shape<DIM>> shape = geometricPrim->getShape();
			if (isImplicit.find(i) != isImplicit.end()) scaleImplicit<DIM>(i, radius, shape);

			shape->update();
		}
	}
}

struct TreeNode{
	int child1, child2;
	bool isLeafChild1, isLeafChild2;
	BooleanOperation operation;
};

template <int DIM>
std::shared_ptr<Aggregate<DIM>> buildTree(int node, std::unordered_map<int, TreeNode>& tree,
										  std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
										  const std::string& aggregateStrategy)
{
	std::shared_ptr<Aggregate<DIM>> aggregate1, aggregate2;

	if (tree[node].isLeafChild1) {
		if (aggregateStrategy == "Bvh") aggregate1 = std::make_shared<Bvh<DIM>>(primitives[tree[node].child1]);
		else aggregate1 = std::make_shared<Baseline<DIM>>(primitives[tree[node].child1]);

	} else {
		aggregate1 = buildTree<DIM>(tree[node].child1, tree, primitives, aggregateStrategy);
	}

	if (tree[node].isLeafChild2) {
		if (aggregateStrategy == "Bvh") aggregate2 = std::make_shared<Bvh<DIM>>(primitives[tree[node].child2]);
		else aggregate2 = std::make_shared<Baseline<DIM>>(primitives[tree[node].child2]);

	} else {
		aggregate2 = buildTree<DIM>(tree[node].child2, tree, primitives, aggregateStrategy);
	}

	return std::make_shared<CsgNode<DIM>>(aggregate1, aggregate2, tree[node].operation);
}

template <int DIM>
std::shared_ptr<Aggregate<DIM>> buildCsgScene(std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
											  const std::string& aggregateStrategy)
{
	// load scene
	std::ifstream in(csgFilename);
	LOG_IF(FATAL, in.is_open() == false) << "Unable to open file: " << csgFilename;
	std::unordered_map<int, TreeNode> tree;

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		int node;
		ss >> node;

		std::string stringOperation, stringChild1, stringChild2;
		ss >> stringOperation >> stringChild1 >> stringChild2;

		std::size_t found1 = stringChild1.find_last_of("_");
		std::size_t found2 = stringChild2.find_last_of("_");
		tree[node].child1 = std::stoi(stringChild1.substr(found1 + 1));
		tree[node].child2 = std::stoi(stringChild2.substr(found2 + 1));
		tree[node].isLeafChild1 = stringChild1.find("node_") == std::string::npos;
		tree[node].isLeafChild2 = stringChild2.find("node_") == std::string::npos;
		tree[node].operation = stringOperation == "Union" ? BooleanOperation::Union :
							  (stringOperation == "Intersection" ? BooleanOperation::Intersection :
							  (stringOperation == "Difference" ? BooleanOperation::Difference : BooleanOperation::None));
	}

	// close file
	in.close();

	return buildTree<DIM>(0, tree, primitives, aggregateStrategy);
}

template <int DIM>
std::shared_ptr<Aggregate<DIM>> buildTransformedAggregates(
											   std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
											   std::vector<std::shared_ptr<Primitive<DIM>>>& transformedAggregates,
											   const std::vector<std::vector<Transform<float, DIM, Affine>>>& instanceTransforms,
											   const std::string& aggregateStrategy)
{
	for (int i = 0; i < (int)primitives.size(); i++) {
		std::shared_ptr<Aggregate<DIM>> aggregate = nullptr;
		if (aggregateStrategy == "Bvh") aggregate = std::make_shared<Bvh<DIM>>(primitives[i]);
		else aggregate = std::make_shared<Baseline<DIM>>(primitives[i]);

		for (int j = 0; j < (int)instanceTransforms[i].size(); j++) {
			transformedAggregates.emplace_back(std::make_shared<TransformedAggregate<DIM>>(aggregate, instanceTransforms[i][j]));
		}
	}

	std::shared_ptr<Aggregate<DIM>> aggregate = nullptr;
	if (aggregateStrategy == "Bvh") aggregate = std::make_shared<Bvh<DIM>>(transformedAggregates);
	else aggregate = std::make_shared<Baseline<DIM>>(transformedAggregates);

	return aggregate;
}

template <int DIM>
std::shared_ptr<Aggregate<DIM>> buildAggregate(std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
											   const std::string& aggregateStrategy, bool useRandomOperation=false,
											   BooleanOperation operation=BooleanOperation::None)
{
	if (!csgFilename.empty()) {
		return buildCsgScene<DIM>(primitives, aggregateStrategy);
	}

	// construct an aggregate for each set of primitives
	int N = (int)primitives.size();
	std::vector<std::shared_ptr<Aggregate<DIM>>> aggregates;
	std::vector<std::pair<Vector<DIM>, int>> centroids;
	BoundingBox<DIM> sceneBox;

	for (int i = 0; i < N; i++) {
		std::shared_ptr<Aggregate<DIM>> aggregate = nullptr;
		if (aggregateStrategy == "Bvh") aggregate = std::make_shared<Bvh<DIM>>(primitives[i]);
		else aggregate = std::make_shared<Baseline<DIM>>(primitives[i]);

		aggregates.emplace_back(aggregate);
		BoundingBox<DIM> aggregateBox = aggregate->boundingBox();
		centroids.emplace_back(std::make_pair(aggregateBox.centroid(), i));
		sceneBox.expandToInclude(aggregateBox);
	}

	// group aggregates by proximity
	int longestAxis = sceneBox.maxDimension();
	auto compareCentroids = [longestAxis](const std::pair<Vector<DIM>, int>& p1,
										  const std::pair<Vector<DIM>, int>& p2) -> bool {
		return p1.first(longestAxis) < p2.first(longestAxis);
	};

	std::sort(centroids.begin(), centroids.end(), compareCentroids);
	std::vector<std::shared_ptr<Aggregate<DIM>>> aggregatesTemp = aggregates;
	for (int i = 0; i < N; i++) aggregatesTemp[centroids[i].second] = aggregates[i];
	aggregates = aggregatesTemp;

	// build csg tree
	while (N != 1) {
		// club neighboring aggregates into CSG nodes
		std::vector<std::shared_ptr<Aggregate<DIM>>> nodes;
		for (int i = 0; i < N - 1; i += 2) {
			if (useRandomOperation) operation = static_cast<BooleanOperation>(uniformIntRandomNumber(0, 2));
			nodes.emplace_back(std::make_shared<CsgNode<DIM>>(aggregates[i], aggregates[i + 1], operation));
		}

		// if there are an odd number of aggregates, club the last node and aggregate
		if (N%2 == 1) {
			if (useRandomOperation) operation = static_cast<BooleanOperation>(uniformIntRandomNumber(0, 2));
			nodes[N/2 - 1] = std::make_shared<CsgNode<DIM>>(nodes[N/2 - 1], aggregates[N - 1], operation);
		}

		aggregates = nodes;
		N = (int)aggregates.size();
	}

	return aggregates[0];
}

std::shared_ptr<PolygonSoup<3>> buildGrid(int extent, int dim, float scale)
{
	// build dim x dim grid
	std::shared_ptr<PolygonSoup<3>> grid = std::make_shared<PolygonSoup<3>>();
	int nPositions = std::powf(extent + 1, dim);
	int nIndices = std::powf(extent, dim);
	grid->positions.reserve(nPositions);
	grid->indices.reserve(nIndices);
	int extentZ = dim == 2 ? 0 : extent;
	int shiftZ = 0;

	for (int d = 0; d < extentZ + 1; d++) {
		int shiftY = 0;

		for (int h = 0; h < extent + 1; h++) {
			for (int w = 0; w < extent + 1; w++) {
				grid->positions.emplace_back(Vector3f(w, h, d));

				if (dim == 2) {
					if (h != extent && w != extent) {
						int i = w + shiftY;
						int j = i + extent + 1;
						grid->indices.emplace_back(std::vector<int>{i, j, j + 1, i + 1});
					}

				} else {
					if (d != extentZ && h != extent && w != extent) {
						int i = w + shiftY + shiftZ;
						int j = i + extent + 1;
						int k = (extent + 1)*(extent + 1);
						grid->indices.emplace_back(std::vector<int>{i, j, j + 1, i + 1,
																	k + i + 1, k + j + 1,
																	k + j, k + i});
					}
				}
			}

			shiftY += extent + 1;
		}

		shiftZ += (extentZ + 1)*(extentZ + 1);
	}

	// compute grid center and scale
	int N = (int)grid->positions.size();
	Vector3f cm = Vector3f::Zero();
	float radius = 0.0f;

	// compute center of mass
	for (int i = 0; i < N; i++) {
		cm += grid->positions[i];
	}
	cm /= N;

	// translate to origin and compute radius
	for (int i = 0; i < N; i++) {
		grid->positions[i] -= cm;
		radius = std::max(radius, grid->positions[i].norm());
	}

	// rescale
	for (int i = 0; i < N; i++) {
		grid->positions[i] /= radius;
		grid->positions[i] *= scale;
	}

	return grid;
}

template <int DIM>
void sampleGrid(const std::shared_ptr<PolygonSoup<3>>& grid, int dim,
				std::shared_ptr<PointCloud<DIM>>& pointCloud)
{
	int N = (int)grid->indices.size();

	// sample points jittered around polygon centers
	for (int i = 0; i < N; i++) {
		VectorXf center = grid->polygonCenter(i);
		float w = (grid->positions[grid->indices[i][1]] - grid->positions[grid->indices[i][0]]).norm();
		float h = (grid->positions[grid->indices[i][3]] - grid->positions[grid->indices[i][2]]).norm();

		VectorXf sample = VectorXf::Zero(DIM);
		sample(0) = center(0) + uniformRealRandomNumber(-0.5f, 0.5f)*w;
		sample(1) = center(1) + uniformRealRandomNumber(-0.5f, 0.5f)*h;
		if (DIM == 3) {
			float d = dim == 3 ? (grid->positions[grid->indices[i][4]] - grid->positions[grid->indices[i][0]]).norm() : 0.0f;
			sample(2) = center(2) + uniformRealRandomNumber(-0.5f, 0.5f)*d;
		}

		pointCloud->points.emplace_back(sample);
	}
}

template <int DIM, typename T>
std::unique_ptr<Sampler<DIM>> constructSampler(std::shared_ptr<PointCloud<DIM>>& pointCloud,
											   const SamplingDomain<DIM>& samplingDomain,
											   const std::shared_ptr<SampleStatistics<DIM, T>>& statistics,
											   bool ignoreScene=false)
{
	std::unique_ptr<Sampler<DIM>> sampler = nullptr;

	if (samplingStrategy == "Random") {
		sampler = std::unique_ptr<RandomSampler<DIM>>(
			new RandomSampler<DIM>(pointCloud->points, samplingDomain, ignoreScene));

	} else if (samplingStrategy == "Poisson Disk") {
		sampler = std::unique_ptr<PoissonDiskSampler<DIM>>(
			new PoissonDiskSampler<DIM>(pointCloud->points, samplingDomain, ignoreScene, false));

	} else if (samplingStrategy == "Distance Weighted") {
		sampler = std::unique_ptr<PoissonDiskSampler<DIM>>(
			new PoissonDiskSampler<DIM>(pointCloud->points, samplingDomain, ignoreScene, true));

	} else if (samplingStrategy == "Adaptive") {
		sampler = std::unique_ptr<AdaptiveSampler<DIM, T>>(
			new AdaptiveSampler<DIM, T>(pointCloud, samplingDomain, statistics, ignoreScene));
	}

	return sampler;
}

template <int DIM, typename T>
void setGreensFunction(const std::string& greensFunctionType, std::shared_ptr<Medium<DIM, T>>& medium)
{
	if (integratorStrategy == "Walk on Spheres") {
		DomainMedium<DIM, BoundingSphere<DIM>, T> *domainMedium =
								static_cast<DomainMedium<DIM, BoundingSphere<DIM>, T> *>(medium.get());

		if (greensFunctionType == "YukawaBall") {
			domainMedium->greensFunction = std::make_shared<GreensFunctionYukawaBall<DIM>>();

		} else {
			domainMedium->greensFunction = std::make_shared<GreensFunctionLaplaceBall<DIM>>();
		}
	}
}

template <int DIM, typename T>
std::shared_ptr<Medium<DIM, T>> constructMedium(const Vector<DIM>& vDir, float vMag, float lambda,
					const std::vector<std::shared_ptr<InitialCondition<DIM, T>>>& initialConditions,
					const std::vector<std::shared_ptr<Source<DIM, T>>>& sources, int nSourceSamples=1,
					bool sampleSource=false, bool performMIS=false, const std::string& greensFunctionType="LaplaceBall")
{
	std::shared_ptr<Medium<DIM, T>> medium = nullptr;

	if (integratorStrategy == "Walk on Spheres") {
		medium = std::make_shared<DomainMedium<DIM, BoundingSphere<DIM>, T>>(
										vDir, vMag, lambda, nullptr, sources, initialConditions,
										nSourceSamples, sampleSource, performMIS);
		setGreensFunction<DIM, T>(greensFunctionType, medium);
	}

	return medium;
}

template <int DIM, typename T>
std::unique_ptr<Integrator<DIM, T>> constructIntegrator(const std::shared_ptr<PointCloud<DIM>>& pointCloud,
														const std::shared_ptr<Aggregate<DIM>>& aggregate,
														const std::shared_ptr<SampleStatistics<DIM, T>>& statistics,
														const std::shared_ptr<Medium<DIM, T>>& medium=nullptr,
														const std::shared_ptr<Medium<DIM, T>>& nestedMedium=nullptr,
														bool constructNested=false)
{
	std::unique_ptr<Integrator<DIM, T>> integrator = nullptr;

	if (integratorStrategy == "Walk on Spheres") {
		if (constructNested) {
			integrator = std::unique_ptr<NestedWalkOnSpheres<DIM, T>>(
				new NestedWalkOnSpheres<DIM, T>(pointCloud->points, aggregate,
												medium, nestedMedium, statistics,
												epsilonShell));

		} else {
			integrator = std::unique_ptr<WalkOnSpheres<DIM, T>>(
				new WalkOnSpheres<DIM, T>(pointCloud->points, aggregate, medium, statistics,
										  epsilonShell, russianRouletteProb));
		}
	}

	return integrator;
}

template <int DIM, typename T>
std::unique_ptr<Interpolator<DIM, T>> constructInterpolator(const std::shared_ptr<PointCloud<DIM>>& pointCloud,
															int dataDim)
{
	std::unique_ptr<Interpolator<DIM, T>> interpolator = nullptr;

	if (interpolationStrategy == "Moving Least Squares") {
		interpolator = std::unique_ptr<MovingLeastSquaresInterpolator<DIM, T>>(
			new MovingLeastSquaresInterpolator<DIM, T>(pointCloud, dataDim));
	}

	return interpolator;
}

template <int DIM, typename T>
void registerNewSamplesToInterpolate(std::unique_ptr<Interpolator<DIM, T>>& interpolator,
									 int nSamples)
{
	if (interpolationStrategy == "Moving Least Squares") {
		MovingLeastSquaresInterpolator<DIM, T> *mls =
			static_cast<MovingLeastSquaresInterpolator<DIM, T> *>(interpolator.get());
		mls->addNewSamples(nSamples);
	}
}

template <int DIM, typename T>
void interpolateOnSectionPlane(const std::shared_ptr<Aggregate<DIM>>& aggregate,
							   const std::unique_ptr<Interpolator<DIM, T>>& interpolator,
							   const std::vector<T>& solutionEstimates,
							   const std::vector<std::vector<T>>& gradientEstimates,
							   const std::shared_ptr<PolygonSoup<3>>& sectionPlane,
							   int neighbors, float radius, std::vector<T>& fFit,
							   T initVal, T minVal, bool checkContainment, bool rejectNonLineOfSight)
{
	int N = (int)sectionPlane->indices.size();
	int pCurrent = 0;
	int pRange = std::max(100, N/nThreads);
	fFit.resize(N, initVal);

	while (pCurrent < N) {
		int pEnd = std::min(N, pCurrent + pRange);
		pool.enqueue([&aggregate, &interpolator, &solutionEstimates, &gradientEstimates,
					  &sectionPlane, &fFit, &initVal, &minVal, neighbors, radius,
					  checkContainment, rejectNonLineOfSight, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				// interpolate
				VectorXf center = sectionPlane->polygonCenter(i).head<DIM>();
				if (!checkContainment || aggregate->contains(center)) {
					if (neighbors > 0) {
						fFit[i] = interpolator->fit(neighbors, center, solutionEstimates, gradientEstimates,
													initVal, rejectNonLineOfSight ? aggregate : nullptr);
					} else {
						fFit[i] = interpolator->fit(radius, center, solutionEstimates, gradientEstimates,
													initVal, rejectNonLineOfSight ? aggregate : nullptr);
					}

				} else {
					fFit[i] = minVal;
				}
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();
}

} // namespace fcpw
