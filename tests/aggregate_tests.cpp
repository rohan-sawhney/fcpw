#include "utilities/scene.h"
#include <ThreadPool.h>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include "args/args.hxx"

using namespace fcpw;

static bool vizScene = false;
static bool checkCorrectness = true;
static bool checkPerformance = true;
static int nPoints = 10000;
static progschj::ThreadPool pool;
static int nThreads = 8;

/*
template <int DIM>
std::shared_ptr<PointCloud<DIM>> generateScatteredPointsAndRays(
								std::vector<VectorXf>& rayDirections,
								const BoundingBox<DIM>& boundingBox)
{
	std::shared_ptr<PointCloud<DIM>> cloud = std::make_shared<PointCloud<DIM>>();
	cloud->points.reserve(nSamples);
	rayDirections.reserve(nSamples);
	VectorXf e = boundingBox.extent();
	VectorXf o = VectorXf::Zero(DIM);
	VectorXf d = VectorXf::Zero(DIM);

	for (int i = 0; i < nSamples; i++) {
		for (int j = 0; j < DIM; j++) {
			o(j) = boundingBox.pMin(j) + e(j)*uniformRealRandomNumber();
			d(j) = uniformRealRandomNumber(-1.0f, 1.0f);
		}

		d.normalize();
		cloud->points.emplace_back(o);
		rayDirections.emplace_back(d);
	}

	return cloud;
}

template <int DIM>
std::shared_ptr<DynamicKdTree<DIM>> constructKdTree(const std::shared_ptr<PointCloud<DIM>>& cloud)
{
	std::shared_ptr<DynamicKdTree<DIM>> kdTree = std::make_shared<DynamicKdTree<DIM>>(cloud);
	int pCurrent = 0;
	int pRange = std::max(100, (int)nSamples/nThreads);

	// dynamically add points to kdtree
	while (pCurrent < nSamples) {
		int pEnd = std::min(nSamples, pCurrent + pRange);
		kdTree->addPoints(pCurrent, pEnd);

		pCurrent += pRange;
	}

	return kdTree;
}

std::shared_ptr<PolygonSoup<2>> constructBezierBaselines(const std::vector<std::vector<std::shared_ptr<Primitive<2>>>>& primitives,
														 std::vector<std::shared_ptr<Primitive<2>>>& bezierPrimitives,
														 std::vector<std::shared_ptr<Primitive<2>>>& tesselatedPrimitives,
														 std::shared_ptr<Aggregate<2>>& bezierBaseline,
														 std::shared_ptr<Aggregate<2>>& tesselatedBaseline,
														 int granularity)
{
	std::vector<std::shared_ptr<Shape<2>>> beziers, tesselatedBeziers;

	// collect beziers
	for (int i = 0; i < (int)primitives.size(); i++) {
		for (int j = 0; j < (int)primitives[i].size(); j++) {
			GeometricPrimitive<2, float> *geometricPrim = static_cast<GeometricPrimitive<2, float> *>(primitives[i][j].get());
			std::shared_ptr<Shape<2>> shape = geometricPrim->getShape();
			const Bezier *bezier = dynamic_cast<const Bezier *>(shape.get());

			if (bezier) beziers.emplace_back(shape);
		}
	}

	if (beziers.size() > 0) {
		// tesselate beziers
		std::shared_ptr<PolygonSoup<2>> tesselatedSoup = tesselateBeziers(beziers, tesselatedBeziers, granularity);

		// construct baselines
		for (int i = 0; i < (int)beziers.size(); i++) {
			bezierPrimitives.emplace_back(std::make_shared<GeometricPrimitive<2, float>>(beziers[i], nullptr));
		}

		for (int i = 0; i < (int)tesselatedBeziers.size(); i++) {
			tesselatedPrimitives.emplace_back(std::make_shared<GeometricPrimitive<2, float>>(tesselatedBeziers[i], nullptr));
		}

		bezierBaseline = std::make_shared<BaselineAggregate<2>>(bezierPrimitives);
		tesselatedBaseline = std::make_shared<BaselineAggregate<2>>(tesselatedPrimitives);

		return tesselatedSoup;
	}

	return nullptr;
}

template <int DIM>
void testIntersectionQueries(const std::shared_ptr<PointCloud<DIM>>& cloud,
							 const std::vector<VectorXf>& rayDirections,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate1,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate2,
							 int start, int end)
{
	// compute and compare baseline and bvh interactions
	for (int i = start; i < end; i++) {
		std::vector<Interaction<DIM>> c1;
		Ray<DIM> r1(cloud->points[i], rayDirections[i]);
		bool hit1 = (bool)aggregate1->intersect(r1, c1);

		std::vector<Interaction<DIM>> c2;
		Ray<DIM> r2(cloud->points[i], rayDirections[i]);
		bool hit2 = (bool)aggregate2->intersect(r2, c2);

		if ((hit1 != hit2) || (hit1 && hit2 && c1[0] != c2[0])) {
			LOG(INFO) << "d1: " << c1[0].d << " d2: " << c2[0].d;
			LOG(INFO) << "p1: " << c1[0].p << " p2: " << c2[0].p;
			LOG(FATAL) << "Intersections do not match!";
		}

		std::vector<Interaction<DIM>> c3;
		Ray<DIM> r3(cloud->points[i], rayDirections[i]);
		int hit3 = aggregate1->intersect(r3, c3, false, true);

		std::vector<Interaction<DIM>> c4;
		Ray<DIM> r4(cloud->points[i], rayDirections[i]);
		int hit4 = aggregate2->intersect(r4, c4, false, true);

		if (hit3 != hit4) {
			LOG(FATAL) << "Number of intersections do not match!"
					   << " Aggregate1: " << hit3
					   << " Aggregate2: " << hit4;
		}
	}
}

template <int DIM>
void testClosestPointQueries(const std::shared_ptr<PointCloud<DIM>>& cloud,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate1,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate2,
							 int start, int end)
{
	// compute and compare baseline and bvh interactions
	for (int i = start; i < end; i++) {
		Interaction<DIM> c1;
		BoundingSphere<DIM> s1(cloud->points[i], maxFloat);
		bool found1 = aggregate1->findClosestPoint(s1, c1);

		Interaction<DIM> c2;
		BoundingSphere<DIM> s2(cloud->points[i], maxFloat);
		bool found2 = aggregate2->findClosestPoint(s2, c2);

		if (found1 != found2 || c1 != c2) {
			LOG(INFO) << "d1: " << c1.d << " d2: " << c2.d;
			LOG(INFO) << "p1: " << c1.p << " p2: " << c2.p;
			LOG(FATAL) << "Closest points do not match!";
		}
	}
}

template <int DIM>
void performKnnQueries(const std::shared_ptr<PointCloud<DIM>>& cloud,
					   const std::shared_ptr<DynamicKdTree<DIM>>& kdTree,
					   int neighbors, int start, int end)
{
	int N = end - start;
	std::vector<std::vector<std::pair<size_t, float>>> resultSet(N);

	for (int i = start; i < end; i++) {
		VectorXf p = cloud->points[i];
		kdTree->findNeighbors(neighbors, p, resultSet[i - start]);
	}
}

template <int DIM>
void performRNNQueries(const std::shared_ptr<PointCloud<DIM>>& cloud,
					   const std::shared_ptr<DynamicKdTree<DIM>>& kdTree,
					   float radius, int start, int end)
{
	int N = end - start;
	std::vector<std::vector<std::pair<size_t, float>>> resultSet(N);

	for (int i = start; i < end; i++) {
		VectorXf p = cloud->points[i];
		kdTree->findNeighbors(radius, p, resultSet[i - start]);
	}
}

template <int DIM>
void testBeziers(const std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
				 const std::shared_ptr<PointCloud<DIM>>& cloud,
				 const std::vector<VectorXf>& rayDirections)
{
	// do nothing
}

template <>
void testBeziers<2>(const std::vector<std::vector<std::shared_ptr<Primitive<2>>>>& primitives,
					const std::shared_ptr<PointCloud<2>>& cloud,
					const std::vector<VectorXf>& rayDirections)
{
	// construct baselines
	std::vector<std::shared_ptr<Primitive<2>>> bezierPrimitives, tesselatedPrimitives;
	std::shared_ptr<Aggregate<2>> bezierBaseline, tesselatedBaseline;
	std::shared_ptr<PolygonSoup<2>> tesselatedSoup = constructBezierBaselines(primitives, bezierPrimitives,
																			  tesselatedPrimitives, bezierBaseline,
																			  tesselatedBaseline, 1000);
	if (tesselatedSoup != nullptr) {
		// compare intersections and closest points
		int pCurrent = 0;
		int pRange = std::max(100, (int)nSamples/nThreads);

		while (pCurrent < nSamples) {
			int pEnd = std::min(nSamples, pCurrent + pRange);
			pool.enqueue([&cloud, &rayDirections, &bezierBaseline, &tesselatedBaseline, pCurrent, pEnd]() {
				#ifdef PROFILE
					PROFILE_THREAD_SCOPED();
				#endif

				testIntersectionQueries<2>(cloud, rayDirections, bezierBaseline, tesselatedBaseline, pCurrent, pEnd);
				testClosestPointQueries<2>(cloud, bezierBaseline, tesselatedBaseline, pCurrent, pEnd);
			});

			pCurrent += pRange;
		}

		pool.wait_until_empty();
		pool.wait_until_nothing_in_flight();
	}
}

template <int DIM>
void testAggregates(const std::shared_ptr<PointCloud<DIM>>& cloud,
					const std::vector<VectorXf>& rayDirections,
					const std::shared_ptr<DynamicKdTree<DIM>>& kdTree,
					const std::shared_ptr<Aggregate<DIM>>& baseline,
					const std::shared_ptr<Aggregate<DIM>>& bvh)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nSamples/nThreads);

	while (pCurrent < nSamples) {
		int pEnd = std::min(nSamples, pCurrent + pRange);
		pool.enqueue([&cloud, &rayDirections, &kdTree, &baseline, &bvh, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			testIntersectionQueries<DIM>(cloud, rayDirections, baseline, bvh, pCurrent, pEnd);
			testClosestPointQueries<DIM>(cloud, baseline, bvh, pCurrent, pEnd);
			performKnnQueries<DIM>(cloud, kdTree, 5, pCurrent, pEnd);
			performRNNQueries<DIM>(cloud, kdTree, 0.1f, pCurrent, pEnd);
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();
}

template <int DIM>
void visualizeScene(const std::vector<std::shared_ptr<PolygonSoup<DIM>>>& soups,
					const std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
					const std::shared_ptr<PointCloud<DIM>>& cloud,
					const std::vector<VectorXf>& rayDirections)
{
	// set a few options
	polyscope::options::programName = "Geometry Tests";
	polyscope::options::verbosity = 0;
	polyscope::options::usePrefsFile = false;
	polyscope::options::autocenterStructures = false;

	// initialize polyscope
	polyscope::init();

	if (DIM == 2) {
		// set the camera to 2D mode (see below)
		polyscope::view::style = polyscope::view::NavigateStyle::Planar;

		// register curve networks
		for (int i = 0; i < (int)soups.size(); i++) {
			std::string meshName = "Polygon_Soup_" + std::to_string(i);

			if (isBezierSoup.find(i) != isBezierSoup.end()) {
				std::shared_ptr<PolygonSoup<DIM>> tesselatedSoup = tesselateBeziers<float>(primitives[i], 100);
				polyscope::registerCurveNetwork2D(meshName, tesselatedSoup->positions, tesselatedSoup->indices);

			} else {
				polyscope::registerCurveNetwork2D(meshName, soups[i]->positions, soups[i]->indices);
			}
		}

		// register point cloud
		polyscope::registerPointCloud2D("Point_Cloud", cloud->points);
		polyscope::getPointCloud("Point_Cloud")->addVectorQuantity2D("Rays", rayDirections);

	} else if (DIM == 3) {
		// register surface meshes
		for (int i = 0; i < (int)soups.size(); i++) {
			polyscope::registerSurfaceMesh("Polygon_Soup_" + std::to_string(i),
										   soups[i]->positions, soups[i]->indices);
		}

		// register point cloud and rays
		polyscope::registerPointCloud("Point_Cloud", cloud->points);
		polyscope::getPointCloud("Point_Cloud")->addVectorQuantity("Rays", rayDirections);
	}

	// give control to polyscope gui
	polyscope::show();
}
*/
template <int DIM>
void run()
{
	// build scene
	Scene<DIM> scene;
	scene.loadFiles(true, false);

	/*
	// build scene
	std::vector<std::shared_ptr<PolygonSoup<DIM>>> soups;
	std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>> primitives;
	buildSoupScene<DIM, float>(soups, primitives, nullptr, nullptr, true);

	// build aggregates
	std::shared_ptr<Aggregate<DIM>> baseline = buildAggregate<DIM>(primitives, "Baseline");
	std::shared_ptr<Aggregate<DIM>> bvh = buildAggregate<DIM>(primitives, "Bvh");

	// generate point cloud
	std::vector<VectorXf> rayDirections;
	std::shared_ptr<PointCloud<DIM>> cloud = generateScatteredPointsAndRays<DIM>(
										rayDirections, baseline->boundingBox());

	// build kd tree
	std::shared_ptr<DynamicKdTree<DIM>> kdTree = constructKdTree<DIM>(cloud);

	// run tests or visualize scene
	if (runTests) {
		testBeziers<DIM>(primitives, cloud, rayDirections);
		testAggregates<DIM>(cloud, rayDirections, kdTree, baseline, bvh);

	} else if (vizScene) {
		visualizeScene<DIM>(soups, primitives, cloud, rayDirections);
	}
	*/
}

int main(int argc, const char *argv[]) {
	google::InitGoogleLogging(argv[0]);
#ifdef PROFILE
	Profiler::detect(argc, argv);
#endif
	// configure the argument parser
	args::ArgumentParser parser("aggregate tests");
	args::Group group(parser, "", args::Group::Validators::DontCare);
	args::Flag vizScene(group, "bool", "visualize scene", {"vizScene"});
	args::Flag checkCorrectness(group, "bool", "check aggregate correctness", {"checkCorrectness"});
	args::Flag checkPerformance(group, "bool", "check aggregate performance", {"checkPerformance"});
	args::ValueFlag<int> dim(parser, "integer", "scene dimension", {"dim"});
	args::ValueFlag<int> nPoints(parser, "integer", "nPoints", {"nPoints"});
	args::ValueFlag<int> nThreads(parser, "integer", "nThreads", {"nThreads"});
	args::ValueFlagList<std::string> triangleFilenames(parser, "string", "triangle soup filenames", {"tFile"});

	// parse args
	try {
		parser.ParseCLI(argc, argv);

	} catch (const args::Help&) {
		std::cout << parser;
		return 0;

	} catch (const args::ParseError& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return 1;
	}

	int DIM = args::get(dim);
	if (!dim) {
		std::cerr << "Please specify dimension" << std::endl;
		return EXIT_FAILURE;

	} else {
		if (DIM != 3) {
			std::cerr << "Only dimension 3 is supported at this moment" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (!triangleFilenames) {
		std::cerr << "Please specify triangle soup filenames" << std::endl;
		return EXIT_FAILURE;
	}

	// set global flags
	if (vizScene) ::vizScene = args::get(vizScene);
	if (checkCorrectness) ::checkCorrectness = args::get(checkCorrectness);
	if (checkPerformance) ::checkPerformance = args::get(checkPerformance);
	if (nPoints) ::nPoints = args::get(nPoints);
	if (nThreads) ::nThreads = args::get(nThreads);
	if (triangleFilenames) {
		for (const auto tsf: args::get(triangleFilenames)) {
			files.emplace_back(std::make_pair(tsf, LoadingOption::ObjTriangles));
		}
	}

	// run app
	if (DIM == 3) run<3>();

#ifdef PROFILE
	Profiler::dumphtml();
#endif
	return 0;
}
