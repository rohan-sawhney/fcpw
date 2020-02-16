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

template <int DIM>
void generateScatteredPointsAndRays(std::vector<fcpw::Vector<DIM>>& scatteredPoints,
									std::vector<fcpw::Vector<DIM>>& randomDirections,
									const BoundingBox<DIM>& boundingBox)
{
	fcpw::Vector<DIM> e = boundingBox.extent();
	fcpw::Vector<DIM> o = fcpw::Vector<DIM>::Zero();
	fcpw::Vector<DIM> d = fcpw::Vector<DIM>::Zero();

	for (int i = 0; i < nPoints; i++) {
		for (int j = 0; j < DIM; j++) {
			o(j) = boundingBox.pMin(j) + e(j)*uniformRealRandomNumber();
			d(j) = uniformRealRandomNumber(-1.0f, 1.0f);
		}

		scatteredPoints.emplace_back(o);
		randomDirections.emplace_back(d.normalized());
	}
}

/*
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
*/
template <int DIM>
void visualizeScene(const Scene<DIM>& scene,
					std::vector<fcpw::Vector<DIM>>& queryPoints,
					std::vector<fcpw::Vector<DIM>>& randomDirections)
{
	// set a few options
	polyscope::options::programName = "Aggregate Tests";
	polyscope::options::verbosity = 0;
	polyscope::options::usePrefsFile = false;
	polyscope::options::autocenterStructures = false;

	// initialize polyscope
	polyscope::init();

	// register point cloud
	polyscope::registerPointCloud("Query_Points", queryPoints);

	if (DIM == 3) {
		// register surface meshes
		for (int i = 0; i < (int)scene.soups.size(); i++) {
			polyscope::registerSurfaceMesh("Polygon_Soup_" + std::to_string(i),
										   scene.soups[i]->positions, scene.soups[i]->indices);
		}

		// add direction vectors
		polyscope::getPointCloud("Query_Points")->addVectorQuantity("Random_Directions", randomDirections);
	}

	// give control to polyscope gui
	polyscope::show();
}

template <int DIM>
void run()
{
	// build baseline scene
	Scene<DIM> baselineScene;
	baselineScene.loadFiles(true, false);
	baselineScene.buildAggregate(AggregateType::Baseline);

	// generate random points and rays used to visualize csg
	BoundingBox<DIM> boundingBox = baselineScene.aggregate->boundingBox();
	std::vector<fcpw::Vector<DIM>> queryPoints, randomDirections;
	generateScatteredPointsAndRays<DIM>(queryPoints, randomDirections, boundingBox);

	if (vizScene) {
		visualizeScene<DIM>(baselineScene, queryPoints, randomDirections);

	} else {
		// build bvh scene
		Scene<DIM> bvhScene;
		bvhScene.loadFiles(true, false);
		bvhScene.buildAggregate(AggregateType::Bvh);

#ifdef BENCHMARK_EMBREE
		// build embree bvh scene
		Scene<DIM> embreeBvhScene;
		embreeBvhScene.loadFiles(true, false);
		embreeBvhScene.buildEmbreeAggregate();
#endif

		// TODO:
		// checkCorrectness
		// checkPerformance
	}
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
