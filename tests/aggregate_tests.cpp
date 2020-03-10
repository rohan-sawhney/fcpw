#include "utilities/scene.h"
#include <ThreadPool.h>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include "args/args.hxx"

using namespace fcpw;
using namespace std::chrono;

static bool vizScene = false;
static bool plotInteriorPoints = false;
static bool checkCorrectness = false;
static bool checkPerformance = false;
static int nQueries = 10000;
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

	for (int i = 0; i < nQueries; i++) {
		for (int j = 0; j < DIM; j++) {
			o(j) = boundingBox.pMin(j) + e(j)*uniformRealRandomNumber();
			d(j) = uniformRealRandomNumber(-1.0f, 1.0f);
		}

		scatteredPoints.emplace_back(o);
		randomDirections.emplace_back(d.normalized());
	}
}

template <int DIM>
void timeIntersectionQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate,
							 const std::vector<fcpw::Vector<DIM>>& rayOrigins,
							 const std::vector<fcpw::Vector<DIM>>& rayDirections,
							 const std::string& aggregateType)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate, &rayOrigins, &rayDirections, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				// perform intersection query
				std::vector<Interaction<DIM>> cs;
				Ray<DIM> r(rayOrigins[i], rayDirections[i]);
				int hit = aggregate->intersect(r, cs);
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << rayOrigins.size() << " intersection queries took "
			  << timeSpan.count() << " seconds with "
			  << aggregateType << " aggregate" << std::endl;
}

template <int DIM>
void timeClosestPointQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate,
							 const std::vector<fcpw::Vector<DIM>>& queryPoints,
							 const std::string& aggregateType)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate, &queryPoints, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				// perform closest point query
				Interaction<DIM> c;
				BoundingSphere<DIM> s(queryPoints[i], maxFloat);
				bool found = aggregate->findClosestPoint(s, c);
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << queryPoints.size() << " closest point queries took "
			  << timeSpan.count() << " seconds with "
			  << aggregateType << " aggregate" << std::endl;
}

template <int DIM>
void testIntersectionQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate1,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate2,
							 const std::vector<fcpw::Vector<DIM>>& rayOrigins,
							 const std::vector<fcpw::Vector<DIM>>& rayDirections)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate1, &aggregate2, &rayOrigins, &rayDirections, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				std::vector<Interaction<DIM>> c1;
				Ray<DIM> r1(rayOrigins[i], rayDirections[i]);
				bool hit1 = (bool)aggregate1->intersect(r1, c1);

				std::vector<Interaction<DIM>> c2;
				Ray<DIM> r2(rayOrigins[i], rayDirections[i]);
				bool hit2 = (bool)aggregate2->intersect(r2, c2);

				if ((hit1 != hit2) || (hit1 && hit2 && c1[0] != c2[0])) {
					LOG(INFO) << "d1: " << c1[0].d << " d2: " << c2[0].d;
					LOG(INFO) << "p1: " << c1[0].p << " p2: " << c2[0].p;
					LOG(FATAL) << "Intersections do not match!";
				}

				std::vector<Interaction<DIM>> c3;
				Ray<DIM> r3(rayOrigins[i], rayDirections[i]);
				int hit3 = aggregate1->intersect(r3, c3, false, true);

				std::vector<Interaction<DIM>> c4;
				Ray<DIM> r4(rayOrigins[i], rayDirections[i]);
				int hit4 = aggregate2->intersect(r4, c4, false, true);

				if (hit3 != hit4) {
					LOG(FATAL) << "Number of intersections do not match!"
							   << " hits1: " << hit3
							   << " hits2: " << hit4;
				}
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();
}

template <int DIM>
void testClosestPointQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate1,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate2,
							 const std::vector<fcpw::Vector<DIM>>& queryPoints)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate1, &aggregate2, &queryPoints, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				Interaction<DIM> c1;
				BoundingSphere<DIM> s1(queryPoints[i], maxFloat);
				bool found1 = aggregate1->findClosestPoint(s1, c1);

				Interaction<DIM> c2;
				BoundingSphere<DIM> s2(queryPoints[i], maxFloat);
				bool found2 = aggregate2->findClosestPoint(s2, c2);

				if (found1 != found2 || c1 != c2) {
					Interaction<DIM> c3;
					BoundingSphere<DIM> s3(queryPoints[i], maxFloat);
					bool found3 = c2.primitive->findClosestPoint(s3, c3);
					LOG_IF(INFO, found2 != found3 || c2 != c3) << "d1: " << c1.d << " d2: " << c2.d << " d3: " << c3.d;
					LOG_IF(INFO, found2 != found3 || c2 != c3) << "p1: " << c1.p << " p2: " << c2.p << " p3: " << c3.p;
					LOG_IF(FATAL, found2 != found3 || c2 != c3) << "Found closest primitive on aggregate 2 doesn't match up to aggregate 2";
					LOG(INFO) << "d1: " << c1.d << " d2: " << c2.d;
					LOG(INFO) << "p1: " << c1.p << " p2: " << c2.p;
					LOG(FATAL) << "Closest points do not match!";
				}
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();
}

template <int DIM>
void isolateInteriorPoints(const std::shared_ptr<Aggregate<DIM>>& aggregate,
						   const std::vector<fcpw::Vector<DIM>>& queryPoints,
						   std::vector<fcpw::Vector<DIM>>& interiorPoints)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);
	std::vector<bool> isInterior(nQueries, false);

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate, &queryPoints, &isInterior, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				if (aggregate->contains(queryPoints[i])) {
					isInterior[i] = true;
				}
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();

	for (int i = 0; i < nQueries; i++) {
		if (isInterior[i]) interiorPoints.emplace_back(queryPoints[i]);
	}
}

template <int DIM>
void visualizeScene(const Scene<DIM>& scene,
					const std::vector<fcpw::Vector<DIM>>& queryPoints,
					const std::vector<fcpw::Vector<DIM>>& randomDirections,
					const std::vector<fcpw::Vector<DIM>>& interiorPoints)
{
	// set a few options
	polyscope::options::programName = "Aggregate Tests";
	polyscope::options::verbosity = 0;
	polyscope::options::usePrefsFile = false;
	polyscope::options::autocenterStructures = false;

	// initialize polyscope
	polyscope::init();

	// register query points and interior points
	polyscope::registerPointCloud("Query_Points", queryPoints);
	if (plotInteriorPoints) polyscope::registerPointCloud("Interior_Points", interiorPoints);

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
	Scene<DIM> scene;
	scene.loadFiles(true, false);
	scene.buildAggregate(AggregateType::Baseline);

	// generate random points and rays used to visualize csg
	BoundingBox<DIM> boundingBox = scene.aggregate->boundingBox();
	std::vector<fcpw::Vector<DIM>> queryPoints, randomDirections;
	generateScatteredPointsAndRays<DIM>(queryPoints, randomDirections, boundingBox);

	std::vector<AggregateType> aggregateTypes({
		AggregateType::Baseline, 
		AggregateType::Bvh
		// AggregateType::Sbvh,
	});

	std::vector<std::string> aggregateNames({
		"Baseline",
		"Bvh",
		"Bvh with Surface Area Heuristic",
		"Bvh with Volume Heuristic",
		"Bvh with Overlap Surface Area Heuristic",
		"Bvh with Overlap Volume Heuristic",
		// "Sbvh",
		"Bvh using SSE"
	});

	std::vector<std::string> bvhNames({
		"Bvh with Midpoint Heuristic",
		"Bvh with Surface Area Heuristic",
		"Bvh with Volume Heuristic",
		"Bvh with Overlap Surface Area Heuristic",
		"Bvh with Overlap Volume Heuristic"
	});

	int startAggregate = 1;
	int endAggregate = aggregateTypes.size();

	bool doSIMDTests = true;
	const int bvhHeuristicStart = 4;
	const int bvhHeuristicEnd = 5;
	bool doSingleTiming = true;

	if (checkPerformance) {
		std::cout << "Running performance tests..." << std::endl;
		scene.setBinSize(32);
		scene.setLeafSize(4);
		scene.setSimdType(0);

		for(int i = startAggregate; i < endAggregate; i++){
			AggregateType aggregateType = aggregateTypes[i];
			switch(aggregateType){
				// benchmark queries
				case(AggregateType::Bvh):
					for(int j = bvhHeuristicStart; j < bvhHeuristicEnd; j++){
						scene.setSplitMethod(j);
						scene.setSimdType(0);
						scene.buildAggregate(aggregateType);
						if(doSingleTiming){
							timeIntersectionQueries<DIM>(scene.aggregate, queryPoints, randomDirections, bvhNames[j]);
							timeClosestPointQueries<DIM>(scene.aggregate, queryPoints, bvhNames[j]);
						}
						std::cout << std::endl;
						if(doSIMDTests){
							scene.setSimdType(2);
							scene.buildAggregate(aggregateType);
							timeClosestPointQueries<DIM>(scene.aggregate, queryPoints, bvhNames[j] + " with SSE");
							std::cout << std::endl;
						}
					}
					break;
				default:
					scene.buildAggregate(aggregateType);
					timeIntersectionQueries<DIM>(scene.aggregate, queryPoints, randomDirections, "Baseline");
					timeClosestPointQueries<DIM>(scene.aggregate, queryPoints, "Baseline");
					break;
			}
		}

// #ifdef BENCHMARK_EMBREE
// 		// build embree bvh aggregate & benchmark queries
// 		scene.buildEmbreeAggregate();
// 		timeIntersectionQueries<DIM>(scene.aggregate, queryPoints, randomDirections, "Embree Bvh");
// 		timeClosestPointQueries<DIM>(scene.aggregate, queryPoints, "Embree Bvh");
// #endif
	}

	if (checkCorrectness) {
		std::cout << "Running correctness tests..." << std::endl;

		// build baseline aggregate
		scene.buildAggregate(AggregateType::Baseline);

		for(int i = startAggregate; i < endAggregate; i++){
			// build bvh aggregate and compare results with baseline
			Scene<DIM> bvhScene;
			bvhScene.loadFiles(true, false);
			AggregateType aggregateType = aggregateTypes[i];
			switch(aggregateType){
				// benchmark queries
				case(AggregateType::Bvh):
					for(int j = bvhHeuristicStart; j < bvhHeuristicEnd; j++){
						std::cout << "Testing " << bvhNames[i] << " results against Baseline" << std::endl;
						bvhScene.setSplitMethod(j);
						bvhScene.setSimdType(0);
						bvhScene.buildAggregate(aggregateType);
						testIntersectionQueries<DIM>(scene.aggregate, bvhScene.aggregate, queryPoints, randomDirections);
						testClosestPointQueries<DIM>(scene.aggregate, bvhScene.aggregate, queryPoints);
						std::cout << std::endl;
						if(doSIMDTests){
							std::cout << "Testing vectorized " << bvhNames[i] << " results against Baseline" << std::endl;
							bvhScene.setSimdType(1);
							bvhScene.buildAggregate(aggregateType);
							testClosestPointQueries<DIM>(scene.aggregate, bvhScene.aggregate, queryPoints);
							std::cout << std::endl;
						}
					}
					break;
				default:
					bvhScene.buildAggregate(aggregateType);
					testIntersectionQueries<DIM>(scene.aggregate, bvhScene.aggregate, queryPoints, randomDirections);
					testClosestPointQueries<DIM>(scene.aggregate, bvhScene.aggregate, queryPoints);
					break;
			}

			// std::cout << "Testing " << aggregateNames[i] << " results against Baseline" << std::endl;
			// Scene<DIM> bvhScene;
			// bvhScene.loadFiles(true, false);
			// bvhScene.buildAggregate(aggregateTypes[i]);
			// if(aggregateTypes[i] != AggregateType::Bvh)
			// 	testIntersectionQueries<DIM>(scene.aggregate, bvhScene.aggregate, queryPoints, randomDirections);
			// testClosestPointQueries<DIM>(scene.aggregate, bvhScene.aggregate, queryPoints);
			// std::cout << std::endl;
		}

#ifdef BENCHMARK_EMBREE
		// build embree bvh aggregate and compare results with baseline
		std::cout << "Testing Embree Bvh results against Baseline" << std::endl;
		Scene<DIM> embreeBvhScene;
		embreeBvhScene.loadFiles(true, false);
		embreeBvhScene.buildEmbreeAggregate();
		testIntersectionQueries<DIM>(scene.aggregate, embreeBvhScene.aggregate, queryPoints, randomDirections);
		testClosestPointQueries<DIM>(scene.aggregate, embreeBvhScene.aggregate, queryPoints);
#endif
	}

	if (vizScene) {
		// build bvh aggregate
		scene.buildAggregate(AggregateType::Bvh);

		// isolate interior points among query points
		std::vector<fcpw::Vector<DIM>> interiorPoints;
		if (plotInteriorPoints) isolateInteriorPoints<DIM>(scene.aggregate, queryPoints, interiorPoints);

		// visualize scene
		visualizeScene<DIM>(scene, queryPoints, randomDirections, interiorPoints);
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
	args::Flag plotInteriorPoints(group, "bool", "plot interior points", {"plotInteriorPoints"});
	args::Flag checkCorrectness(group, "bool", "check aggregate correctness", {"checkCorrectness"});
	args::Flag checkPerformance(group, "bool", "check aggregate performance", {"checkPerformance"});
	args::ValueFlag<int> dim(parser, "integer", "scene dimension", {"dim"});
	args::ValueFlag<int> nQueries(parser, "integer", "number of queries", {"nQueries"});
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
	if (plotInteriorPoints) ::plotInteriorPoints = args::get(plotInteriorPoints);
	if (checkCorrectness) ::checkCorrectness = args::get(checkCorrectness);
	if (checkPerformance) ::checkPerformance = args::get(checkPerformance);
	if (nQueries) ::nQueries = args::get(nQueries);
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
