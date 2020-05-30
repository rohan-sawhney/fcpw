#include "utilities/scene.h"
#include <ThreadPool.h>
#include <atomic>

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

// TODO:
// - too many nodes being visited while backtracking during closest point traversal
// - write timings to file
// - plot BVH scaling behavior with increasing mesh sizes

template<int DIM>
void splitBoxRecursive(BoundingBox<DIM> boundingBox,
					   std::vector<BoundingBox<DIM>>& boxes, int depth)
{
	if (depth == 0) {
		boxes.emplace_back(boundingBox);

	} else {
		int splitDim = boundingBox.maxDimension();
		float splitCoord = (boundingBox.pMin[splitDim] + boundingBox.pMax[splitDim])*0.5f;

		BoundingBox<DIM> boxLeft = boundingBox;
		boxLeft.pMax[splitDim] = splitCoord;
		splitBoxRecursive<DIM>(boxLeft, boxes, depth - 1);

		BoundingBox<DIM> boxRight = boundingBox;
		boxRight.pMin[splitDim] = splitCoord;
		splitBoxRecursive<DIM>(boxRight, boxes, depth - 1);
	}
}

template<int DIM>
void generateScatteredPointsAndRays(std::vector<Vector<DIM>>& scatteredPoints,
									std::vector<Vector<DIM>>& randomDirections,
									const BoundingBox<DIM>& boundingBox)
{
	// parition the scene bounding box into nThreads boxes
	std::vector<BoundingBox<DIM>> boxes;
	splitBoxRecursive<DIM>(boundingBox, boxes, 6);

	// generate queries in each box
	int nBoxes = (int)boxes.size();
	int nQueriesPerBox = std::ceil(nQueries/nBoxes);

	for (int i = 0; i < nBoxes; i++) {
		Vector<DIM> e = boxes[i].extent();

		for (int j = 0; j < nQueriesPerBox; j++) {
			Vector<DIM> o = boxes[i].pMin + cwiseProduct<DIM>(e, uniformRealRandomVector<DIM>());
			Vector<DIM> d = unit<DIM>(uniformRealRandomVector<DIM>(-1.0f, 1.0f));

			scatteredPoints.emplace_back(o);
			randomDirections.emplace_back(d);
		}
	}

	// resize if required
	scatteredPoints.resize(nQueries);
	randomDirections.resize(nQueries);
}

template<int DIM>
void timeIntersectionQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate,
							 const std::vector<Vector<DIM>>& rayOrigins,
							 const std::vector<Vector<DIM>>& rayDirections,
							 const std::vector<int>& indices,
							 const std::string& aggregateType,
							 bool queriesCoherent=false)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);
	std::atomic<int> totalNodesVisited(0);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate, &rayOrigins, &rayDirections, &indices,
					  &totalNodesVisited, queriesCoherent, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			int nodesVisitedByThread = 0;
			Interaction<DIM> cPrev;

			for (int i = pCurrent; i < pEnd; i++) {
				int I = indices[i];
				int nodeIndex = !queriesCoherent || cPrev.nodeIndex == -1 ? 0 : cPrev.nodeIndex;

				int nodesVisited = 0;
				std::vector<Interaction<DIM>> cs;
				Ray<DIM> r(rayOrigins[I], rayDirections[I]);
				bool hit = (bool)aggregate->intersectFromNode(r, cs, nodeIndex, nodesVisited);
				nodesVisitedByThread += nodesVisited;

				if (hit) cPrev = cs[0];
			}

			totalNodesVisited += nodesVisitedByThread;
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << rayOrigins.size() << " intersection queries"
			  << (queriesCoherent ? " with backtracking search" : "")
			  << " took " << timeSpan.count() << " seconds with "
			  << aggregateType << "; "
			  << (totalNodesVisited/nQueries) << " nodes visited on avg"
			  << std::endl;
}

template<int DIM>
void timeClosestPointQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate,
							 const std::vector<Vector<DIM>>& queryPoints,
							 const std::vector<int>& indices,
							 const std::string& aggregateType,
							 bool queriesCoherent=false)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);
	std::atomic<int> totalNodesVisited(0);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate, &queryPoints, &indices, &totalNodesVisited,
					  queriesCoherent, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			int nodesVisitedByThread = 0;
			Interaction<DIM> cPrev;
			Vector<DIM> queryPrev = zeroVector<DIM>();

			for (int i = pCurrent; i < pEnd; i++) {
				int I = indices[i];
				float distPrev = norm<DIM>(queryPoints[I] - queryPrev);
				float r2 = i == pCurrent ? maxFloat : std::pow(cPrev.d + distPrev, 2);
				int nodeIndex = !queriesCoherent || cPrev.nodeIndex == -1 ? 0 : cPrev.nodeIndex;

				int nodesVisited = 0;
				Interaction<DIM> c;
				BoundingSphere<DIM> s(queryPoints[I], r2);
				bool found = aggregate->findClosestPointFromNode(s, c, nodeIndex, nodesVisited);
				nodesVisitedByThread += nodesVisited;

				if (found) cPrev = c;
				else LOG(FATAL) << "Closest points not found!";
				queryPrev = queryPoints[I];
			}

			totalNodesVisited += nodesVisitedByThread;
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << queryPoints.size() << " closest point queries"
			  << (queriesCoherent ? " with backtracking search" : "")
			  << " took " << timeSpan.count() << " seconds with "
			  << aggregateType << "; "
			  << (totalNodesVisited/nQueries) << " nodes visited on avg"
			  << std::endl;
}

template<int DIM>
void testIntersectionQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate1,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate2,
							 const std::vector<Vector<DIM>>& rayOrigins,
							 const std::vector<Vector<DIM>>& rayDirections,
							 const std::vector<int>& indices,
							 bool queriesCoherent=false)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate1, &aggregate2, &rayOrigins, &rayDirections,
					  &indices, queriesCoherent, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			Interaction<DIM> cPrev;

			for (int i = pCurrent; i < pEnd; i++) {
				int I = indices[i];
				int nodeIndex = !queriesCoherent || cPrev.nodeIndex == -1 ? 0 : cPrev.nodeIndex;

				std::vector<Interaction<DIM>> c1;
				Ray<DIM> r1(rayOrigins[I], rayDirections[I]);
				bool hit1 = (bool)aggregate1->intersect(r1, c1);

				int nodesVisited = 0;
				std::vector<Interaction<DIM>> c2;
				Ray<DIM> r2(rayOrigins[I], rayDirections[I]);
				bool hit2 = (bool)aggregate2->intersectFromNode(r2, c2, nodeIndex, nodesVisited);

				if ((hit1 != hit2) || (hit1 && hit2 && c1[0] != c2[0])) {
					LOG(INFO) << "d1: " << c1[0].d << " d2: " << c2[0].d;
					LOG(INFO) << "p1: " << c1[0].p << " p2: " << c2[0].p;
					LOG(FATAL) << "Intersections do not match!";
				}

				if (hit2) cPrev = c2[0];

				std::vector<Interaction<DIM>> c3;
				Ray<DIM> r3(rayOrigins[I], rayDirections[I]);
				int hit3 = aggregate1->intersect(r3, c3, false, true);

				nodesVisited = 0;
				std::vector<Interaction<DIM>> c4;
				Ray<DIM> r4(rayOrigins[I], rayDirections[I]);
				int hit4 = aggregate2->intersectFromNode(r4, c4, 0, nodesVisited, false, true);

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

template<int DIM>
void testClosestPointQueries(const std::shared_ptr<Aggregate<DIM>>& aggregate1,
							 const std::shared_ptr<Aggregate<DIM>>& aggregate2,
							 const std::vector<Vector<DIM>>& queryPoints,
							 const std::vector<int>& indices,
							 bool queriesCoherent=false)
{
	int pCurrent = 0;
	int pRange = std::max(100, (int)nQueries/nThreads);

	while (pCurrent < nQueries) {
		int pEnd = std::min(nQueries, pCurrent + pRange);
		pool.enqueue([&aggregate1, &aggregate2, &queryPoints, &indices,
					  queriesCoherent, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			Interaction<DIM> cPrev;
			Vector<DIM> queryPrev = zeroVector<DIM>();

			for (int i = pCurrent; i < pEnd; i++) {
				int I = indices[i];
				float distPrev = norm<DIM>(queryPoints[I] - queryPrev);
				float r2 = i == pCurrent ? maxFloat : std::pow(cPrev.d + distPrev, 2);
				int nodeIndex = !queriesCoherent || cPrev.nodeIndex == -1 ? 0 : cPrev.nodeIndex;

				Interaction<DIM> c1;
				BoundingSphere<DIM> s1(queryPoints[I], maxFloat);
				bool found1 = aggregate1->findClosestPoint(s1, c1);

				int nodesVisited = 0;
				Interaction<DIM> c2;
				BoundingSphere<DIM> s2(queryPoints[I], r2);
				bool found2 = aggregate2->findClosestPointFromNode(s2, c2, nodeIndex, nodesVisited);

				if (found1 != found2 || std::fabs(c1.d - c2.d) > 1e-6) {
					LOG(INFO) << "d1: " << c1.d << " d2: " << c2.d;
					LOG(INFO) << "p1: " << c1.p << " p2: " << c2.p;
					LOG(FATAL) << "Closest points do not match!";
				}

				if (found2) cPrev = c2;
				else LOG(FATAL) << "Closest points not found!";
				queryPrev = queryPoints[I];
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();
}

template<int DIM>
void isolateInteriorPoints(const std::shared_ptr<Aggregate<DIM>>& aggregate,
						   const std::vector<Vector<DIM>>& queryPoints,
						   std::vector<Vector<DIM>>& interiorPoints)
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

template<int DIM>
void visualizeScene(const Scene<DIM>& scene,
					const std::vector<Vector<DIM>>& queryPoints,
					const std::vector<Vector<DIM>>& randomDirections,
					const std::vector<Vector<DIM>>& interiorPoints)
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

template<int DIM>
void run()
{
	// build baseline scene
	Scene<DIM> scene;
	scene.loadFiles(true);
	scene.buildAggregate(AggregateType::Baseline);

	// generate random points and rays used to visualize csg
	BoundingBox<DIM> boundingBox = scene.aggregate->boundingBox();
	std::vector<Vector<DIM>> queryPoints, randomDirections;
	generateScatteredPointsAndRays<DIM>(queryPoints, randomDirections, boundingBox);

	// generate indices
	std::vector<int> indices, shuffledIndices;
	for (int i = 0; i < nQueries; i++) {
		indices.emplace_back(i);
		shuffledIndices.emplace_back(i);
	}

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), std::default_random_engine(seed));

	std::vector<std::string> bvhTypes({"Bvh_LongestAxisCenter", "Bvh_SurfaceArea",
									   "Bvh_OverlapSurfaceArea", "Bvh_Volume",
									   "Bvh_OverlapVolume", "Sbvh_SurfaceArea",
									   "Sbvh_Volume"});

	if (checkPerformance) {
		std::cout << "Running performance tests..." << std::endl;

		// benchmark baseline queries
		//timeIntersectionQueries<DIM>(scene.aggregate, queryPoints, randomDirections,
		//							   shuffledIndices, "Baseline");
		//timeClosestPointQueries<DIM>(scene.aggregate, queryPoints,
		//							   shuffledIndices, "Baseline");

		// build bvh aggregates and benchmark queries
		for (int bvh = 1; bvh < 8; bvh++) {
			scene.buildAggregate(static_cast<AggregateType>(bvh));
			timeIntersectionQueries<DIM>(scene.aggregate, queryPoints, randomDirections,
										 shuffledIndices, bvhTypes[bvh - 1]);
			timeIntersectionQueries<DIM>(scene.aggregate, queryPoints, randomDirections,
										 indices, bvhTypes[bvh - 1], true);
			timeClosestPointQueries<DIM>(scene.aggregate, queryPoints,
										 shuffledIndices, bvhTypes[bvh - 1]);
			timeClosestPointQueries<DIM>(scene.aggregate, queryPoints,
										 indices, bvhTypes[bvh - 1], true);
		}

#ifdef BENCHMARK_EMBREE
		// build embree bvh aggregate & benchmark queries
		scene.buildEmbreeAggregate();
		timeIntersectionQueries<DIM>(scene.aggregate, queryPoints, randomDirections,
									 shuffledIndices, "Embree Bvh");
		timeClosestPointQueries<DIM>(scene.aggregate, queryPoints,
									 shuffledIndices, "Embree Bvh");
#endif
	}

	if (checkCorrectness) {
		std::cout << "Running correctness tests..." << std::endl;

		// build baseline aggregate
		scene.buildAggregate(AggregateType::Baseline);

		// build bvh aggregates and compare results with baseline
		Scene<DIM> bvhScene;
		bvhScene.loadFiles(true);

		for (int bvh = 1; bvh < 8; bvh++) {
			std::cout << "Testing " << bvhTypes[bvh - 1] << " results against Baseline" << std::endl;
			bvhScene.buildAggregate(static_cast<AggregateType>(bvh));
			testIntersectionQueries<DIM>(scene.aggregate, bvhScene.aggregate,
										 queryPoints, randomDirections, shuffledIndices);
			testIntersectionQueries<DIM>(scene.aggregate, bvhScene.aggregate,
										 queryPoints, randomDirections, indices, true);
			testClosestPointQueries<DIM>(scene.aggregate, bvhScene.aggregate,
										 queryPoints, shuffledIndices);
			testClosestPointQueries<DIM>(scene.aggregate, bvhScene.aggregate,
										 queryPoints, indices, true);
		}

#ifdef BENCHMARK_EMBREE
		// build embree bvh aggregate and compare results with baseline
		std::cout << "Testing Embree Bvh results against Baseline" << std::endl;
		Scene<DIM> embreeBvhScene;
		embreeBvhScene.loadFiles(true);
		embreeBvhScene.buildEmbreeAggregate();
		testIntersectionQueries<DIM>(scene.aggregate, embreeBvhScene.aggregate,
									 queryPoints, randomDirections, shuffledIndices);
		testClosestPointQueries<DIM>(scene.aggregate, embreeBvhScene.aggregate,
									 queryPoints, shuffledIndices);
#endif
	}

	if (vizScene) {
		// build bvh aggregate
		scene.buildAggregate(AggregateType::Bvh_LongestAxisCenter);

		// isolate interior points among query points
		std::vector<Vector<DIM>> interiorPoints;
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
