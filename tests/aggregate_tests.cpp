
#include <fcpw/utilities/scene_loader.h>
#include <atomic>
#include <thread>
#include <mutex>
#include "CLI11.hpp"

using namespace fcpw;
constexpr int DIM = 3;

static int opt_threads = 1;
static int opt_queries = 10000;
static int opt_build_heuristic = 3;
static int opt_bounding_volume = 0;
static float opt_bb_scale = 1.5f;
static bool opt_vectorize = true;
static bool opt_coherent = false;
static bool opt_run_auto = false;
static bool opt_time_rays = false;
static bool opt_embree = false;
static bool opt_check = false;

const std::vector<std::string> bvh_type_names = 
		{"Baseline", "Bvh_LongestAxisCenter", "Bvh_OverlapSurfaceArea", "Bvh_SurfaceArea", "Bvh_OverlapVolume", "Bvh_Volume"};
const std::vector<std::string> vol_type_names = 
		{"AxisAlignedBox", "Sphere", "OrientedBox", "SphereSweptRect"};

template<size_t DIM>
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

template<size_t DIM>
void generateScatteredPointsAndRays(std::vector<Vector<DIM>>& scatteredPoints,
									std::vector<Vector<DIM>>& randomDirections,
									const BoundingBox<DIM>& boundingBox)
{
	// parition the scene bounding box into boxes
	std::vector<BoundingBox<DIM>> boxes;
	splitBoxRecursive<DIM>(boundingBox, boxes, 6);

	// generate queries in each box
	int nBoxes = (int)boxes.size();
	int nQueriesPerBox = (int)std::ceil((float)opt_queries/nBoxes);

	for (int i = 0; i < nBoxes; i++) {
		Vector<DIM> e = boxes[i].extent();

		for (int j = 0; j < nQueriesPerBox; j++) {
			Vector<DIM> o = boxes[i].pMin + e.cwiseProduct(uniformRealRandomVector<DIM>());
			Vector<DIM> d = uniformRealRandomVector<DIM>(-1.0f, 1.0f);
			if (std::fabs(e[DIM - 1]) < 5*epsilon) {
				o[DIM - 1] = 0.0f;
				d[DIM - 1] = 0.0f;
			}
			d.normalize();

			scatteredPoints.emplace_back(o);
			randomDirections.emplace_back(d);
		}
	}

	// resize if required
	scatteredPoints.resize(opt_queries);
	randomDirections.resize(opt_queries);
}

template<size_t DIM>
uint64_t timeClosestPointQueries(const std::unique_ptr<Aggregate<DIM>>& aggregate,
							 const std::vector<Vector<DIM>>& queryPoints,
							 const std::vector<int>& indices,
							 int& max_nodes,
							 double& prim_percent,
							 bool queriesCoherent=false)
{
	std::atomic<int> totalNodesVisited(0);
	std::atomic<int> maxNodesVisited(0);
	std::atomic<uint64_t> primTicks(0), totalTicks(0);
	std::atomic<bool> stopQueries(false);
	auto t1 = std::chrono::high_resolution_clock::now();

	auto time = [&](int begin, int end) {
		int nodesVisitedByThread = 0;
		int maxNodesVisitedByThread = 0;
		uint64_t prim_ticks_thread = 0;
		Interaction<DIM> cPrev;
		Vector<DIM> queryPrev = Vector<DIM>::Zero();

		auto tt1 = std::chrono::high_resolution_clock::now();
		for (int i = begin; i < end; ++i) {
			if (stopQueries) break;
			int I = indices[i];
			float distPrev = (queryPoints[I] - queryPrev).norm();
			float r2 = cPrev.nodeIndex == -1 ? maxFloat : (float)std::pow((cPrev.d + distPrev)*1.25f, 2);

			int nodesVisited = 0;
			Interaction<DIM> c;
			BoundingSphere<DIM> s(queryPoints[I], r2);
			bool found = aggregate->findClosestPointFromNodeTimed(s, c, 0, aggregate->index, Vector<DIM>::Zero(), nodesVisited, prim_ticks_thread);
			nodesVisitedByThread += nodesVisited;
			maxNodesVisitedByThread = std::max(maxNodesVisitedByThread, nodesVisited);

			if (found) cPrev = c;
			else {
				std::cerr << "Closest points not found!" << std::endl;
				stopQueries = true;
			}

			queryPrev = queryPoints[I];
		}
		auto tt2 = std::chrono::high_resolution_clock::now();

		primTicks += prim_ticks_thread;
		totalTicks += (tt2 - tt1).count();
		totalNodesVisited += nodesVisitedByThread;
		if (maxNodesVisited < maxNodesVisitedByThread) {
			maxNodesVisited = maxNodesVisitedByThread; // not thread-safe, but ok for test
		}
	};

	{
		if(opt_threads == 0) opt_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;
		int grain = opt_queries / opt_threads;
		for(int i = 0; i < opt_threads; i++) {
			int begin = i * grain;
			int end = i == opt_threads - 1 ? opt_queries : begin + grain;
			threads.emplace_back([&]() {
				time(begin, end);
			});
		}
		for(auto& t : threads) t.join();
	}

	prim_percent = 100.0 * primTicks.load() / totalTicks.load();
	max_nodes = maxNodesVisited.load();

	auto t2 = std::chrono::high_resolution_clock::now();
	return (t2 - t1).count();
}

template<size_t DIM>
uint64_t timeIntersectionQueries(const std::unique_ptr<Aggregate<DIM>>& aggregate,
							 const std::vector<Vector<DIM>>& rayOrigins,
							 const std::vector<Vector<DIM>>& rayDirections,
							 const std::vector<int>& indices,
							 int& max_nodes,
							 double& prim_percent,
							 bool queriesCoherent=false)
{
	std::atomic<int> totalNodesVisited(0);
	std::atomic<int> maxNodesVisited(0);
	std::atomic<uint64_t> primTicks(0), totalTicks(0);
	auto t1 = std::chrono::high_resolution_clock::now();

	auto time = [&](int begin, int end) {
		int nodesVisitedByThread = 0;
		int maxNodesVisitedByThread = 0;
		uint64_t prim_ticks_thread = 0;
		Interaction<DIM> cPrev;

		auto tt1 = std::chrono::high_resolution_clock::now();
		for (int i = begin; i < end; ++i) {
			int I = indices[i];

			int nodesVisited = 0;
			std::vector<Interaction<DIM>> cs;
			Ray<DIM> r(rayOrigins[I], rayDirections[I]);
			bool hit = (bool)aggregate->intersectFromNodeTimed(r, cs, 0, aggregate->index, nodesVisited, prim_ticks_thread);
			nodesVisitedByThread += nodesVisited;
			maxNodesVisitedByThread = std::max(maxNodesVisitedByThread, nodesVisited);

			if (hit) cPrev = cs[0];
		}
		auto tt2 = std::chrono::high_resolution_clock::now();

		totalTicks += (tt2 - tt1).count();
		primTicks += prim_ticks_thread;
		totalNodesVisited += nodesVisitedByThread;
		if (maxNodesVisited < maxNodesVisitedByThread) {
			maxNodesVisited = maxNodesVisitedByThread; // not thread-safe, but ok for test
		}
	};

	{
		if(opt_threads == 0) opt_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;
		int grain = opt_queries / opt_threads;
		for(int i = 0; i < opt_threads; i++) {
			int begin = i * grain;
			int end = i == opt_threads - 1 ? opt_queries : begin + grain;
			threads.emplace_back([&]() {
				time(begin, end);
			});
		}
		for(auto& t : threads) t.join();
	}

	prim_percent = 100.0 * primTicks.load() / totalTicks.load();
	max_nodes = maxNodesVisited.load();

	auto t2 = std::chrono::high_resolution_clock::now();
	return (t2 - t1).count();
}


template<size_t DIM>
void testClosestPointQueries(const std::unique_ptr<Aggregate<DIM>>& aggregate1,
							 const std::unique_ptr<Aggregate<DIM>>& aggregate2,
							 const std::vector<Vector<DIM>>& queryPoints,
							 const std::vector<int>& indices)
{

	std::atomic<bool> stopQueries(false);
	std::mutex mut;

	auto test = [&](int begin, int end) {
		Interaction<DIM> cPrev;
		Vector<DIM> queryPrev = Vector<DIM>::Zero();

		for (int i = begin; i < end; ++i) {
			if (stopQueries) break;

			int I = indices[i];
			float distPrev = (queryPoints[I] - queryPrev).norm();

			float r = (cPrev.d + distPrev)*1.25f;
			float r2 = cPrev.nodeIndex == -1 ? maxFloat : r * r;

			Interaction<DIM> c1;
			BoundingSphere<DIM> s1(queryPoints[I], maxFloat);
			bool found1 = aggregate1->findClosestPoint(s1, c1);

			int nodesVisited = 0;
			Interaction<DIM> c2;
			BoundingSphere<DIM> s2(queryPoints[I], r2);
			bool found2 = aggregate2->findClosestPointFromNode(s2, c2, 0, aggregate2->index, Vector<DIM>::Zero(), nodesVisited);

			if (found1 != found2 || std::fabs(c1.d - c2.d) > 1e-6) {
				std::lock_guard<std::mutex> lock(mut);
				std::cerr << "d1: " << c1.d << " d2: " << c2.d
						  << "\np1: " << c1.p << " p2: " << c2.p
						  << "\nClosest points do not match!" << std::endl;
				stopQueries = true;
			}

			if (found2) cPrev = c2;
			else {
				std::lock_guard<std::mutex> lock(mut);
				std::cerr << "Closest points not found!" << std::endl;
				stopQueries = true;
			}

			queryPrev = queryPoints[I];
		}
	};

	{
		int n_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;
		int grain = opt_queries / n_threads;
		for(int i = 0; i < n_threads; i++) {
			int begin = i * grain;
			int end = i == n_threads - 1 ? opt_queries : begin + grain;
			threads.emplace_back([&]() {
				test(begin, end);
			});
		}
		for(auto& t : threads) t.join();
	}
}

template<size_t DIM>
void testIntersectionQueries(const std::unique_ptr<Aggregate<DIM>>& aggregate1,
							 const std::unique_ptr<Aggregate<DIM>>& aggregate2,
							 const std::vector<Vector<DIM>>& rayOrigins,
							 const std::vector<Vector<DIM>>& rayDirections,
							 const std::vector<int>& indices)
{
	std::atomic<bool> stopQueries(false);
	auto test = [&](int begin, int end) {
		Interaction<DIM> cPrev;

		for (int i = begin; i < end; ++i) {
			if (stopQueries) break;
			int I = indices[i];

			std::vector<Interaction<DIM>> c1;
			Ray<DIM> r1(rayOrigins[I], rayDirections[I]);
			bool hit1 = (bool)aggregate1->intersect(r1, c1);

			int nodesVisited = 0;
			std::vector<Interaction<DIM>> c2;
			Ray<DIM> r2(rayOrigins[I], rayDirections[I]);
			bool hit2 = (bool)aggregate2->intersectFromNode(r2, c2, 0, aggregate2->index, nodesVisited);

			if ((hit1 != hit2) || (hit1 && hit2 && c1[0] != c2[0])) {
				std::cerr << "d1: " << c1[0].d << " d2: " << c2[0].d
						  << "\np1: " << c1[0].p << " p2: " << c2[0].p
						  << "\nIntersections do not match!" << std::endl;
				stopQueries = true;
			}

			if (hit2) cPrev = c2[0];

			std::vector<Interaction<DIM>> c3;
			Ray<DIM> r3(rayOrigins[I], rayDirections[I]);
			int hit3 = aggregate1->intersect(r3, c3, false, true);

			nodesVisited = 0;
			std::vector<Interaction<DIM>> c4;
			Ray<DIM> r4(rayOrigins[I], rayDirections[I]);
			int hit4 = aggregate2->intersectFromNode(r4, c4, 0, aggregate2->index, nodesVisited, false, true);

			if (hit3 != hit4) {
				std::cerr << "Number of intersections do not match!"
						  << " hits1: " << hit3
						  << " hits2: " << hit4
						  << std::endl;
				stopQueries = true;
			}
		}
	};

	{
		int n_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;
		int grain = opt_queries / n_threads;
		for(int i = 0; i < n_threads; i++) {
			int begin = i * grain;
			int end = i == n_threads - 1 ? opt_queries : begin + grain;
			threads.emplace_back([&]() {
				test(begin, end);
			});
		}
		for(auto& t : threads) t.join();
	}
}

void run_checks() {

	Scene<DIM> baseScene;
	SceneLoader<DIM> sceneLoader;
	sceneLoader.loadFiles(baseScene, false);
	baseScene.build(AggregateType::Baseline, BoundingVolumeType::AxisAlignedBox, false, true);
	SceneData<DIM> *baseSceneData = baseScene.getSceneData();

	// generate random points and rays used to visualize csg
	BoundingBox<DIM> boundingBox = baseSceneData->aggregate->boundingBox();
	boundingBox.pMin *= opt_bb_scale;
	boundingBox.pMax *= opt_bb_scale;

	std::vector<Vector<DIM>> queryPoints, randomDirections;
	generateScatteredPointsAndRays<DIM>(queryPoints, randomDirections, boundingBox);

	// generate indices
	std::vector<int> indices, shuffledIndices;
	for (int i = 0; i < opt_queries; i++) {
		indices.emplace_back(i);
		shuffledIndices.emplace_back(i);
	}

	unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), std::default_random_engine(seed));

	Scene<DIM> testScene;
	sceneLoader.loadFiles(testScene, false);

	auto run_tests = [&](bool rays, BoundingVolumeType volume, AggregateType heuristic, bool vector) {
			
		printf("CHECKING: %s, %s, %s, %s\n", rays ? "RAY" : "CPQ", vector ? "yes" : "no", bvh_type_names[(int)heuristic].c_str(), vol_type_names[(int)volume].c_str());

		testScene.build(heuristic, volume, vector, true);
		SceneData<DIM>* sceneData = testScene.getSceneData();
		if (rays) {
			testIntersectionQueries<DIM>(baseSceneData->aggregate, sceneData->aggregate, queryPoints, randomDirections, shuffledIndices);
		} else {
			testClosestPointQueries<DIM>(baseSceneData->aggregate, sceneData->aggregate, queryPoints, shuffledIndices);
		}
	};

	std::vector<BoundingVolumeType> Btypes = {BoundingVolumeType::AxisAlignedBox, BoundingVolumeType::Sphere, BoundingVolumeType::OrientedBox, BoundingVolumeType::SphereSweptRect};
	std::vector<AggregateType> Stypes = {AggregateType::Baseline, AggregateType::Bvh_LongestAxisCenter, AggregateType::Bvh_OverlapSurfaceArea, 
										 AggregateType::Bvh_SurfaceArea, AggregateType::Bvh_OverlapVolume, AggregateType::Bvh_Volume};
	std::vector<bool> Svectorize = {false, true};
	std::vector<bool> Srays = {false, true};

	if(opt_run_auto) {
		for(const auto& use_rays : Srays) {
			for (const auto& vectorize : Svectorize) {
				for (const auto& bvh_type : Stypes) {
					
					if(bvh_type == AggregateType::Baseline) continue;
					
					for (const auto& vol_type : Btypes) {

						run_tests(use_rays, vol_type, bvh_type, vectorize);
					}
				}
			}
		}
	} else {
		run_tests(opt_time_rays, Btypes[opt_bounding_volume], Stypes[opt_build_heuristic], opt_vectorize);
	}
}

void run()
{
	// build baseline scene
	Scene<DIM> scene;
	SceneLoader<DIM> sceneLoader;
	sceneLoader.loadFiles(scene, false);
	scene.build(AggregateType::Baseline, BoundingVolumeType::AxisAlignedBox, false, true);
	SceneData<DIM> *sceneData = scene.getSceneData();

	// generate random points and rays used to visualize csg
	BoundingBox<DIM> boundingBox = sceneData->aggregate->boundingBox();
	boundingBox.pMin *= opt_bb_scale;
	boundingBox.pMax *= opt_bb_scale;

	std::vector<Vector<DIM>> queryPoints, randomDirections;
	generateScatteredPointsAndRays<DIM>(queryPoints, randomDirections, boundingBox);

	// generate indices
	std::vector<int> indices, shuffledIndices;
	for (int i = 0; i < opt_queries; i++) {
		indices.emplace_back(i);
		shuffledIndices.emplace_back(i);
	}

	unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), std::default_random_engine(seed));

	// Parameters 
	//  - CPQs / rays
	//	- BVH build heuristic 
	//  - Thread count
	//  - Vectorized or not (and width, but this is only varied at compile time)
	//  - Coherent queries or not
	//  - bounding volume type

	auto run_benchmark = [&](bool rays, BoundingVolumeType volume, AggregateType heuristic, bool vector, bool coherent, int threads) {
			
		scene.build(heuristic, volume, vector, true);
		sceneData = scene.getSceneData();

		opt_threads = threads;
		uint64_t time = 0;
		int max_nodes = 0;
		double prim_percent = 0.0;

		if (rays) {
			if (coherent) {
				time = timeIntersectionQueries<DIM>(sceneData->aggregate, queryPoints, randomDirections, indices, max_nodes, prim_percent, true);
			} else {
				time = timeIntersectionQueries<DIM>(sceneData->aggregate, queryPoints, randomDirections, shuffledIndices, max_nodes, prim_percent);
			}
		} else {
			if (coherent) {
				time = timeClosestPointQueries<DIM>(sceneData->aggregate, queryPoints, indices, max_nodes, prim_percent, true);
			} else {
				time = timeClosestPointQueries<DIM>(sceneData->aggregate, queryPoints, shuffledIndices, max_nodes, prim_percent);
			}
		}

		printf("%10s, %10s, %8s, %22s, %16s, %7d, %8d, %12.2f%%, %12f\n", rays ? "RAY" : "CPQ", vector ? "yes" : "no", coherent ? "yes" : "no", 
			   bvh_type_names[(int)heuristic].c_str(), vol_type_names[(int)volume].c_str(), threads, max_nodes, prim_percent, time / 1e9);
	};

	std::vector<BoundingVolumeType> Btypes = {BoundingVolumeType::AxisAlignedBox, BoundingVolumeType::Sphere, BoundingVolumeType::OrientedBox, BoundingVolumeType::SphereSweptRect};
	std::vector<AggregateType> Stypes = {AggregateType::Baseline, AggregateType::Bvh_LongestAxisCenter, AggregateType::Bvh_OverlapSurfaceArea, 
										 AggregateType::Bvh_SurfaceArea, AggregateType::Bvh_OverlapVolume, AggregateType::Bvh_Volume};
	std::vector<bool> Svectorize = {false, true};
	std::vector<bool> Scoherent = {false, true};
	std::vector<int> Sthreads = {8};
	std::vector<bool> Srays = {false, true};

	printf("\n");
	printf("FCPQ Benchmark: %d queries\n", opt_queries);
	printf("%10s, %10s, %8s, %22s, %16s, %7s, %8s, %13s, %12s\n", "CPQ/RAY", "Vectorized", "Coherent", "Build Heuristic", "Bounding Volume", "Threads", "Nodes", "% Primitive", "Time");
	if(opt_run_auto) {
		for(const auto& use_rays : Srays) {
			if(use_rays) continue;
			for (const auto& vectorize : Svectorize) {
				if(vectorize == true) continue;
				for (const auto& coherent : Scoherent) {
					for (const auto& bvh_type : Stypes) {
						
						if(bvh_type == AggregateType::Baseline) continue;

						for (const auto& vol_type : Btypes) {
							
							if(vol_type == BoundingVolumeType::Sphere) continue;
							if(vol_type == BoundingVolumeType::SphereSweptRect || vol_type == BoundingVolumeType::OrientedBox) {
								// Not yet supported
								if(bvh_type == AggregateType::Bvh_OverlapSurfaceArea) continue;
								if(bvh_type == AggregateType::Bvh_OverlapVolume) continue;
							}
							
							for (const auto& threads : Sthreads) {
								if(threads > (int)std::thread::hardware_concurrency()) break;
								run_benchmark(use_rays, vol_type, bvh_type, vectorize, coherent, threads);
							}
						}
					}
				}
			}
		}
	} else {
		run_benchmark(opt_time_rays, Btypes[opt_bounding_volume], Stypes[opt_build_heuristic], opt_vectorize, opt_coherent, opt_threads);
	}
}

int main(int argc, const char *argv[]) {

	CLI::App args{"FCPW Aggregate Test: CPQs"};
	std::string file;

	args.add_option("-n,--queries", opt_queries, "number of queries");
	args.add_option("-s,--scene", file, "triangle soup file");
	args.add_option("--scale", opt_bb_scale, "generate points in the scene bounding box * this scale");
	args.add_flag("--rays", opt_time_rays, "time ray intersections");
	
	args.add_flag("--auto", opt_run_auto, "sweep all parameters automatically");
	args.add_option("-t,--threads", opt_threads, "number of threads");
	args.add_option("--heuristic", opt_build_heuristic, "type of build heuristic");
	args.add_option("--volume", opt_bounding_volume, "type of bounding volume");
	args.add_flag("--vectorize", opt_vectorize, "use vectorized bvh");
	args.add_flag("--coherent", opt_coherent, "use coherent queries");
	args.add_flag("--embree", opt_coherent, "benchmark embree");

	args.add_flag("--check", opt_check, "run correctness checks");

    CLI11_PARSE(args, argc, argv);

	opt_build_heuristic = clamp(opt_build_heuristic, 0, 5);
	opt_bounding_volume = clamp(opt_bounding_volume, 0, 3);

	files.emplace_back(std::make_pair(file, LoadingOption::ObjTriangles));

	if(opt_check) {
		run_checks();
	} else {
		run();
	}
	return 0;
}
