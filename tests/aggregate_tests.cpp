
#include <fcpw/utilities/scene_loader.h>
#include <atomic>
#include <thread>
#include "CLI11.hpp"

using namespace fcpw;
constexpr int DIM = 3;

static int opt_threads = 1;
static int opt_queries = 10000;
static int opt_build_heuristic = 3;
static float opt_bb_scale = 2.0f;
static bool opt_vectorize = true;
static bool opt_coherent = false;
static bool opt_run_auto = false;
static bool opt_time_rays = false;
static bool opt_embree = false;

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

void run()
{
	// build baseline scene
	Scene<DIM> scene;
	SceneLoader<DIM> sceneLoader;
	sceneLoader.loadFiles(scene, false);
	scene.build(AggregateType::Baseline, false, true);
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
	//  - [WIP] bounding volume type

	std::vector<std::string> bvh_type_names = 
		{"Baseline", "Bvh_LongestAxisCenter", "Bvh_OverlapSurfaceArea", "Bvh_SurfaceArea", "Bvh_OverlapVolume", "Bvh_Volume"};

	auto run_benchmark = [&](bool rays, AggregateType heuristic, bool vector, bool coherent, int threads) {
			
		scene.build(heuristic, vector, true);
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

		printf("%10s, %10s, %8s, %22s, %7d, %8d, %12.2f%%, %12f\n", rays ? "RAY" : "CPQ", vector ? "yes" : "no", coherent ? "yes" : "no", 
			   bvh_type_names[(int)heuristic].c_str(), threads, max_nodes, prim_percent, time / 1e9);
	};

	std::vector<AggregateType> Stypes = {AggregateType::Baseline, AggregateType::Bvh_LongestAxisCenter, AggregateType::Bvh_OverlapSurfaceArea, 
										 AggregateType::Bvh_SurfaceArea, AggregateType::Bvh_OverlapVolume, AggregateType::Bvh_Volume};
	std::vector<bool> Svectorize = {false, true};
	std::vector<bool> Scoherent = {false, true};
	std::vector<int> Sthreads = {1, 2, 4, 8, 16, 32, 64, 128};
	std::vector<bool> Srays = {false, true};

	printf("\n");
	printf("FCPQ Benchmark: %d queries\n", opt_queries);
	printf("%10s, %10s, %8s, %22s, %7s, %8s, %13s, %12s\n", "CPQ/RAY", "Vectorized", "Coherent", "Build Heuristic", "Threads", "Nodes", "% Primitive", "Time");
	if(opt_run_auto) {
		for(const auto& use_rays : Srays) {
			for (const auto& vectorize : Svectorize) {
				for (const auto& coherent : Scoherent) {
					for (const auto& bvh_type : Stypes) {
						if(bvh_type == AggregateType::Baseline) continue;
						for (const auto& threads : Sthreads) {
							if(threads > 2 * (int)std::thread::hardware_concurrency()) break;
							run_benchmark(use_rays, bvh_type, vectorize, coherent, threads);
						}
					}
				}
			}
		}
	} else {
		run_benchmark(opt_time_rays, Stypes[opt_build_heuristic], opt_vectorize, opt_coherent, opt_threads);
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
	args.add_flag("--vectorize", opt_vectorize, "use vectorized bvh");
	args.add_flag("--coherent", opt_coherent, "use coherent queries");
	args.add_flag("--embree", opt_coherent, "benchmark embree");

    CLI11_PARSE(args, argc, argv);

	opt_build_heuristic = clamp(opt_build_heuristic, 0, 5);

	files.emplace_back(std::make_pair(file, LoadingOption::ObjTriangles));

	run();
	return 0;
}
