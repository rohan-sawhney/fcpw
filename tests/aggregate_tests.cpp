
#include <fcpw/utilities/scene_loader.h>
#include <atomic>
#include <thread>
#include "CLI11.hpp"

using namespace fcpw;
using namespace std::chrono;
constexpr int DIM = 3;

static int nQueries = 10000;
static bool vectorize = false;
static int n_threads = 0;

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
	int nQueriesPerBox = (int)std::ceil((float)nQueries/nBoxes);

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
	scatteredPoints.resize(nQueries);
	randomDirections.resize(nQueries);
}

template<size_t DIM>
void timeClosestPointQueries(const std::unique_ptr<Aggregate<DIM>>& aggregate,
							 const std::vector<Vector<DIM>>& queryPoints,
							 const std::vector<int>& indices,
							 const std::string& aggregateType,
							 bool queriesCoherent=false)
{
	std::atomic<int> totalNodesVisited(0);
	std::atomic<int> maxNodesVisited(0);
	std::atomic<bool> stopQueries(false);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	auto time = [&](int begin, int end) {
		int nodesVisitedByThread = 0;
		int maxNodesVisitedByThread = 0;
		Interaction<DIM> cPrev;
		Vector<DIM> queryPrev = Vector<DIM>::Zero();

		for (int i = begin; i < end; ++i) {
			if (stopQueries) break;
			int I = indices[i];
			float distPrev = (queryPoints[I] - queryPrev).norm();
			float r2 = cPrev.nodeIndex == -1 ? maxFloat : std::powf((cPrev.d + distPrev)*1.25f, 2);

			int nodesVisited = 0;
			Interaction<DIM> c;
			BoundingSphere<DIM> s(queryPoints[I], r2);
			bool found = aggregate->findClosestPointFromNode(s, c, 0, aggregate->index, Vector<DIM>::Zero(), nodesVisited);
			nodesVisitedByThread += nodesVisited;
			maxNodesVisitedByThread = std::max(maxNodesVisitedByThread, nodesVisited);

			if (found) cPrev = c;
			else {
				std::cerr << "Closest points not found!" << std::endl;
				stopQueries = true;
			}

			queryPrev = queryPoints[I];
		}

		totalNodesVisited += nodesVisitedByThread;
		if (maxNodesVisited < maxNodesVisitedByThread) {
			maxNodesVisited = maxNodesVisitedByThread; // not thread-safe, but ok for test
		}
	};

	{
		if(n_threads == 0) n_threads = std::thread::hardware_concurrency();

		std::vector<std::thread> threads;
		int grain = nQueries / n_threads;
		for(int i = 0; i < n_threads; i++) {
			int begin = i * grain;
			int end = i == n_threads - 1 ? nQueries : begin + grain;
			threads.emplace_back([&]() {
				time(begin, end);
			});
		}
		for(auto& t : threads) t.join();
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
	std::cout << timeSpan.count() << std::endl;
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
	std::vector<Vector<DIM>> queryPoints, randomDirections;
	generateScatteredPointsAndRays<DIM>(queryPoints, randomDirections, boundingBox);

	// generate indices
	std::vector<int> indices, shuffledIndices;
	for (int i = 0; i < nQueries; i++) {
		indices.emplace_back(i);
		shuffledIndices.emplace_back(i);
	}

	unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), std::default_random_engine(seed));

	std::vector<std::string> bvhTypes({"Bvh_LongestAxisCenter", "Bvh_OverlapSurfaceArea",
									   "Bvh_SurfaceArea", "Bvh_OverlapVolume", "Bvh_Volume"});

	for (int bvh = 1; bvh < 6; bvh++) {
		scene.build(static_cast<AggregateType>(bvh), vectorize, true);
		sceneData = scene.getSceneData();

		timeClosestPointQueries<DIM>(sceneData->aggregate, queryPoints,
										shuffledIndices, bvhTypes[bvh - 1]);
		timeClosestPointQueries<DIM>(sceneData->aggregate, queryPoints,
										indices, bvhTypes[bvh - 1], true);
	}
}

int main(int argc, const char *argv[]) {

	CLI::App args{"FCPW Aggregate Test: CPQs"};
	std::string file;

	args.add_option("-n,--queries", nQueries, "number of queries");
	args.add_option("-t,--threads", n_threads, "number of threads");
	args.add_option("-s,--scene", file, "triangle soup filenames");
	args.add_flag("-v,--vectorize", vectorize, "use vectorized bvh");
    CLI11_PARSE(args, argc, argv);

	files.emplace_back(std::make_pair(file, LoadingOption::ObjTriangles));

	run();
	return 0;
}
