#include "utilities/scene.h"
#include <ThreadPool.h>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include "args/args.hxx"

using namespace fcpw;

static progschj::ThreadPool pool;
static int nThreads = 8;

template<int DIM>
void generateScatteredPointsAndRays(int nPoints, std::vector<Vector<DIM>>& scatteredPoints,
									std::vector<Vector<DIM>>& randomDirections,
									const BoundingBox<DIM>& boundingBox)
{
	Vector<DIM> e = boundingBox.extent();

	for (int i = 0; i < nPoints; i++) {
		Vector<DIM> o = boundingBox.pMin + cwiseProduct<DIM>(e, uniformRealRandomVector<DIM>());
		Vector<DIM> d = unit<DIM>(uniformRealRandomVector<DIM>(-1.0f, 1.0f));

		scatteredPoints.emplace_back(o);
		randomDirections.emplace_back(d);
	}
}

template<int DIM>
bool raymarch(const std::shared_ptr<Aggregate<DIM>>& aggregate,
			  const BoundingBox<DIM>& boundingBox,
			  Ray<DIM> r, Interaction<DIM>& i)
{
	r.tMax = 0.0f;
	Vector<DIM> x = r(r.tMax);

	while (boundingBox.contains(x)) {
		Interaction<DIM> c;
		BoundingSphere<DIM> s(x, maxFloat);
		bool found = aggregate->findClosestPoint(s, c);
		LOG_IF(INFO, !found) << "No closest point found while raymarching!";
		r.tMax += c.d;
		x = r(r.tMax);

		if (c.d < 1e-5) {
			i = c;
			return true;
		}
	}

	return false;
}

template<int DIM>
void clampToCsg(const std::string& method,
				const std::vector<Vector<DIM>>& scatteredPoints,
				const std::vector<Vector<DIM>>& randomDirections,
				const std::shared_ptr<Aggregate<DIM>>& aggregate,
				const BoundingBox<DIM>& boundingBox,
				std::vector<Vector<DIM>>& clampedPoints)
{
	int N = (int)scatteredPoints.size();
	int pCurrent = 0;
	int pRange = std::max(100, N/nThreads);
	std::vector<Vector<DIM>> hitPoints(N);
	std::vector<bool> didHit(N, false);

	while (pCurrent < N) {
		int pEnd = std::min(N, pCurrent + pRange);
		pool.enqueue([&scatteredPoints, &randomDirections, &hitPoints, &didHit,
					  &method, &aggregate, &boundingBox, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				Ray<DIM> r(scatteredPoints[i], randomDirections[i]);

				if (method == "intersect") {
					std::vector<Interaction<DIM>> cs;
					int hits = aggregate->intersect(r, cs, false, true);
					if (hits > 0) {
						hitPoints[i] = cs[0].p;
						didHit[i] = true;
					}

				} else {
					Interaction<DIM> c;
					if (raymarch<DIM>(aggregate, boundingBox, r, c)) {
						hitPoints[i] = c.p;
						didHit[i] = true;
					}
				}
			}
		});

		pCurrent += pRange;
	}

	pool.wait_until_empty();
	pool.wait_until_nothing_in_flight();

	// add points for which hit was found
	for (int i = 0; i < N; i++) {
		if (didHit[i]) clampedPoints.emplace_back(hitPoints[i]);
	}
}

template<int DIM>
void guiCallback(const Scene<DIM>& scene, const BoundingBox<DIM>& boundingBox,
				 std::vector<Vector<DIM>>& intersectedPoints,
				 std::vector<Vector<DIM>>& raymarchedPoints)
{
	// make ui elements 100 pixels wide, instead of full width
	ImGui::PushItemWidth(100);

	if (ImGui::Button("Add Samples to Visualize CSG")) {
		// generate random points and rays used to visualize csg
		std::vector<Vector<DIM>> scatteredPoints, randomDirections;
		generateScatteredPointsAndRays<DIM>(1000, scatteredPoints, randomDirections, boundingBox);

		// intersect and raymarch points
		clampToCsg<DIM>("intersect", scatteredPoints, randomDirections,
						scene.aggregate, boundingBox, intersectedPoints);
		clampToCsg<DIM>("raymarch", scatteredPoints, randomDirections,
						scene.aggregate, boundingBox, raymarchedPoints);

		if (DIM == 3) {
			// register point clouds
			polyscope::registerPointCloud("Intersected_Points", intersectedPoints);
			polyscope::registerPointCloud("Raymarched_Points", raymarchedPoints);
		}
	}

	ImGui::PopItemWidth();
}

template<int DIM>
void visualizeScene(const Scene<DIM>& scene, const BoundingBox<DIM>& boundingBox,
					std::vector<Vector<DIM>>& intersectedPoints,
					std::vector<Vector<DIM>>& raymarchedPoints)
{
	// set a few options
	polyscope::options::programName = "CSG Tests";
	polyscope::options::verbosity = 0;
	polyscope::options::usePrefsFile = false;
	polyscope::options::autocenterStructures = false;

	// initialize polyscope
	polyscope::init();

	if (DIM == 3) {
		// register surface meshes
		for (int i = 0; i < (int)scene.soups.size(); i++) {
			std::string meshName = "Polygon_Soup_" + std::to_string(i);
			const std::vector<std::vector<int>>& indices = scene.soups[i]->indices;
			const std::vector<Vector<DIM>>& positions = scene.soups[i]->positions;

			if (scene.instanceTransforms[i].size() > 0) {
				for (int j = 0; j < (int)scene.instanceTransforms[i].size(); j++) {
					std::string transformedMeshName = meshName + "_" + std::to_string(j);
					std::vector<Vector<DIM>> transformedPositions;

					for (int k = 0; k < (int)positions.size(); k++) {
						transformedPositions.emplace_back(transformVector<DIM>(
														scene.instanceTransforms[i][j], positions[k]));
					}

					polyscope::registerSurfaceMesh(transformedMeshName, transformedPositions, indices);
				}

			} else {
				polyscope::registerSurfaceMesh(meshName, positions, indices);
			}
		}
	}

	// register point clouds
	polyscope::registerPointCloud("Intersected_Points", intersectedPoints);
	polyscope::registerPointCloud("Raymarched_Points", raymarchedPoints);

	// register callback
	polyscope::state::userCallback = std::bind(&guiCallback<DIM>, std::cref(scene),
											   std::cref(boundingBox),
											   std::ref(intersectedPoints),
											   std::ref(raymarchedPoints));

	// give control to polyscope gui
	polyscope::show();
}

template<int DIM>
void run()
{
	// build scene
	Scene<DIM> scene;
	scene.loadFiles(true);
	scene.buildAggregate(AggregateType::Bvh_LongestAxisCenter);

	// generate random points and rays used to visualize csg
	BoundingBox<DIM> boundingBox = scene.aggregate->boundingBox();
	std::vector<Vector<DIM>> scatteredPoints, randomDirections;
	generateScatteredPointsAndRays<DIM>(1000, scatteredPoints, randomDirections, boundingBox);

	// intersect and raymarch points
	std::vector<Vector<DIM>> intersectedPoints, raymarchedPoints;
	clampToCsg<DIM>("intersect", scatteredPoints, randomDirections,
					scene.aggregate, boundingBox, intersectedPoints);
	clampToCsg<DIM>("raymarch", scatteredPoints, randomDirections,
					scene.aggregate, boundingBox, raymarchedPoints);

	// visualize scene
	visualizeScene<DIM>(scene, boundingBox, intersectedPoints, raymarchedPoints);
}

int main(int argc, const char *argv[]) {
	google::InitGoogleLogging(argv[0]);
#ifdef PROFILE
	Profiler::detect(argc, argv);
#endif
	// configure the argument parser
	args::ArgumentParser parser("csg tests");
	args::Group group(parser, "", args::Group::Validators::DontCare);
	args::ValueFlag<int> dim(parser, "integer", "scene dimension", {"dim"});
	args::ValueFlag<int> nThreads(parser, "integer", "nThreads", {"nThreads"});
	args::ValueFlag<std::string> csgFilename(parser, "string", "csg filename", {"csgFile"});
	args::ValueFlag<std::string> instanceFilename(parser, "string", "instance filename", {"instanceFile"});
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

	if (!csgFilename || !triangleFilenames) {
		std::cerr << "Please specify csg and triangle soup filenames" << std::endl;
		return EXIT_FAILURE;
	}

	// set global flags
	if (nThreads) ::nThreads = args::get(nThreads);
	if (csgFilename) ::csgFilename = args::get(csgFilename);
	if (instanceFilename) ::instanceFilename = args::get(instanceFilename);
	if (triangleFilenames) {
		for (const auto tsf: args::get(triangleFilenames)) {
			files.emplace_back(std::make_pair(tsf, LoadingOption::ObjTriangles));
		}
	}

	if (files.size() <= 1) {
		std::cerr << "Specify atleast 2 soups" << std::endl;
		return EXIT_FAILURE;
	}

	// run app
	if (DIM == 3) run<3>();

#ifdef PROFILE
	Profiler::dumphtml();
#endif
	return 0;
}
