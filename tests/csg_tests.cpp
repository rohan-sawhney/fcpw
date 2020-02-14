#include "utilities/scene.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include "args/args.hxx"

using namespace fcpw;
/*
int nSamples = 10000;

template <int DIM>
void generateRandomRayDirections(int nDirections, std::vector<Vector<DIM>>& rayDirections)
{
	Vector<DIM> d = Vector<DIM>::Zero();
	rayDirections.clear();

	for (int i = 0; i < nSamples; i++) {
		for (int j = 0; j < DIM; j++) {
			d(j) = uniformRealRandomNumber(-1.0f, 1.0f);
		}

		d.normalize();
		rayDirections.emplace_back(d);
	}
}

template <int DIM>
bool raymarch(const std::shared_ptr<Aggregate<DIM>>& aggregate,
			  const BoundingBox<DIM>& box, Ray<DIM> r, Interaction<DIM>& i)
{
	r.tMax = 0.0f;
	Vector<DIM> x = r(r.tMax);

	while (box.contains(x)) {
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

template <int DIM>
void clampToCsg(const std::string& method, int nSamples,
				const std::shared_ptr<PointCloud<DIM>>& pointCloud,
				const std::vector<Vector<DIM>>& rayDirections,
				const std::shared_ptr<Aggregate<DIM>>& aggregate,
				std::vector<Vector<DIM>>& points)
{
	int N = nSamples;
	int pCurrent = 0;
	int pRange = std::max(100, N/nThreads);
	std::vector<Vector<DIM>> hitPoints(N);
	std::vector<bool> didHit(N, false);
	BoundingBox<DIM> boundingBox = aggregate->boundingBox();

	while (pCurrent < N) {
		int pEnd = std::min(N, pCurrent + pRange);
		pool.enqueue([&pointCloud, &rayDirections, &hitPoints, &didHit,
					  &method, &aggregate, &boundingBox, pCurrent, pEnd]() {
			#ifdef PROFILE
				PROFILE_THREAD_SCOPED();
			#endif

			for (int i = pCurrent; i < pEnd; i++) {
				Ray<DIM> r(pointCloud->points[i], rayDirections[i]);

				if (method == "intersect") {
					std::vector<Interaction<DIM>> cs;
					int hits = aggregate->intersect(r, cs, false, true, true);
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
		if (didHit[i]) points.emplace_back(hitPoints[i]);
	}
}

template <int DIM>
void updateScene(const std::shared_ptr<Aggregate<DIM>>& aggregate,
				 std::shared_ptr<PointCloud<DIM>>& boundaryPointCloud,
				 std::shared_ptr<PointCloud<DIM>>& interiorPointCloud,
				 SamplingDomain<DIM>& domain,
				 std::vector<Vector<DIM>>& pointsIntersected,
				 std::vector<Vector<DIM>>& pointsRaymarched)
{
	domain = SamplingDomain<DIM>(aggregate->boundingBox());
	boundaryPointCloud->points.clear();
	interiorPointCloud->points.clear();
	pointsIntersected.clear();
	pointsRaymarched.clear();

	polyscope::removeStructure("Intersected_Point_Cloud", false);
	polyscope::removeStructure("Raymarched_Point_Cloud", false);
	polyscope::removeStructure("Interior_Point_Cloud", false);
}

template <int DIM>
void guiCallback(int nSoups, std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
				 std::shared_ptr<PointCloud<DIM>>& boundaryPointCloud,
				 std::shared_ptr<PointCloud<DIM>>& interiorPointCloud,
				 std::unique_ptr<Sampler<DIM>>& boundarySampler,
				 std::unique_ptr<Sampler<DIM>>& interiorSampler,
				 SamplingDomain<DIM>& domain,
				 std::vector<Vector<DIM>>& pointsIntersected,
				 std::vector<Vector<DIM>>& pointsRaymarched,
				 std::shared_ptr<Aggregate<DIM>>& aggregate)
{
	// make ui elements 100 pixels wide, instead of full width
	ImGui::PushItemWidth(100);

	if (csgFilename.empty()) {
		if (nSoups == 2) {
			static std::string operation = "None";
			std::string operationItems[] = {"None", "Union", "Intersection", "Difference"};
			if (ImGui::BeginCombo("Operation", operation.c_str())) {
				for (int n = 0; n < IM_ARRAYSIZE(operationItems); n++) {
					bool selected = (operation == operationItems[n]);
					if (ImGui::Selectable(operationItems[n].c_str(), selected)) {
						if (operation != operationItems[n]) {
							operation = operationItems[n];

							// rebuild aggregate with requested operation and clear point clouds
							aggregate = buildAggregate<DIM>(primitives, "Bvh", false,
										operation == "Union" ? BooleanOperation::Union :
										(operation == "Intersection" ? BooleanOperation::Intersection :
										(operation == "Difference" ? BooleanOperation::Difference :
										BooleanOperation::None)));
							updateScene<DIM>(aggregate, boundaryPointCloud, interiorPointCloud,
											 domain, pointsIntersected, pointsRaymarched);
						}
					}
				}

				ImGui::EndCombo();
			}

		} else {
			if (ImGui::Button("Randomize Operations")) {
				// rebuild aggregate with random operations and clear point clouds
				aggregate = buildAggregate<DIM>(primitives, "Bvh", true);
				updateScene<DIM>(aggregate, boundaryPointCloud, interiorPointCloud,
								 domain, pointsIntersected, pointsRaymarched);
			}
		}
	}

	if (ImGui::Button("Add Samples to Visualize CSG")) {
		std::vector<Vector<DIM>> rayDirections;
		generateRandomRayDirections(nSamples, rayDirections);
		boundarySampler = constructSampler<DIM, float>(boundaryPointCloud, domain, nullptr, true);
		boundarySampler->generateSamples(nSamples, aggregate);
		clampToCsg<DIM>("intersect", nSamples, boundaryPointCloud, rayDirections, aggregate, pointsIntersected);
		clampToCsg<DIM>("raymarch", nSamples, boundaryPointCloud, rayDirections, aggregate, pointsRaymarched);

		if (interiorPointCloud->points.size() == 0 &&
			(pointsIntersected.size() > 0 || pointsRaymarched.size() > 0)) {
			interiorSampler = constructSampler<DIM, float>(interiorPointCloud, domain, nullptr);
			interiorSampler->generateSamples(nSamples, aggregate);
		}

		if (DIM == 2) {
			polyscope::registerPointCloud2D("Intersected_Point_Cloud", pointsIntersected);
			polyscope::registerPointCloud2D("Raymarched_Point_Cloud", pointsRaymarched);
			polyscope::registerPointCloud2D("Interior_Point_Cloud", interiorPointCloud->points);

		} else {
			polyscope::registerPointCloud("Intersected_Point_Cloud", pointsIntersected);
			polyscope::registerPointCloud("Raymarched_Point_Cloud", pointsRaymarched);
			polyscope::registerPointCloud("Interior_Point_Cloud", interiorPointCloud->points);
		}
	}

	ImGui::PopItemWidth();
}

template <int DIM>
void visualizeScene(const std::vector<std::shared_ptr<PolygonSoup<DIM>>>& soups,
					std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>>& primitives,
					std::shared_ptr<PointCloud<DIM>>& boundaryPointCloud,
					std::shared_ptr<PointCloud<DIM>>& interiorPointCloud,
					std::unique_ptr<Sampler<DIM>>& boundarySampler,
					std::unique_ptr<Sampler<DIM>>& interiorSampler,
					SamplingDomain<DIM>& domain,
					std::vector<Vector<DIM>>& pointsIntersected,
					std::vector<Vector<DIM>>& pointsRaymarched,
					std::shared_ptr<Aggregate<DIM>>& aggregate)
{
	// set a few options
	polyscope::options::programName = "Constructive Solid Geometry Tests";
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

		// register point clouds
		polyscope::registerPointCloud2D("Intersected_Point_Cloud", pointsIntersected);
		polyscope::registerPointCloud2D("Raymarched_Point_Cloud", pointsRaymarched);
		polyscope::registerPointCloud2D("Interior_Point_Cloud", interiorPointCloud->points);

	} else if (DIM == 3) {
		// register surface meshes
		for (int i = 0; i < (int)soups.size(); i++) {
			polyscope::registerSurfaceMesh("Polygon_Soup_" + std::to_string(i),
										   soups[i]->positions, soups[i]->indices);
		}

		// register point clouds
		polyscope::registerPointCloud("Intersected_Point_Cloud", pointsIntersected);
		polyscope::registerPointCloud("Raymarched_Point_Cloud", pointsRaymarched);
		polyscope::registerPointCloud("Interior_Point_Cloud", interiorPointCloud->points);
	}

	// register callback
	polyscope::state::userCallback = std::bind(&guiCallback<DIM>, soups.size(), std::ref(primitives),
											   std::ref(boundaryPointCloud), std::ref(interiorPointCloud),
											   std::ref(boundarySampler), std::ref(interiorSampler),
											   std::ref(domain), std::ref(pointsIntersected),
											   std::ref(pointsRaymarched), std::ref(aggregate));

	// give control to polyscope gui
	polyscope::show();
}

template <int DIM>
void run()
{
	// build scene
	std::vector<std::shared_ptr<PolygonSoup<DIM>>> soups;
	std::vector<std::vector<std::shared_ptr<Primitive<DIM>>>> primitives;
	buildSoupScene<DIM, float>(soups, primitives, nullptr, nullptr, true, true);

	// normalize scene
	VectorXf center;
	float radius;
	normalizeScene<DIM, float>(soups, primitives, center, radius, true);

	// build aggregate
	bool useRandomOperations = soups.size() > 2;
	std::shared_ptr<Aggregate<DIM>> aggregate = buildAggregate<DIM>(
											primitives, "Bvh", useRandomOperations, BooleanOperation::None);

	// seed cloud used to visualize csg
	std::shared_ptr<PointCloud<DIM>> boundaryPointCloud = std::make_shared<PointCloud<DIM>>();
	SamplingDomain<DIM> domain(aggregate->boundingBox());
	std::unique_ptr<Sampler<DIM>> boundarySampler = constructSampler<DIM, float>(boundaryPointCloud, domain, nullptr, true);
	boundarySampler->generateSamples(nSamples, aggregate);

	// generate random ray directions
	std::vector<Vector<DIM>> rayDirections;
	generateRandomRayDirections(nSamples, rayDirections);

	// intersect and raymarch points in ray directions
	std::vector<Vector<DIM>> pointsIntersected, pointsRaymarched;
	clampToCsg<DIM>("intersect", nSamples, boundaryPointCloud, rayDirections, aggregate, pointsIntersected);
	clampToCsg<DIM>("raymarch", nSamples, boundaryPointCloud, rayDirections, aggregate, pointsRaymarched);

	// generate points inside csg
	std::shared_ptr<PointCloud<DIM>> interiorPointCloud = std::make_shared<PointCloud<DIM>>();
	std::unique_ptr<Sampler<DIM>> interiorSampler = constructSampler<DIM, float>(interiorPointCloud, domain, nullptr);
	if (pointsIntersected.size() > 0 || pointsRaymarched.size() > 0) interiorSampler->generateSamples(nSamples, aggregate);

	// visualize scene
	visualizeScene<DIM>(soups, primitives, boundaryPointCloud, interiorPointCloud,
						boundarySampler, interiorSampler, domain, pointsIntersected,
						pointsRaymarched, aggregate);
}
*/
int main(int argc, const char *argv[]) {
	google::InitGoogleLogging(argv[0]);
#ifdef PROFILE
	Profiler::detect(argc, argv);
#endif
/*
	// configure the argument parser
	args::ArgumentParser parser("csg tests");
	args::Group group(parser, "", args::Group::Validators::DontCare);
	args::ValueFlag<int> dim(parser, "integer", "scene dimension", {"dim"});
	args::ValueFlag<int> nSamples(parser, "integer", "total samples", {"nSamples"});
	args::ValueFlag<int> nThreads(parser, "integer", "nThreads", {"nThreads"});
	args::ValueFlag<std::string> samplingStrategy(parser, "string", "sampler", {"sampler"});
	args::ValueFlag<std::string> csgFilename(parser, "string", "csg filename", {"csgFile"});
	args::ValueFlagList<std::string> lineSegmentFilenames(parser, "string", "line segment soup filenames", {"lsFile"});
	args::ValueFlagList<std::string> bezierFilenames(parser, "string", "bezier soup filenames", {"bFile"});
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
		if (DIM != 2 && DIM != 3) {
			std::cerr << "Only dimensions 2 and 3 are supported at this moment" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (DIM == 2 && !lineSegmentFilenames && !bezierFilenames && !triangleFilenames) {
		std::cerr << "Please specify line segment, bezier or triangle soup filenames" << std::endl;
		return EXIT_FAILURE;

	} else if (DIM == 3 && !triangleFilenames) {
		std::cerr << "Please specify triangle soup filenames" << std::endl;
		return EXIT_FAILURE;
	}

	if (args::get(samplingStrategy) == "random") {
		::samplingStrategy = "Random";

	} else if (args::get(samplingStrategy) == "poissondisk") {
		::samplingStrategy = "Poisson Disk";

	} else {
		std::cerr << "Using default random sampler" << std::endl;
		::samplingStrategy = "Random";
	}

	// set global flags
	if (nSamples) ::nSamples = args::get(nSamples);
	if (nThreads) fcpw::nThreads = args::get(nThreads);
	if (csgFilename) ::csgFilename = args::get(csgFilename);
	if (lineSegmentFilenames) {
		for (const auto lsf: args::get(lineSegmentFilenames)) {
			::lineSegmentFilenames.emplace_back(lsf);
		}
	}
	if (bezierFilenames) {
		for (const auto bsf: args::get(bezierFilenames)) {
			::bezierFilenames.emplace_back(bsf);
		}
	}
	if (triangleFilenames) {
		for (const auto tsf: args::get(triangleFilenames)) {
			::triangleFilenames.emplace_back(tsf);
		}
	}

	if (::lineSegmentFilenames.size() + ::bezierFilenames.size() + ::triangleFilenames.size() <= 1) {
		std::cerr << "Specify atleast 2 soups" << std::endl;
		return EXIT_FAILURE;
	}

	// run app
	if (DIM == 2) run<2>();
	else if (DIM == 3) run<3>();
*/
#ifdef PROFILE
	Profiler::dumphtml();
#endif
	return 0;
}
