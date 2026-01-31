/*
    GPU Watertight Intersection Test
    
    Tests the GPU watertight ray-triangle intersection by comparing it with
    GPU default intersection, mirroring the CPU watertight_tests.
    
    Part I (Typical behavior): Random rays from centroid - expected similar results.
    Part II (Edge behavior): Rays toward edge points - watertight should catch more hits.
    
    Run from build directory:
    > ./tests/gpu_watertight_tests --tFile ../tests/input/bunny.obj [--nQueries 10000]
    
    Note: Requires FCPW_ENABLE_GPU_SUPPORT=ON when building.
*/

#include <fcpw/fcpw.h>
#include <fcpw/fcpw_gpu.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include "args/args.hxx"

using namespace fcpw;

// Configuration
static int nQueries = 10000;
static std::string objFilename;
static bool verbose = false;

// Helper to load OBJ file
void loadObj(const std::string& objFilePath,
             std::vector<Vector<3>>& positions,
             std::vector<Vector3i>& indices)
{
    std::ifstream in(objFilePath);
    if (!in.is_open()) {
        std::cerr << "Unable to open file: " << objFilePath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    positions.clear();
    indices.clear();

    while (getline(in, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.emplace_back(Vector3(x, y, z));
        
        } else if (token == "f") {
            std::vector<int> faceIndices;
            while (ss >> token) {
                // Handle "v", "v/vt", "v/vt/vn", "v//vn" formats
                std::stringstream tokenStream(token);
                std::string indexStr;
                std::getline(tokenStream, indexStr, '/');
                int idx = std::stoi(indexStr) - 1; // OBJ is 1-indexed
                faceIndices.push_back(idx);
            }
            // Triangulate if needed
            for (size_t i = 1; i + 1 < faceIndices.size(); i++) {
                indices.emplace_back(Vector3i(faceIndices[0], faceIndices[i], faceIndices[i+1]));
            }
        }
    }
    in.close();
}

// Compute mesh centroid
Vector3 computeCentroid(const std::vector<Vector<3>>& positions)
{
    Vector3 centroid = Vector3::Zero();
    for (const auto& p : positions) {
        centroid += p;
    }
    return centroid / static_cast<float>(positions.size());
}

// Generate a random direction on the sphere
Vector3 randomDirection()
{
    while (true) {
        Vector3 v = uniformRealRandomVector<3>(-1.0f, 1.0f);
        float lenSq = v.squaredNorm();
        if (lenSq > 0.0001f && lenSq <= 1.0f) {
            return v.normalized();
        }
    }
}

// Structure to hold test statistics
struct TestStatistics {
    int totalRays = 0;
    int defaultHits = 0;
    int watertightHits = 0;
    int bothHit = 0;
    int bothMiss = 0;
    int onlyDefaultHit = 0;
    int onlyWatertightHit = 0;
    int distanceMismatches = 0;
    float maxDistanceDiff = 0.0f;
    double totalDistanceDiff = 0.0;
    
    void print(const std::string& testName) const {
        std::cout << "\n========================================" << std::endl;
        std::cout << testName << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total rays:           " << totalRays << std::endl;
        std::cout << "Default hits:         " << defaultHits << " (" 
                  << std::fixed << std::setprecision(2) 
                  << (100.0 * defaultHits / totalRays) << "%)" << std::endl;
        std::cout << "Watertight hits:      " << watertightHits << " (" 
                  << (100.0 * watertightHits / totalRays) << "%)" << std::endl;
        std::cout << "Both hit:             " << bothHit << std::endl;
        std::cout << "Both miss:            " << bothMiss << std::endl;
        std::cout << "Only default hit:     " << onlyDefaultHit << std::endl;
        std::cout << "Only watertight hit:  " << onlyWatertightHit << std::endl;
        
        if (bothHit > 0) {
            std::cout << "\nWhen both hit:" << std::endl;
            std::cout << "  Distance mismatches:  " << distanceMismatches 
                      << " (" << (100.0 * distanceMismatches / bothHit) << "%)" << std::endl;
            std::cout << "  Max distance diff:    " << maxDistanceDiff << std::endl;
            std::cout << "  Avg distance diff:    " << (totalDistanceDiff / bothHit) << std::endl;
        }
        
        std::cout << "\nSummary:" << std::endl;
        int diff = watertightHits - defaultHits;
        if (diff > 0) {
            std::cout << "  Watertight found " << diff << " more intersections than default." << std::endl;
        } else if (diff < 0) {
            std::cout << "  WARNING: Default found " << (-diff) << " more intersections than watertight!" << std::endl;
        } else {
            std::cout << "  Both methods found the same number of intersections." << std::endl;
        }
    }
};

// Run GPU comparison test
TestStatistics runGPUComparison(GPUScene<3>& gpuScene,
                                const Vector3& origin,
                                const std::vector<Vector3>& targets)
{
    TestStatistics stats;
    stats.totalRays = static_cast<int>(targets.size());
    
    const float distanceThreshold = 1e-5f;
    
    // Build GPU rays
    std::vector<GPURay> gpuRays;
    gpuRays.reserve(targets.size());
    for (const auto& target : targets) {
        Vector3 direction = (target - origin).normalized();
        float3 o = {origin[0], origin[1], origin[2]};
        float3 d = {direction[0], direction[1], direction[2]};
        gpuRays.emplace_back(GPURay(o, d));
    }
    
    // Run default intersection
    std::vector<GPUInteraction> defaultInteractions;
    gpuScene.intersect(gpuRays, defaultInteractions, false, false);  // watertight=false
    
    // Run watertight intersection
    std::vector<GPUInteraction> watertightInteractions;
    gpuScene.intersect(gpuRays, watertightInteractions, false, true);  // watertight=true
    
    // Compare results
    for (size_t i = 0; i < targets.size(); i++) {
        bool hitDefault = (defaultInteractions[i].index != FCPW_GPU_UINT_MAX);
        bool hitWatertight = (watertightInteractions[i].index != FCPW_GPU_UINT_MAX);
        
        if (hitDefault) stats.defaultHits++;
        if (hitWatertight) stats.watertightHits++;
        
        if (hitDefault && hitWatertight) {
            stats.bothHit++;
            float distDiff = std::abs(defaultInteractions[i].d - watertightInteractions[i].d);
            stats.totalDistanceDiff += distDiff;
            stats.maxDistanceDiff = std::max(stats.maxDistanceDiff, distDiff);
            if (distDiff > distanceThreshold) {
                stats.distanceMismatches++;
            }
        } else if (!hitDefault && !hitWatertight) {
            stats.bothMiss++;
        } else if (hitDefault && !hitWatertight) {
            stats.onlyDefaultHit++;
            if (verbose) {
                std::cout << "  ONLY DEFAULT HIT ray " << i 
                          << ": d=" << defaultInteractions[i].d 
                          << " prim=" << defaultInteractions[i].index << std::endl;
            }
        } else {
            stats.onlyWatertightHit++;
        }
    }
    
    return stats;
}

// Part I: Random directions from centroid
TestStatistics testRandomDirections(GPUScene<3>& gpuScene, const Vector3& centroid, int n)
{
    std::vector<Vector3> targets;
    targets.reserve(n);
    
    for (int i = 0; i < n; i++) {
        Vector3 dir = randomDirection();
        targets.push_back(centroid + dir * 1000.0f);
    }
    
    return runGPUComparison(gpuScene, centroid, targets);
}

// Part II: Rays toward edge points
TestStatistics testEdgeTargets(GPUScene<3>& gpuScene,
                               const std::vector<Vector<3>>& positions,
                               const std::vector<Vector3i>& indices,
                               const Vector3& centroid,
                               int n)
{
    std::vector<Vector3> targets;
    targets.reserve(n);
    
    int numTriangles = static_cast<int>(indices.size());
    
    for (int i = 0; i < n; i++) {
        // Pick a random triangle
        int triIdx = static_cast<int>(uniformRealRandomNumber(0.0f, static_cast<float>(numTriangles) - 0.001f));
        const Vector3i& tri = indices[triIdx];
        
        // Pick one of the three edges
        int edgeIdx = static_cast<int>(uniformRealRandomNumber(0.0f, 2.999f));
        
        // Get the two vertices of the edge
        int v0Idx = tri[edgeIdx];
        int v1Idx = tri[(edgeIdx + 1) % 3];
        const Vector3& v0 = positions[v0Idx];
        const Vector3& v1 = positions[v1Idx];
        
        // Pick a random point on the edge
        float t = uniformRealRandomNumber(0.0f, 1.0f);
        Vector3 edgePoint = v0 + t * (v1 - v0);
        
        targets.push_back(edgePoint);
    }
    
    return runGPUComparison(gpuScene, centroid, targets);
}

void run()
{
    // Load mesh
    std::vector<Vector<3>> positions;
    std::vector<Vector3i> indices;
    
    std::cout << "Loading mesh: " << objFilename << std::endl;
    loadObj(objFilename, positions, indices);
    std::cout << "Loaded " << positions.size() << " vertices and " 
              << indices.size() << " triangles" << std::endl;
    
    // Compute centroid
    Vector3 centroid = computeCentroid(positions);
    std::cout << "Centroid: (" << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ")" << std::endl;
    
    // Build CPU scene first
    Scene<3> scene;
    scene.setObjectCount(1);
    scene.setObjectVertices(positions, 0);
    scene.setObjectTriangles(indices, 0);
    scene.build(AggregateType::Bvh_SurfaceArea, false, false, false);
    
    // Transfer to GPU
    std::cout << "\nInitializing GPU..." << std::endl;
    std::filesystem::path fcpwDirectoryPath = std::filesystem::current_path().parent_path();
    GPUScene<3> gpuScene(fcpwDirectoryPath.string(), false);  // hasSilhouetteGeometry=false
    gpuScene.transferToGPU(scene);
    
    std::cout << "\nRunning " << nQueries << " queries for each test...\n" << std::endl;
    
    // Part I: Random directions
    std::cout << "Running Part I: Random directions from centroid..." << std::endl;
    TestStatistics statsRandom = testRandomDirections(gpuScene, centroid, nQueries);
    statsRandom.print("Part I: Typical Behavior (Random Directions)");
    
    // Part II: Edge targets
    std::cout << "\nRunning Part II: Rays toward edge points..." << std::endl;
    if (verbose) std::cout << "\nFailure details:" << std::endl;
    TestStatistics statsEdge = testEdgeTargets(gpuScene, positions, indices, centroid, nQueries);
    statsEdge.print("Part II: Edge Behavior (Rays Toward Edge Points)");
    
    // Overall summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "OVERALL SUMMARY (GPU)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Part I (Random):  Watertight found " 
              << (statsRandom.onlyWatertightHit - statsRandom.onlyDefaultHit)
              << " more hits than default" << std::endl;
    std::cout << "Part II (Edges):  Watertight found " 
              << (statsEdge.onlyWatertightHit - statsEdge.onlyDefaultHit)
              << " more hits than default" << std::endl;
    
    if (statsEdge.onlyWatertightHit > statsEdge.onlyDefaultHit + 10) {
        std::cout << "\n✓ As expected, GPU watertight intersection catches significantly more " << std::endl;
        std::cout << "  edge-case intersections in Part II." << std::endl;
    }
}

int main(int argc, const char *argv[])
{
    // Configure the argument parser
    args::ArgumentParser parser("GPU Watertight intersection comparison tests");
    args::Group group(parser, "", args::Group::Validators::DontCare);
    args::ValueFlag<int> nQueriesFlag(parser, "integer", "number of queries per test", {"nQueries"});
    args::ValueFlag<std::string> triangleFilename(parser, "string", "triangle mesh OBJ file", {"tFile"});
    args::Flag verboseFlag(group, "bool", "print details about failures", {"verbose"});

    // Parse args
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

    if (!triangleFilename) {
        std::cerr << "Please specify a triangle mesh file with --tFile" << std::endl;
        return EXIT_FAILURE;
    }

    objFilename = args::get(triangleFilename);
    if (nQueriesFlag) {
        nQueries = args::get(nQueriesFlag);
    }
    if (verboseFlag) {
        verbose = true;
    }

    run();
    return 0;
}
