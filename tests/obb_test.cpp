
#include <fcpw/utilities/scene_loader.h>
#include <atomic>
#include <thread>
#include "CLI11.hpp"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"

using namespace fcpw;

void viz_obb(OrientedBoundingBox<3> obb, std::string name) {
    
    Eigen::Matrix3f u = obb.rot_mat();
    Vector3 c = obb.center();
    Vector3 min = c - obb.e;
    Vector3 max = c + obb.e;
    
    std::vector<Vector3> v;
    std::vector<std::array<size_t,2>> i;

    auto edge = [&](Vector3 a, Vector3 b) {
        i.push_back({v.size(), v.size() + 1});
        v.push_back(u * a);
        v.push_back(u * b);
    };

    edge(min, Vector3{max.x(), min.y(), min.z()});
    edge(min, Vector3{min.x(), max.y(), min.z()});
    edge(min, Vector3{min.x(), min.y(), max.z()});
    edge(max, Vector3{min.x(), max.y(), max.z()});
    edge(max, Vector3{max.x(), min.y(), max.z()});
    edge(max, Vector3{max.x(), max.y(), min.z()});
    edge(Vector3{min.x(), max.y(), min.z()}, Vector3{max.x(), max.y(), min.z()});
    edge(Vector3{min.x(), max.y(), min.z()}, Vector3{min.x(), max.y(), max.z()});
    edge(Vector3{min.x(), min.y(), max.z()}, Vector3{max.x(), min.y(), max.z()});
    edge(Vector3{min.x(), min.y(), max.z()}, Vector3{min.x(), max.y(), max.z()});
    edge(Vector3{max.x(), min.y(), min.z()}, Vector3{max.x(), max.y(), min.z()});
    edge(Vector3{max.x(), min.y(), min.z()}, Vector3{max.x(), min.y(), max.z()});

    polyscope::registerCurveNetwork(name, v, i);
}

void viz_bb(BoundingBox<3> bb, std::string name) {
    
    Vector3 min = bb.pMin;
    Vector3 max = bb.pMax;
    
    std::vector<Vector3> v;
    std::vector<std::array<size_t,2>> i;

    auto edge = [&](Vector3 a, Vector3 b) {
        i.push_back({v.size(), v.size() + 1});
        v.push_back(a);
        v.push_back(b);
    };

    edge(min, Vector3{max.x(), min.y(), min.z()});
    edge(min, Vector3{min.x(), max.y(), min.z()});
    edge(min, Vector3{min.x(), min.y(), max.z()});
    edge(max, Vector3{min.x(), max.y(), max.z()});
    edge(max, Vector3{max.x(), min.y(), max.z()});
    edge(max, Vector3{max.x(), max.y(), min.z()});
    edge(Vector3{min.x(), max.y(), min.z()}, Vector3{max.x(), max.y(), min.z()});
    edge(Vector3{min.x(), max.y(), min.z()}, Vector3{min.x(), max.y(), max.z()});
    edge(Vector3{min.x(), min.y(), max.z()}, Vector3{max.x(), min.y(), max.z()});
    edge(Vector3{min.x(), min.y(), max.z()}, Vector3{min.x(), max.y(), max.z()});
    edge(Vector3{max.x(), min.y(), min.z()}, Vector3{max.x(), max.y(), min.z()});
    edge(Vector3{max.x(), min.y(), min.z()}, Vector3{max.x(), min.y(), max.z()});

    polyscope::registerCurveNetwork(name, v, i);
}

void viz_object(const std::vector<Vector3>& points) {

    OrientedBoundingBox<3> obb;
    obb.fromPoints(points);

    BoundingBox<3> bb;
    bb.fromPoints(points);

    viz_bb(bb, "AABB");
    viz_bb(obb.box(), "OBB->AABB");
    viz_obb(obb, "OBB");
    
    Transform<3> id = Transform<3>::Identity();
    id.translate(Vector3{0.0f, 5.0f, 0.0f});
    viz_obb(obb.transform(id), "OBB->T->OBB");
}

void viz()
{
    Scene<3> scene;
	SceneLoader<3> sceneLoader;
	sceneLoader.loadFiles(scene, false);
	scene.build(AggregateType::Baseline, BoundingVolumeType::AABB, false, false);
	SceneData<3> *sceneData = scene.getSceneData();

	// set a few options
	polyscope::options::programName = "OBB Test";
	polyscope::options::verbosity = 0;
	polyscope::options::usePrefsFile = false;
	polyscope::options::autocenterStructures = false;

	// initialize polyscope
	polyscope::init();
    for (int i = 0; i < (int)sceneData->soups.size(); i++) {
        std::string meshName = "Polygon_Soup_" + std::to_string(i);

        if (sceneData->soupToObjectsMap[i][0].first == ObjectType::Triangles) {
            // register surface mesh
            int N = (int)sceneData->soups[i].indices.size()/3;
            std::vector<std::vector<int>> indices(N, std::vector<int>(3));
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < 3; k++) {
                    indices[j][k] = sceneData->soups[i].indices[3*j + k];
                }
            }

            viz_object(sceneData->soups[i].positions);

            polyscope::registerSurfaceMesh(meshName, sceneData->soups[i].positions, indices);
        }
    }

	// give control to polyscope gui
	polyscope::show();
}

int main(int argc, const char *argv[]) {

	CLI::App args{"FCPW OBB test"};
	std::string file;

	args.add_option("-s,--scene", file, "triangle soup file");

    CLI11_PARSE(args, argc, argv);

	files.emplace_back(std::make_pair(file, LoadingOption::ObjTriangles));

	viz();
	return 0;
}
