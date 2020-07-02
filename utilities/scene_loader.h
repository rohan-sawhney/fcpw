#pragma once

#include "fcpw.h"

namespace fcpw {

enum class LoadingOption {
	ObjLineSegments,
	ObjTriangles
};

std::vector<std::pair<std::string, LoadingOption>> files;
std::string instanceFilename;
std::string csgFilename;

template<size_t DIM>
class SceneLoader {
public:
	// loads files; NOTE: this method does not build the scene aggregate/accelerator,
	// it just populates its geometry
	void loadFiles(Scene<DIM>& scene, bool computeNormals);
};

} // namespace fcpw

#include "scene_loader.inl"
