#pragma once

#include "fcpw.h"

enum class LoadingOption {
	ObjLineSegments,
	ObjTriangles
};

namespace fcpw {

template<size_t DIM>
class SceneLoader {
public:
	// constructor
	SceneLoader();

	// loads files; NOTE: this method does not build the scene aggregate/accelerator,
	// it just populates its geometry
	void loadFiles(Scene<DIM>& scene, bool computeNormals);

	// members
	std::vector<std::pair<std::string, LoadingOption>> files;
	std::string instanceFilename;
	std::string csgFilename;
};

} // namespace fcpw

#include "scene_loader.inl"
