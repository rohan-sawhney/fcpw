#include <fstream>
#include <sstream>

namespace fcpw {

template<size_t DIM>
inline SceneLoader<DIM>::SceneLoader():
instanceFilename(""),
csgFilename("")
{

}

// helper class for loading polygons from obj files
struct Index {
	Index() {}

	Index(int v, int vt, int vn) : position(v), uv(vt), normal(vn) {}

	bool operator<(const Index& i) const {
		if (position < i.position) return true;
		if (position > i.position) return false;
		if (uv < i.uv) return true;
		if (uv > i.uv) return false;
		if (normal < i.normal) return true;
		if (normal > i.normal) return false;

		return false;
	}

	int position;
	int uv;
	int normal;
};

// helper function for loading polygons from obj files
inline Index parseFaceIndex(const std::string& token) {
	std::stringstream in(token);
	std::string indexString;
	int indices[3] = {1, 1, 1};

	int i = 0;
	while (std::getline(in, indexString, '/')) {
		if (indexString != "\\") {
			std::stringstream ss(indexString);
			ss >> indices[i++];
		}
	}

	// decrement since indices in OBJ files are 1-based
	return Index(indices[0] - 1, indices[1] - 1, indices[2] - 1);
}

inline void readLineSegmentSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// initialize
	std::ifstream in(filename);
	if (in.is_open() == false) {
		std::cerr << "Unable to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		std::string token;
		ss >> token;

		if (token == "v") {
			float x, y, z;
			ss >> x >> y >> z;

			soup.positions.emplace_back(Vector3(x, y, z));

		} else if (token == "f" || token == "l") {
			bool tokenIsF = token == "f";
			std::vector<int> indices;

			while (ss >> token) {
				Index index = parseFaceIndex(token);

				if (index.position < 0) {
					getline(in, line);
					size_t i = line.find_first_not_of("\t\n\v\f\r ");
					index = parseFaceIndex(line.substr(i));
				}

				if (tokenIsF) indices.emplace_back(index.position);
				else soup.indices.emplace_back(index.position);
			}

			if (tokenIsF) {
				int F = (int)indices.size();
				for (int i = 0; i < F - 1; i++) {
					int j = (i + 1)%F;
					soup.indices.emplace_back(indices[i]);
					soup.indices.emplace_back(indices[j]);
				}
			}
		}
	}

	// close
	in.close();
}

inline void readTriangleSoupFromOBJFile(const std::string& filename, PolygonSoup<3>& soup)
{
#ifdef PROFILE
	PROFILE_SCOPED();
#endif

	// initialize
	std::ifstream in(filename);
	if (in.is_open() == false) {
		std::cerr << "Unable to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		std::string token;
		ss >> token;

		if (token == "v") {
			float x, y, z;
			ss >> x >> y >> z;

			soup.positions.emplace_back(Vector3(x, y, z));

		} else if (token == "vt") {
			float u, v;
			ss >> u >> v;

			soup.textureCoordinates.emplace_back(Vector2(u, v));

		} else if (token == "f") {
			while (ss >> token) {
				Index index = parseFaceIndex(token);

				if (index.position < 0) {
					getline(in, line);
					size_t i = line.find_first_not_of("\t\n\v\f\r ");
					index = parseFaceIndex(line.substr(i));
				}

				soup.indices.emplace_back(index.position);
				soup.tIndices.emplace_back(index.uv);
			}
		}
	}

	// close
	in.close();

	if (soup.textureCoordinates.size() == 0) {
		soup.tIndices.clear();
	}
}

template<size_t DIM>
inline void loadInstanceTransforms(const std::string& filename,
								   std::vector<std::vector<Transform<DIM>>>& instanceTransforms)
{
	// load file
	std::ifstream in(filename);
	if (in.is_open() == false) {
		std::cerr << "Unable to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// parse transforms
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		int objectIndex;
		ss >> objectIndex;

		int nTransform = (int)instanceTransforms[objectIndex].size();
		instanceTransforms[objectIndex].emplace_back(Transform<DIM>());

		for (int i = 0; i <= DIM; i++) {
			for (int j = 0; j <= DIM; j++) {
				ss >> instanceTransforms[objectIndex][nTransform].matrix()(i, j);
			}
		}
	}

	// close file
	in.close();
}

inline void loadCsgTree(const std::string& filename, std::unordered_map<int, CsgTreeNode>& csgTree)
{
	// load scene
	std::ifstream in(filename);
	if (in.is_open() == false) {
		std::cerr << "Unable to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	// parse obj format
	std::string line;
	while (getline(in, line)) {
		std::stringstream ss(line);
		int node;
		ss >> node;

		std::string operationStr, child1Str, child2Str;
		ss >> operationStr >> child1Str >> child2Str;

		std::size_t found1 = child1Str.find_last_of("_");
		std::size_t found2 = child2Str.find_last_of("_");
		csgTree[node].child1 = std::stoi(child1Str.substr(found1 + 1));
		csgTree[node].child2 = std::stoi(child2Str.substr(found2 + 1));
		csgTree[node].isLeafChild1 = child1Str.find("node_") == std::string::npos;
		csgTree[node].isLeafChild2 = child2Str.find("node_") == std::string::npos;
		csgTree[node].operation = operationStr == "Union" ? BooleanOperation::Union :
								 (operationStr == "Intersection" ? BooleanOperation::Intersection :
								 (operationStr == "Difference" ? BooleanOperation::Difference : BooleanOperation::None));
	}

	// close file
	in.close();
}

template<size_t DIM>
inline void SceneLoader<DIM>::loadFiles(Scene<DIM>& scene, bool computeNormals)
{
	// inform the scene about the various object types about to be loaded
	int nFiles = (int)files.size();
	std::vector<std::vector<PrimitiveType>> objectTypes(nFiles);

	for (int i = 0; i < nFiles; i++) {
		if (files[i].second == LoadingOption::ObjLineSegments) {
			objectTypes.emplace_back(PrimitiveType::LineSegment);

		} else if (files[i].second == LoadingOption::ObjTriangles) {
			objectTypes.emplace_back(PrimitiveType::Triangle);
		}
	}

	scene.setObjectTypes(objectTypes);

	// fill out scene data (soups, lineSegmentObjects and triangleObjects) directly
	SceneData<DIM> *sceneData = scene.getSceneData();
	// TODO: fill sceneData

	// load instance transforms
	if (!instanceFilename.empty()) {
		loadInstanceTransforms<DIM>(instanceFilename, sceneData->instanceTransforms);
	}

	// load csg tree
	if (!csgFilename.empty()) {
		loadCsgTree(csgFilename, sceneData->csgTree);

		if (!computeNormals) {
			std::cout << "SceneLoader::loadFiles(): Turning on normal computation, required for distance queries to csg"
					  << std::endl;
			computeNormals = true;
		}
	}

	// compute normals
	if (computeNormals) {
		for (int i = 0; i < nFiles; i++) {
			scene.computeNormals(i);
		}
	}
}

} // namespace fcpw
