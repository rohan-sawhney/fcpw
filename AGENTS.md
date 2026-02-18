# Repository Guidelines

## Project Structure & Module Organization
- `include/fcpw/`: header-only C++ library (core, geometry, aggregates, GPU shaders).
- `python/`: nanobind extension sources (`fcpw_py.cpp`) and CMake for Python package builds.
- `demos/`: C++ and Python usage examples.
- `tests/`: C++ test binaries (`aggregate_tests`, `csg_tests`, optional `gpu_tests`), Python test driver (`fcpw_tests.py`), and test assets in `tests/input/`.
- `deps/`: third-party dependencies (submodules and vendored code). Avoid modifying unless dependency updates are intentional.
- `build/`: local CMake/scikit-build output; do not commit generated artifacts.

## Build, Test, and Development Commands
- Configure/build C++ project:
  ```bash
  cmake -S . -B build -DFCPW_BUILD_DEMO=ON -DFCPW_BUILD_TESTS=ON
  cmake --build build -j8
  ```
- Run demo:
  ```bash
  ./build/demos/demo
  ```
- Run C++ tests (examples):
  ```bash
  ./build/tests/aggregate_tests --dim=3 --tFile tests/input/bunny.obj --nQueries=1024 --checkCorrectness
  ./build/tests/csg_tests --dim=2 --lFile tests/input/spiral.obj --csgFile tests/input/csg.txt --instanceFile tests/input/instances2d.txt
  ```
- Build/install Python bindings locally:
  ```bash
  pip install .
  ```
- Run Python test driver:
  ```bash
  python3 tests/fcpw_tests.py --file_path=tests/input/bunny.obj --dim=3 --n_queries=1024
  ```

## GPU Backend Selection
- Slang backend (original): configure with `-DFCPW_ENABLE_GPU_SUPPORT=ON` and keep `FCPW_ENABLE_CUDA_SUPPORT=OFF`.
- CUDA backend (new): configure with `-DFCPW_ENABLE_CUDA_SUPPORT=ON` and optionally disable Slang via `-DFCPW_ENABLE_GPU_SUPPORT=OFF`.
- Legacy includes still use `include/fcpw/fcpw_gpu.h`; backend selection is compile-time (`FCPW_USE_SLANG` or `FCPW_USE_CUDA`).

## Coding Style & Naming Conventions
- C++: C++17, 4-space indentation, keep headers self-contained, prefer `.h`/`.inl` patterns used in `include/fcpw/`.
- Python: follow PEP 8-style spacing and `snake_case` naming (`scene_3D`, `find_closest_points`).
- Naming: types/classes use `PascalCase`; functions/variables use `camelCase` in C++ and `snake_case` in Python.
- No top-level formatter config is enforced in this repo; match surrounding file style exactly.

## Testing Guidelines
- Prefer deterministic correctness runs (`--checkCorrectness`) before large performance runs.
- Keep new test assets under `tests/input/` and reference them with repo-relative paths.
- For GPU changes, run both CPU and GPU paths (`gpu_tests` and Python `--run_gpu_queries`) when available.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative, lower-case commit subjects (for example: `update instruction`, `remove stub`, `typo`).
- Keep commits focused by concern (core query logic, GPU path, bindings, tests/docs).
- PRs should include:
  - concise summary of behavioral changes,
  - test commands run and key outputs,
  - linked issue(s) when applicable,
  - screenshots only for visualization/demo UI changes.
