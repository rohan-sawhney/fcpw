import numpy as np
import polyscope as ps
import warp as wp
import fcpw
import time
import argparse
from pathlib import Path
from typing import Callable, Optional

class BoundingBox:
    def __init__(self, p_min, p_max):
        self.p_min = np.array(p_min)
        self.p_max = np.array(p_max)

    def max_dimension(self):
        return np.argmax(self.p_max - self.p_min)

def compute_bounding_box(positions):
    p_min = np.min(positions, axis=0)
    p_max = np.max(positions, axis=0)

    return BoundingBox(p_min, p_max)

def generate_scattered_points_and_rays(n_queries, bounding_box, dim):
    epsilon = 1e-6
    extents = bounding_box.p_max - bounding_box.p_min
    scattered_points = bounding_box.p_min + extents * np.random.rand(n_queries, dim)
    random_directions = np.random.rand(n_queries, dim) * 2 - 1
    random_squared_radii = 0.1 * np.random.rand(n_queries) * np.linalg.norm(extents)
    if np.abs(extents[dim - 1]) < 5 * epsilon:
        scattered_points[:, dim - 1] = 0
        random_directions[:, dim - 1] = 0
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]

    return scattered_points, random_directions, random_squared_radii

def generate_edge_rays(n_queries, positions, indices, centroid):
    """Generate rays from centroid toward random edge points on the mesh (3D only)."""
    n_triangles = len(indices)
    edge_points = []
    
    for _ in range(n_queries):
        # Pick a random triangle
        tri_idx = np.random.randint(0, n_triangles)
        tri = indices[tri_idx]
        
        # Pick a random edge (0, 1, or 2)
        edge_idx = np.random.randint(0, 3)
        v0_idx = tri[edge_idx]
        v1_idx = tri[(edge_idx + 1) % 3]
        
        # Pick a random point on the edge
        t = np.random.rand()
        edge_point = (1 - t) * positions[v0_idx] + t * positions[v1_idx]
        edge_points.append(edge_point)
    
    edge_points = np.array(edge_points, dtype=np.float32)
    ray_origins = np.tile(centroid.astype(np.float32), (n_queries, 1))
    ray_directions = edge_points - ray_origins
    ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, None]
    
    return ray_origins, ray_directions

def tag_interior_points(scene, query_points):
    n_queries = len(query_points)
    is_interior = fcpw.uint32_list()
    scene.contains(query_points, is_interior)

    return is_interior

def isolate_interior_points(scene, query_points):
    is_interior = tag_interior_points(scene, query_points)
    interior_points = [query_point for query_point, interior in zip(query_points, is_interior) if interior]

    return np.array(interior_points)

def load_obj(file_path, dim):
    positions = []
    indices = []

    with open(file_path, 'r') as file:
        for line in file:
            if dim == 2:
                if line.startswith('v '):
                    position = list(map(float, line.strip().split()[1:]))[0:2]
                    positions.append(np.array(position, dtype=np.float32, order='C'))
                elif line.startswith('l '):
                    index = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                    indices.append(np.array(index, dtype=np.int32, order='C'))
                elif line.startswith('f '):
                    index = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                    F = len(index)
                    for i in range(F - 1):
                        j = (i + 1) % F
                        indices.append(np.array([index[i], index[j]], dtype=np.int32, order='C'))

            elif dim == 3:
                if line.startswith('v '):
                    position = list(map(float, line.strip().split()[1:]))
                    positions.append(np.array(position, dtype=np.float32, order='C'))
                elif line.startswith('f '):
                    index = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                    indices.append(np.array(index, dtype=np.int32, order='C'))

    return np.array(positions), np.array(indices)

def ignore_silhouette(angle: float, index: int):
    return False # stub

def load_fcpw_scene(positions, indices, aggregate_type, compute_silhouettes, vectorize,
                    dim, print_stats = False, reduce_memory_footprint = False):
    scene = None

    if dim == 2:
        scene = fcpw.scene_2D()
        scene.set_object_count(1)
        scene.set_object_line_segments(indices, 0)

    elif dim == 3:
        scene = fcpw.scene_3D()
        scene.set_object_count(1)
        scene.set_object_triangles(indices, 0)

    if scene is not None:
        scene.set_object_vertices(positions, 0)

        if compute_silhouettes:
            scene.compute_silhouettes(None)

        scene.build(aggregate_type, vectorize, print_stats, reduce_memory_footprint)

    return scene

def init_gpu_data(n_queries, dim):
    scene = None
    fcpw_directory_path = str(Path.cwd().parent)
    if dim == 2:
        scene = fcpw.gpu_scene_2D(fcpw_directory_path, True)

    elif dim == 3:
        scene = fcpw.gpu_scene_3D(fcpw_directory_path, True)

    return scene

def run_cpu_ray_intersection_queries(scene, ray_origins, ray_directions, ray_distance_bounds,
                                     dim, run_bundled_queries = True, watertight = False):
    n_queries = len(ray_origins)
    if run_bundled_queries:
        interactions = None
        if dim == 2:
            interactions = fcpw.interaction_2D_list()

        elif dim == 3:
            interactions = fcpw.interaction_3D_list()

        start_time = time.perf_counter()
        scene.intersect(ray_origins, ray_directions, ray_distance_bounds, interactions, False, watertight)
        end_time = time.perf_counter()
        mode_str = " (watertight)" if watertight else ""
        print(f"{n_queries} ray intersection queries{mode_str} took {end_time - start_time} seconds")

        return interactions

    else:
        start_time = time.perf_counter()
        interactions = [None] * n_queries

        if dim == 2:
            for q in range(n_queries):
                ray = fcpw.ray_2D(ray_origins[q], ray_directions[q], ray_distance_bounds[q])
                interactions[q] = fcpw.interaction_2D()
                hit = scene.intersect(ray, interactions[q], False, watertight)

        elif dim == 3:
            for q in range(n_queries):
                ray = fcpw.ray_3D(ray_origins[q], ray_directions[q], ray_distance_bounds[q])
                interactions[q] = fcpw.interaction_3D()
                hit = scene.intersect(ray, interactions[q], False, watertight)

        end_time = time.perf_counter()
        mode_str = " (watertight)" if watertight else ""
        print(f"{n_queries} ray intersection queries{mode_str} took {end_time - start_time} seconds")

        return interactions

def branch_traversal_weight(r2: float):
    return 1.0 # stub

def run_cpu_sphere_intersection_queries(scene, sphere_centers, sphere_squared_radii,
                                        rand_nums, dim, run_bundled_queries = True):
    n_queries = len(sphere_centers)
    if run_bundled_queries:
        interactions = None
        if dim == 2:
            interactions = fcpw.interaction_2D_list()

        elif dim == 3:
            interactions = fcpw.interaction_3D_list()

        start_time = time.perf_counter()
        scene.intersect(sphere_centers, sphere_squared_radii, interactions, rand_nums, None)
        end_time = time.perf_counter()
        print(f"{n_queries} sphere intersection queries took {end_time - start_time} seconds")

        return interactions

    else:
        start_time = time.perf_counter()
        interactions = [None] * n_queries

        if dim == 2:
            for q in range(n_queries):
                sphere = fcpw.bounding_sphere_2D(sphere_centers[q], sphere_squared_radii[q])
                interactions[q] = fcpw.interaction_2D()
                hits = scene.intersect(sphere, interactions[q], rand_nums[q], None)

        elif dim == 3:
            for q in range(n_queries):
                sphere = fcpw.bounding_sphere_3D(sphere_centers[q], sphere_squared_radii[q])
                interactions[q] = fcpw.interaction_3D()
                hits = scene.intersect(sphere, interactions[q], rand_nums[q], None)

        end_time = time.perf_counter()
        print(f"{n_queries} sphere intersection queries took {end_time - start_time} seconds")

        return interactions

def run_cpu_closest_point_queries(scene, query_points, squared_max_radii, dim, run_bundled_queries = True):
    n_queries = len(query_points)
    if run_bundled_queries:
        interactions = None
        if dim == 2:
            interactions = fcpw.interaction_2D_list()

        elif dim == 3:
            interactions = fcpw.interaction_3D_list()

        start_time = time.perf_counter()
        scene.find_closest_points(query_points, squared_max_radii, interactions)
        end_time = time.perf_counter()
        print(f"{n_queries} closest point queries took {end_time - start_time} seconds")

        return interactions

    else:
        start_time = time.perf_counter()
        interactions = [None] * n_queries

        if dim == 2:
            for q in range(n_queries):
                interactions[q] = fcpw.interaction_2D()
                found = scene.find_closest_point(query_points[q], interactions[q], squared_max_radii[q])

        elif dim == 3:
            for q in range(n_queries):
                interactions[q] = fcpw.interaction_3D()
                found = scene.find_closest_point(query_points[q], interactions[q], squared_max_radii[q])

        end_time = time.perf_counter()
        print(f"{n_queries} closest point queries took {end_time - start_time} seconds")

        return interactions

def run_cpu_closest_silhouette_point_queries(scene, query_points, squared_max_radii, flip_normal_orientation,
                                             dim, run_bundled_queries = True):
    n_queries = len(query_points)
    if run_bundled_queries:
        interactions = None
        if dim == 2:
            interactions = fcpw.interaction_2D_list()

        elif dim == 3:
            interactions = fcpw.interaction_3D_list()

        start_time = time.perf_counter()
        scene.find_closest_silhouette_points(query_points, squared_max_radii, interactions, flip_normal_orientation)
        end_time = time.perf_counter()
        print(f"{n_queries} closest silhouette point queries took {end_time - start_time} seconds")

        return interactions

    else:
        start_time = time.perf_counter()
        interactions = [None] * n_queries

        if dim == 2:
            for q in range(n_queries):
                interactions[q] = fcpw.interaction_2D()
                found = scene.find_closest_silhouette_point(query_points[q], interactions[q],
                                                            flip_normal_orientation[q], squared_max_radii[q])

        elif dim == 3:
            for q in range(n_queries):
                interactions[q] = fcpw.interaction_3D()
                found = scene.find_closest_silhouette_point(query_points[q], interactions[q],
                                                            flip_normal_orientation[q], squared_max_radii[q])

        end_time = time.perf_counter()
        print(f"{n_queries} closest silhouette point queries took {end_time - start_time} seconds")

        return interactions

@wp.kernel
def run_warp_ray_intersection_queries(mesh: wp.uint64,
                                      query_origins: wp.array(dtype=wp.vec3),
                                      query_directions: wp.array(dtype=wp.vec3),
                                      query_t_max: wp.array(dtype=float),
                                      query_faces: wp.array(dtype=int),
                                      query_hit_points: wp.array(dtype=wp.vec3),
                                      query_dist: wp.array(dtype=float)):
    tid = wp.tid()
    o = query_origins[tid]
    d = query_directions[tid]
    t_max = query_t_max[tid]
    query = wp.mesh_query_ray(mesh, o, d, t_max)

    if query.result:
        hp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        query_faces[tid] = query.face
        query_hit_points[tid] = hp
        query_dist[tid] = query.t

@wp.kernel
def run_warp_closest_point_queries(mesh: wp.uint64,
                                   query_points: wp.array(dtype=wp.vec3),
                                   query_d_max: wp.array(dtype=float),
                                   query_faces: wp.array(dtype=int),
                                   query_closest_points: wp.array(dtype=wp.vec3),
                                   query_dist: wp.array(dtype=float)):
    tid = wp.tid()
    p = query_points[tid]
    d_max = query_d_max[tid]
    query = wp.mesh_query_point_no_sign(mesh, p, d_max)

    if query.result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        query_faces[tid] = query.face
        query_closest_points[tid] = cp
        query_dist[tid] = wp.length(cp - p)

def compare_cpu_interactions(cpu_interactions_baseline, cpu_interactions, dim):
    n_queries = len(cpu_interactions)
    for i in range(n_queries):
        cpu_interaction_baseline = cpu_interactions_baseline[i]
        cpu_interaction = cpu_interactions[i]
        different_indices = (cpu_interaction_baseline.primitive_index == -1 and cpu_interaction.primitive_index != -1) or \
                            (cpu_interaction_baseline.primitive_index != -1 and cpu_interaction.primitive_index == -1)

        if different_indices:
            print(f"#{i}/{n_queries}")
            if dim == 2:
                print("CPU Interaction Baseline")
                print(f"\tp: {cpu_interaction_baseline.p[0]} {cpu_interaction_baseline.p[1]}")
                print(f"\tn: {cpu_interaction_baseline.n[0]} {cpu_interaction_baseline.n[1]}")
                print(f"\tuv: {cpu_interaction_baseline.uv[0]}")
                print(f"\td: {cpu_interaction_baseline.d}")
                print(f"\tindex: {cpu_interaction_baseline.primitive_index}")
                print("CPU Interaction")
                print(f"\tp: {cpu_interaction.p[0]} {cpu_interaction.p[1]}")
                print(f"\tn: {cpu_interaction.n[0]} {cpu_interaction.n[1]}")
                print(f"\tuv: {cpu_interaction.uv[0]}")
                print(f"\td: {cpu_interaction.d}")
                print(f"\tindex: {cpu_interaction.primitive_index}")

            elif dim == 3:
                print("CPU Interaction Baseline")
                print(f"\tp: {cpu_interaction_baseline.p[0]} {cpu_interaction_baseline.p[1]} {cpu_interaction_baseline.p[2]}")
                print(f"\tn: {cpu_interaction_baseline.n[0]} {cpu_interaction_baseline.n[1]} {cpu_interaction_baseline.n[2]}")
                print(f"\tuv: {cpu_interaction_baseline.uv[0]} {cpu_interaction_baseline.uv[1]}")
                print(f"\td: {cpu_interaction_baseline.d}")
                print(f"\tindex: {cpu_interaction_baseline.primitive_index}")
                print("CPU Interaction")
                print(f"\tp: {cpu_interaction.p[0]} {cpu_interaction.p[1]} {cpu_interaction.p[2]}")
                print(f"\tn: {cpu_interaction.n[0]} {cpu_interaction.n[1]} {cpu_interaction.n[2]}")
                print(f"\tuv: {cpu_interaction.uv[0]} {cpu_interaction.uv[1]}")
                print(f"\td: {cpu_interaction.d}")
                print(f"\tindex: {cpu_interaction.primitive_index}")

def compare_cpu_gpu_interactions(cpu_interactions, gpu_interactions, dim):
    n_queries = len(cpu_interactions)
    for i in range(n_queries):
        cpu_interaction = cpu_interactions[i]
        gpu_interaction = gpu_interactions[i]
        different_indices = (cpu_interaction.primitive_index == -1 and gpu_interaction.index != 4294967295) or \
                            (cpu_interaction.primitive_index != -1 and gpu_interaction.index == 4294967295)

        if different_indices:
            print(f"#{i}/{n_queries}")
            if dim == 2:
                print("CPU Interaction")
                print(f"\tp: {cpu_interaction.p[0]} {cpu_interaction.p[1]}")
                print(f"\tn: {cpu_interaction.n[0]} {cpu_interaction.n[1]}")
                print(f"\tuv: {cpu_interaction.uv[0]}")
                print(f"\td: {cpu_interaction.d}")
                print(f"\tindex: {cpu_interaction.primitive_index}")
                print("GPU Interaction")
                print(f"\tp: {gpu_interaction.p.x} {gpu_interaction.p.y}")
                print(f"\tn: {gpu_interaction.n.x} {gpu_interaction.n.y}")
                print(f"\tuv: {gpu_interaction.uv.x}")
                print(f"\td: {gpu_interaction.d}")
                print(f"\tindex: {gpu_interaction.index}")

            elif dim == 3:
                print("CPU Interaction")
                print(f"\tp: {cpu_interaction.p[0]} {cpu_interaction.p[1]} {cpu_interaction.p[2]}")
                print(f"\tn: {cpu_interaction.n[0]} {cpu_interaction.n[1]} {cpu_interaction.n[2]}")
                print(f"\tuv: {cpu_interaction.uv[0]} {cpu_interaction.uv[1]}")
                print(f"\td: {cpu_interaction.d}")
                print(f"\tindex: {cpu_interaction.primitive_index}")
                print("GPU Interaction")
                print(f"\tp: {gpu_interaction.p.x} {gpu_interaction.p.y} {gpu_interaction.p.z}")
                print(f"\tn: {gpu_interaction.n.x} {gpu_interaction.n.y} {gpu_interaction.n.z}")
                print(f"\tuv: {gpu_interaction.uv.x} {gpu_interaction.uv.y}")
                print(f"\td: {gpu_interaction.d}")
                print(f"\tindex: {gpu_interaction.index}")

def compare_warp_and_gpu_interactions(warp_faces, warp_points, warp_dist, gpu_interactions):
    n_queries = len(gpu_interactions)
    for i in range(n_queries):
        gpu_interaction = gpu_interactions[i]
        different_indices = (warp_faces[i] == -1 and gpu_interaction.index != 4294967295) or \
                            (warp_faces[i] != -1 and gpu_interaction.index == 4294967295)

        if different_indices:
            print(f"#{i}/{n_queries}")
            print("Warp Interaction")
            print(f"\tp: {warp_points[i][0]} {warp_points[i][1]} {warp_points[i][2]}")
            print(f"\td: {warp_dist[i]}")
            print(f"\tindex: {warp_faces[i]}")
            print("GPU Interaction")
            print(f"\tp: {gpu_interaction.p.x} {gpu_interaction.p.y} {gpu_interaction.p.z}")
            print(f"\td: {gpu_interaction.d}")
            print(f"\tindex: {gpu_interaction.index}")

def test_watertight_intersection(positions, indices, scene, n_queries):
    """Test watertight ray-triangle intersection (3D only).
    
    This test verifies that the watertight intersection binding works properly
    by shooting rays from the mesh centroid toward random edge points.
    The watertight method should achieve better hit rates than the default method.
    """
    print("\n=== Testing Watertight Ray Intersection ===")
    
    # Compute centroid
    centroid = np.mean(positions, axis=0)
    print(f"Mesh centroid: {centroid}")
    
    # Generate rays toward edge points
    ray_origins, ray_directions = generate_edge_rays(n_queries, positions, indices, centroid)
    ray_bounds = np.inf * np.ones(n_queries, dtype=np.float32)
    
    # Run default intersection
    default_interactions = fcpw.interaction_3D_list()
    scene.intersect(ray_origins, ray_directions, ray_bounds, default_interactions, False, False)
    
    # Run watertight intersection
    watertight_interactions = fcpw.interaction_3D_list()
    scene.intersect(ray_origins, ray_directions, ray_bounds, watertight_interactions, False, True)
    
    # Count hits
    default_hits = sum(1 for i in default_interactions if i.primitive_index != -1)
    watertight_hits = sum(1 for i in watertight_interactions if i.primitive_index != -1)
    
    print(f"Default method hits:    {default_hits}/{n_queries} ({100*default_hits/n_queries:.1f}%)")
    print(f"Watertight method hits: {watertight_hits}/{n_queries} ({100*watertight_hits/n_queries:.1f}%)")
    
    # Also test single-ray API
    ray = fcpw.ray_3D(ray_origins[0], ray_directions[0], ray_bounds[0])
    interaction = fcpw.interaction_3D()
    hit_default = scene.intersect(ray, interaction, False, False)
    
    ray = fcpw.ray_3D(ray_origins[0], ray_directions[0], ray_bounds[0])
    interaction_wt = fcpw.interaction_3D()
    hit_watertight = scene.intersect(ray, interaction_wt, False, True)
    
    print(f"Single-ray API test: default={hit_default}, watertight={hit_watertight}")
    print("=== Watertight test complete ===\n")
    
    return watertight_hits >= default_hits

def visualize_polyscope_scene(positions, indices, query_points, random_directions,
                              random_squared_radii, interior_points, dim):
    ps.init()
    ps.register_point_cloud("query points", query_points)
    if len(interior_points) > 0:
        ps.register_point_cloud("interior points", interior_points)

    if dim == 2:
        ps.register_curve_network("scene", positions, indices)

    elif dim == 3:
        ps.register_surface_mesh("scene", positions, indices)

    ps.get_point_cloud("query points").add_vector_quantity("random directions", random_directions)
    ps.get_point_cloud("query points").add_scalar_quantity("random squared radii", random_squared_radii)
    ps.get_point_cloud("query points").set_point_radius_quantity("random squared radii")

    ps.show()

def run(file_path, n_queries, compute_silhouettes, compare_with_cpu_baseline,
        run_gpu_queries, compare_with_warp, refit_gpu_scene, visualize_scene, dim,
        test_watertight=False):
    print("Loading OBJ")
    positions, indices = load_obj(file_path, dim)

    print("\nBuilding BVH on CPU")
    scene = load_fcpw_scene(positions, indices, fcpw.aggregate_type.bvh_overlap_surface_area,
                            compute_silhouettes, False, dim, True)

    # Run watertight test if requested (3D only)
    if test_watertight and dim == 3:
        test_watertight_intersection(positions, indices, scene, n_queries)

    print("\nGenerating query data")
    bounding_box = compute_bounding_box(positions)
    query_points, random_directions, random_squared_radii = generate_scattered_points_and_rays(n_queries, bounding_box, dim)

    query_bounds = np.inf * np.ones(n_queries, dtype=np.float32)
    rand_nums = np.random.rand(n_queries, dim)
    is_interior = tag_interior_points(scene, query_points)
    flip_normal_orientation = np.array([0 if not is_interior[q] else 1 for q in range(n_queries)], dtype=int)

    baseline_cpu_ray_interactions = None
    baseline_cpu_sphere_interactions = None
    baseline_cpu_cpq_interactions = None
    baseline_cpu_cspq_interactions = None
    if compare_with_cpu_baseline:
        baseline_scene = load_fcpw_scene(positions, indices, fcpw.aggregate_type.baseline,
                                         compute_silhouettes, False, dim, True)

        print("\nRunning Baseline CPU Queries")
        baseline_cpu_ray_interactions = run_cpu_ray_intersection_queries(
            baseline_scene, query_points, random_directions, query_bounds, dim)
        baseline_cpu_sphere_interactions = run_cpu_sphere_intersection_queries(
            baseline_scene, query_points, random_squared_radii, rand_nums, dim)
        baseline_cpu_cpq_interactions = run_cpu_closest_point_queries(
            baseline_scene, query_points, query_bounds, dim)
        if compute_silhouettes:
            baseline_cpu_cspq_interactions = run_cpu_closest_silhouette_point_queries(
                baseline_scene, query_points, query_bounds, flip_normal_orientation, dim)

    gpu_ray_interactions = None
    gpu_sphere_interactions = None
    gpu_cpq_interactions = None
    gpu_cspq_interactions = None
    if run_gpu_queries:
        print("\nTransferring CPU BVH to GPU")
        gpu_scene = init_gpu_data(n_queries, dim)
        gpu_scene.transfer_to_gpu(scene)

        if refit_gpu_scene:
            print("\nRefitting GPU BVH")
            gpu_scene.refit(scene)

        print("\nRunning BVH GPU Queries")
        gpu_ray_interactions = fcpw.gpu_interaction_list()
        gpu_scene.intersect(query_points, random_directions, query_bounds, gpu_ray_interactions)

        gpu_sphere_interactions = fcpw.gpu_interaction_list()
        gpu_scene.intersect(query_points, random_squared_radii, rand_nums, gpu_sphere_interactions)

        gpu_cpq_interactions = fcpw.gpu_interaction_list()
        gpu_scene.find_closest_points(query_points, query_bounds, gpu_cpq_interactions)

        if compute_silhouettes:
            gpu_cspq_interactions = fcpw.gpu_interaction_list()
            gpu_scene.find_closest_silhouette_points(query_points, query_bounds,
                                                     flip_normal_orientation,
                                                     gpu_cspq_interactions)

    if compare_with_warp and run_gpu_queries and dim == 3:
        wp.init()
        device_points = wp.array(data=query_points, dtype=wp.vec3, device="cuda:0")
        device_directions = wp.array(data=random_directions, dtype=wp.vec3, device="cuda:0")
        device_parametric_dist = wp.full(shape=n_queries, value=np.inf, dtype=float, device="cuda:0")
        wp_mesh = wp.Mesh(points=wp.array(data=positions, dtype=wp.vec3, device="cuda:0"),
                          indices=wp.array(data=indices.flatten(), dtype=int, device="cuda:0"),
                          velocities=None, bvh_constructor="sah")

        host_intersection_faces = wp.full(shape=n_queries, value=-1, dtype=int, device="cpu")
        host_intersection_hit_points = wp.zeros(shape=n_queries, dtype=wp.vec3, device="cpu")
        host_intersection_dist = wp.full(shape=n_queries, value=np.inf, dtype=float, device="cpu")
        device_intersection_faces = wp.full(shape=n_queries, value=-1, dtype=int, device="cuda:0")
        device_intersection_hit_points = wp.zeros(shape=n_queries, dtype=wp.vec3, device="cuda:0")
        device_intersection_dist = wp.full(shape=n_queries, value=np.inf, dtype=float, device="cuda:0")
        with wp.ScopedTimer("Warp ray intersection queries", cuda_filter=wp.TIMING_KERNEL):
            wp.launch(kernel=run_warp_ray_intersection_queries, dim=n_queries,
                    inputs=[wp_mesh.id, device_points, device_directions, device_parametric_dist,
                            device_intersection_faces, device_intersection_hit_points,
                            device_intersection_dist], device="cuda:0")

        wp.copy(host_intersection_faces, device_intersection_faces)
        wp.copy(host_intersection_hit_points, device_intersection_hit_points)
        wp.copy(host_intersection_dist, device_intersection_dist)
        wp.synchronize()

        host_closest_point_faces = wp.full(shape=n_queries, value=-1, dtype=int, device="cpu")
        host_closest_points = wp.zeros(shape=n_queries, dtype=wp.vec3, device="cpu")
        host_closest_point_dist = wp.full(shape=n_queries, value=np.inf, dtype=float, device="cpu")
        device_closest_point_faces = wp.full(shape=n_queries, value=-1, dtype=int, device="cuda:0")
        device_closest_points = wp.zeros(shape=n_queries, dtype=wp.vec3, device="cuda:0")
        device_closest_point_dist = wp.full(shape=n_queries, value=np.inf, dtype=float, device="cuda:0")
        with wp.ScopedTimer("Warp closest point queries", cuda_filter=wp.TIMING_KERNEL):
            wp.launch(kernel=run_warp_closest_point_queries, dim=n_queries,
                    inputs=[wp_mesh.id, device_points, device_parametric_dist,
                            device_closest_point_faces, device_closest_points,
                            device_closest_point_dist], device="cuda:0")

        wp.copy(host_closest_point_faces, device_closest_point_faces)
        wp.copy(host_closest_points, device_closest_points)
        wp.copy(host_closest_point_dist, device_closest_point_dist)
        wp.synchronize()

        print("\nComparing GPU & Warp ray intersection query results...")
        compare_warp_and_gpu_interactions(host_intersection_faces.numpy(),
                                          host_intersection_hit_points.numpy(),
                                          host_intersection_dist.numpy(),
                                          gpu_ray_interactions)
        print("\nComparing GPU & Warp closest point query results...")
        compare_warp_and_gpu_interactions(host_closest_point_faces.numpy(),
                                          host_closest_points.numpy(),
                                          host_closest_point_dist.numpy(),
                                          gpu_cpq_interactions)

    print("\nRunning BVH CPU Queries")
    cpu_sphere_interactions = run_cpu_sphere_intersection_queries(
        scene, query_points, random_squared_radii, rand_nums, dim)
    scene.build(fcpw.aggregate_type.bvh_overlap_surface_area, True, True)
    cpu_ray_interactions = run_cpu_ray_intersection_queries(
        scene, query_points, random_directions, query_bounds, dim)
    cpu_cpq_interactions = run_cpu_closest_point_queries(
        scene, query_points, query_bounds, dim)
    cpu_cspq_interactions = None
    if compute_silhouettes:
        cpu_cspq_interactions = run_cpu_closest_silhouette_point_queries(
            scene, query_points, query_bounds, flip_normal_orientation, dim)

    if compare_with_cpu_baseline:
        print("\nComparing CPU ray intersection query results...")
        compare_cpu_interactions(baseline_cpu_ray_interactions, cpu_ray_interactions, dim)
        print("\nComparing CPU closest point query results...")
        compare_cpu_interactions(baseline_cpu_cpq_interactions, cpu_cpq_interactions, dim)
        if compute_silhouettes:
            print("\nComparing CPU closest silhouette point query results...")
            compare_cpu_interactions(baseline_cpu_cspq_interactions, cpu_cspq_interactions, dim)

    if run_gpu_queries:
        print("\nComparing CPU & GPU ray intersection query results...")
        compare_cpu_gpu_interactions(cpu_ray_interactions, gpu_ray_interactions, dim)
        print("\nComparing CPU & GPU sphere intersection query results...")
        compare_cpu_gpu_interactions(cpu_sphere_interactions, gpu_sphere_interactions, dim)
        print("\nComparing CPU & GPU closest point query results...")
        compare_cpu_gpu_interactions(cpu_cpq_interactions, gpu_cpq_interactions, dim)
        if compute_silhouettes:
            print("\nComparing CPU & GPU closest silhouette point query results...")
            compare_cpu_gpu_interactions(cpu_cspq_interactions, gpu_cspq_interactions, dim)

    if visualize_scene:
        interior_points = isolate_interior_points(scene, query_points)
        visualize_polyscope_scene(positions, indices, query_points, random_directions,
                                  random_squared_radii, interior_points, dim)

def main():
    parser = argparse.ArgumentParser(description="fcpw tests")
    parser.add_argument("--file_path", type=str, required=True, help="path to the input OBJ file")
    parser.add_argument("--dim", type=int, required=True, help="scene dimensionality")
    parser.add_argument("--n_queries", type=int, required=True, help="number of queries")
    parser.add_argument("--compute_silhouettes", action="store_true", help="compute silhouettes")
    parser.add_argument("--compare_with_cpu_baseline", action="store_true", help="compare with CPU baseline")
    parser.add_argument("--run_gpu_queries", action="store_true", help="run GPU queries")
    parser.add_argument("--compare_with_warp", action="store_true", help="compare with warp")
    parser.add_argument("--refit_gpu_scene", action="store_true", help="refit GPU scene")
    parser.add_argument("--visualize_scene", action="store_true", help="visualize scene")
    parser.add_argument("--test_watertight", action="store_true", help="test watertight ray intersection (3D only)")
    args = parser.parse_args()

    run(args.file_path, args.n_queries, args.compute_silhouettes, args.compare_with_cpu_baseline,
        args.run_gpu_queries, args.compare_with_warp, args.refit_gpu_scene, args.visualize_scene, args.dim,
        args.test_watertight)

if __name__ == "__main__":
    main()