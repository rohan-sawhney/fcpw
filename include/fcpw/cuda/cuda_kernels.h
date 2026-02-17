#pragma once

#include <cstdint>

namespace fcpw {

// Forward declarations for host-callable kernel launch wrappers.
// These are defined in cuda_kernels.cu and compiled by nvcc.

// launches ray intersection kernel on GPU
void launchRayIntersectionKernel(int bvhType,
                                 void* d_nodes, void* d_primitives, void* d_silhouettes,
                                 void* d_rays, void* d_interactions,
                                 uint32_t checkForOcclusion, uint32_t nQueries,
                                 void* stream);

// launches sphere intersection kernel on GPU
void launchSphereIntersectionKernel(int bvhType,
                                    void* d_nodes, void* d_primitives, void* d_silhouettes,
                                    void* d_boundingSpheres, void* d_randNums,
                                    void* d_interactions,
                                    uint32_t nQueries,
                                    void* stream);

// launches closest point kernel on GPU
void launchClosestPointKernel(int bvhType,
                              void* d_nodes, void* d_primitives, void* d_silhouettes,
                              void* d_boundingSpheres, void* d_interactions,
                              uint32_t recordNormals, uint32_t nQueries,
                              void* stream);

// launches closest silhouette point kernel on GPU
void launchClosestSilhouettePointKernel(int bvhType,
                                        void* d_nodes, void* d_primitives, void* d_silhouettes,
                                        void* d_boundingSpheres, void* d_flipNormalOrientation,
                                        void* d_interactions,
                                        float squaredMinRadius, float precision,
                                        uint32_t nQueries,
                                        void* stream);

// launches transformed ray intersection kernel on GPU
void launchTransformedRayIntersectionKernel(int bvhType,
                                            void* d_nodes, void* d_primitives, void* d_silhouettes,
                                            void* d_transform,
                                            void* d_rays, void* d_interactions,
                                            uint32_t checkForOcclusion, uint32_t nQueries,
                                            void* stream);

// launches transformed sphere intersection kernel on GPU
void launchTransformedSphereIntersectionKernel(int bvhType,
                                               void* d_nodes, void* d_primitives, void* d_silhouettes,
                                               void* d_transform,
                                               void* d_boundingSpheres, void* d_randNums,
                                               void* d_interactions,
                                               uint32_t nQueries,
                                               void* stream);

// launches transformed closest point kernel on GPU
void launchTransformedClosestPointKernel(int bvhType,
                                         void* d_nodes, void* d_primitives, void* d_silhouettes,
                                         void* d_transform,
                                         void* d_boundingSpheres, void* d_interactions,
                                         uint32_t recordNormals, uint32_t nQueries,
                                         void* stream);

// launches transformed closest silhouette point kernel on GPU
void launchTransformedClosestSilhouettePointKernel(int bvhType,
                                                    void* d_nodes, void* d_primitives, void* d_silhouettes,
                                                    void* d_transform,
                                                    void* d_boundingSpheres, void* d_flipNormalOrientation,
                                                    void* d_interactions,
                                                    float squaredMinRadius, float precision,
                                                    uint32_t nQueries,
                                                    void* stream);

// launches BVH refit kernel on GPU
void launchRefitKernel(int bvhType,
                       void* d_nodes, void* d_primitives, void* d_silhouettes,
                       uint32_t* d_nodeIndices,
                       uint32_t firstNodeOffset, uint32_t nodeCount,
                       void* stream);

} // namespace fcpw
