#include <fcpw/fcpw.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool runAggregateTest(fcpw::AggregateType aggregateType, bool enableVectorization)
{
    constexpr float nearbyHitDistance = 2.0e-6f;
    constexpr float lowerBound = 1.0e-6f;
    constexpr float upperBound = 1.0e-4f;
    constexpr float tolerance = 1.0e-9f;

    std::vector<fcpw::Vector3> positions{
        fcpw::Vector3(0.0f, -1.0f, -1.0f),
        fcpw::Vector3(0.0f,  1.0f, -1.0f),
        fcpw::Vector3(0.0f,  0.0f,  1.0f),
        fcpw::Vector3(nearbyHitDistance, -1.0f, -1.0f),
        fcpw::Vector3(nearbyHitDistance,  1.0f, -1.0f),
        fcpw::Vector3(nearbyHitDistance,  0.0f,  1.0f),
    };
    std::vector<fcpw::Vector3i> triangles{
        fcpw::Vector3i(0, 1, 2),
        fcpw::Vector3i(3, 4, 5),
    };

    fcpw::Scene<3> scene;
    scene.setObjectCount(1);
    scene.setObjectVertices(positions, 0);
    scene.setObjectTriangles(triangles, 0);
    scene.build(aggregateType, enableVectorization, false, true);
    fcpw::Aggregate<3> *aggregate = scene.getSceneData()->aggregate.get();

    const fcpw::Vector3 origin(0.0f, 0.0f, 0.0f);
    const fcpw::Vector3 direction(1.0f, 0.0f, 0.0f);

    fcpw::Interaction<3> legacyInteraction;
    fcpw::Ray<3> legacyRay(origin, direction, upperBound);
    const bool legacyHit = aggregate->intersect(
        legacyRay, legacyInteraction, false
    );

    fcpw::Interaction<3> boundedInteraction;
    fcpw::Ray<3> boundedRay(origin, direction, upperBound, lowerBound);
    const bool boundedHit = aggregate->intersect(
        boundedRay, boundedInteraction, false
    );

    fcpw::Interaction<3> rejectedInteraction;
    fcpw::Ray<3> rejectedRay(
        origin, direction, upperBound, 3.0e-6f
    );
    const bool rejectedHit = aggregate->intersect(
        rejectedRay, rejectedInteraction, false
    );

    fcpw::Interaction<3> defaultInteriorInteraction;
    fcpw::Ray<3> defaultInteriorRay(
        fcpw::Vector3(-1.0e-4f, 0.0f, 0.0f), direction, 2.0e-4f
    );
    const bool defaultInteriorHit = aggregate->intersect(
        defaultInteriorRay, defaultInteriorInteraction, false
    );
    fcpw::Interaction<3> explicitInteriorInteraction;
    fcpw::Ray<3> explicitInteriorRay(
        fcpw::Vector3(-1.0e-4f, 0.0f, 0.0f), direction, 2.0e-4f, 0.0f
    );
    const bool explicitInteriorHit = aggregate->intersect(
        explicitInteriorRay, explicitInteriorInteraction, false
    );

    return legacyHit
        && std::fabs(legacyInteraction.d) <= tolerance
        && boundedHit
        && std::fabs(boundedInteraction.d - nearbyHitDistance) <= tolerance
        && !rejectedHit
        && defaultInteriorHit
        && explicitInteriorHit
        && std::fabs(defaultInteriorInteraction.d
                     - explicitInteriorInteraction.d) <= tolerance
        && (defaultInteriorInteraction.p
            - explicitInteriorInteraction.p).norm() <= tolerance;
}

bool runLineSegmentTest()
{
    constexpr float nearbyHitDistance = 2.0e-6f;
    std::vector<fcpw::Vector2> positions{
        fcpw::Vector2(0.0f, -1.0f),
        fcpw::Vector2(0.0f,  1.0f),
        fcpw::Vector2(nearbyHitDistance, -1.0f),
        fcpw::Vector2(nearbyHitDistance,  1.0f),
    };
    std::vector<fcpw::Vector2i> lineSegments{
        fcpw::Vector2i(0, 1), fcpw::Vector2i(2, 3)
    };
    fcpw::Scene<2> scene;
    scene.setObjectCount(1);
    scene.setObjectVertices(positions, 0);
    scene.setObjectLineSegments(lineSegments, 0);
    scene.build(fcpw::AggregateType::Bvh_SurfaceArea, false, false, true);

    fcpw::Ray<2> ray(
        fcpw::Vector2::Zero(), fcpw::Vector2(1.0f, 0.0f),
        1.0e-4f, 1.0e-6f
    );
    fcpw::Interaction<2> interaction;
    const bool hit = scene.getSceneData()->aggregate->intersect(
        ray, interaction, false
    );
    return hit && std::fabs(interaction.d - nearbyHitDistance) <= 1.0e-9f;
}

bool runTransformTest()
{
    fcpw::Ray<3> ray(
        fcpw::Vector3::Zero(), fcpw::Vector3(1.0f, 0.0f, 0.0f),
        10.0f, 1.0f
    );
    fcpw::Transform<3> transform = fcpw::Transform<3>::Identity();
    transform.scale(2.0f);
    const fcpw::Ray<3> transformedRay = ray.transform(transform);
    return std::fabs(transformedRay.tMin - 2.0f) <= 1.0e-6f
        && std::fabs(transformedRay.tMax - 20.0f) <= 1.0e-6f;
}

} // namespace

int main()
{
    const bool baselinePass = runAggregateTest(
        fcpw::AggregateType::Baseline, false
    );
    const bool bvhPass = runAggregateTest(
        fcpw::AggregateType::Bvh_SurfaceArea, false
    );
    bool mbvhPass = true;
#ifdef FCPW_USE_ENOKI
    mbvhPass = runAggregateTest(fcpw::AggregateType::Bvh_SurfaceArea, true);
#endif
    const bool lineSegmentPass = runLineSegmentTest();
    const bool transformPass = runTransformTest();
    const bool pass = baselinePass && bvhPass && mbvhPass
                   && lineSegmentPass && transformPass;

    std::cout << "{\"baseline\":" << (baselinePass ? "true" : "false")
              << ",\"bvh\":" << (bvhPass ? "true" : "false")
              << ",\"mbvh\":" << (mbvhPass ? "true" : "false")
              << ",\"line_segment\":"
              << (lineSegmentPass ? "true" : "false")
              << ",\"transform\":" << (transformPass ? "true" : "false")
              << ",\"result\":\"" << (pass ? "PASS" : "FAIL")
              << "\"}\n";
    return pass ? 0 : 1;
}
