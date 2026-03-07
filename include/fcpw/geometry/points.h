#pragma once

#include <fcpw/geometry/polygon_soup.h>

namespace fcpw {

template <size_t DIM>
class Point: public GeometricPrimitive<DIM> {
public:
    
    static_assert(DIM != 0);
    
    // constructor
    Point();

    // returns bounding box
    BoundingBox<DIM> boundingBox() const;

    // returns centroid
    Vector<DIM> centroid() const;

    // returns surface area
    float surfaceArea() const;

    // returns signed volume
    float signedVolume() const;
    
    // returns normal
    Vector<DIM> normal(bool normalize=false) const;

    // returns the normalized normal based on the local parameterization
    Vector<DIM> normal(const Vector<DIM-1>& uv) const;

    // returns barycentric coordinates
    Vector<DIM-1> barycentricCoordinates(const Vector<DIM>& p) const;
    
    // samples a random point on the geometric primitive and returns sampling pdf
    float samplePoint(const Vector<DIM>& randNums, Vector<DIM-1>& uv, Vector<DIM>& p, Vector<DIM>& n) const;

    // splits the line segment along the provided coordinate and axis
    void split(int dim, float splitCoord, BoundingBox<DIM>& boxLeft, BoundingBox<DIM>& boxRight) const;
    
    // TODO add fuzzy tolerance for point intersection?
    
    // intersects with ray
    bool intersect(const Ray<DIM>& r, Interaction<DIM>& i, bool checkForOcclusion=false) const { return false; }

    // intersects with ray
    int intersect(const Ray<DIM>& r, std::vector<Interaction<DIM>>& is,
                  bool checkForOcclusion=false, bool recordAllHits=false) const { return false; }

    // intersects with sphere
    bool intersect(const BoundingSphere<DIM>& s, Interaction<DIM>& i,
                   bool recordSurfaceArea=false) const { return false; }

    // finds closest point to sphere center
    bool findClosestPoint(const BoundingSphere<DIM>& s, Interaction<DIM>& i) const;

    // get and set index
    int getIndex() const { return pIndex; }
    void setIndex(int index) { pIndex = index; }

    // members
    int indices[1];
    int pIndex;
    const PolygonSoup<DIM> *soup;
};

} // namespace fcpw

#include "points.inl"
