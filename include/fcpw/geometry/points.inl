namespace fcpw {

template <size_t DIM>
inline Point<DIM>::Point()
{
    indices[0] = -1;
    soup = nullptr;
    pIndex = -1;
}

template <size_t DIM>
inline BoundingBox<DIM> Point<DIM>::boundingBox() const
{
    return BoundingBox<DIM>(soup->positions[indices[0]]);
}

template <size_t DIM>
inline Vector<DIM> Point<DIM>::centroid() const
{
    return soup->positions[indices[0]];
}

template <size_t DIM>
inline float Point<DIM>::surfaceArea() const
{
    return 1.f;
}

template <size_t DIM>
inline float Point<DIM>::signedVolume() const
{
    return 1.f;
}

template <size_t DIM>
inline Vector<DIM> Point<DIM>::normal(bool normalize) const
{
    Vector<DIM> n = Vector<DIM>::Zero();
    n[DIM-1] = 1.0f;
    
    return n;
}

template <size_t DIM>
inline Vector<DIM> Point<DIM>::normal(const Vector<DIM-1>& uv) const
{
    return normal(true);
}

template <size_t DIM>
inline Vector<DIM-1> Point<DIM>::barycentricCoordinates(const Vector<DIM>& p) const
{
    return Vector<DIM-1>::Zero();
}

template <size_t DIM>
inline float Point<DIM>::samplePoint(const Vector<DIM>& randNums,
                                     Vector<DIM-1>& uv, Vector<DIM>& p, Vector<DIM>& n) const
{
    p = soup->positions[indices[0]];
    n = this->normal(true);
    uv = Vector<DIM-1>::Zero();

    return 0.0f;
}

template <size_t DIM>
inline void Point<DIM>::split(int dim, float splitCoord, BoundingBox<DIM>& boxLeft,
                              BoundingBox<DIM>& boxRight) const
{
    const Vector<DIM>& p = soup->positions[indices[0]];
    
    if (p[dim] <= splitCoord)
        boxLeft.expandToInclude(p);
    else
        boxRight.expandToInclude(p);
}

template<size_t DIM>
inline bool Point<DIM>::findClosestPoint(const BoundingSphere<DIM>& s, Interaction<DIM>& i) const
{
    const Vector<DIM>& p = soup->positions[indices[0]];

    Vector<DIM> diff = s.c - p;
    float dist2 = diff.squaredNorm();
    
    i.p = p;
    
    if (dist2 <= s.r2) {
        i.d = std::sqrt(dist2);
        i.primitiveIndex = pIndex;

        return true;
    }

    return false;
}

} // namespace fcpw
