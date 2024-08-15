namespace fcpw {

inline Vector3 lerp(const Vector3& A, const Vector3& B, const float tau) 
{
    return A + ((B - A) * tau);
}

inline Vector3 Puv(const Vector3& Pa, const Vector3& Pb, const Vector3& Pc, const Vector3& Pd, const Vector2& t) 
{
    Vector3 PaW = Pa*(1-t[0])*(1-t[1]);
    Vector3 PbW = Pb*(1-t[0])*t[1];
    Vector3 PcW = Pc*t[0]*t[1];
    Vector3 PdW = Pd*t[0]*(1-t[1]);
    return PaW + PbW + PcW + PdW;
}

inline Quad::Quad()
{
    indices[0] = -1;
    indices[1] = -1;
    indices[2] = -1;
    indices[3] = -1;
    soup = nullptr;
    pIndex = -1;
}

inline BoundingBox<3> Quad::boundingBox() const
{
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];
    const Vector3& pd = soup->positions[indices[3]];

    BoundingBox<3> box(pa);
    box.expandToInclude(pb);
    box.expandToInclude(pc);
    box.expandToInclude(pd);

    return box;
}

inline Vector3 Quad::centroid() const
{
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];
    const Vector3& pd = soup->positions[indices[3]];

    return (pa + pb + pc + pd)/4.0f;
}

inline float Quad::surfaceArea() const
{
    return normal().norm();
}

// almost the volume of a tetrahedron?
inline float Quad::signedVolume() const
{
    // REWRITE
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];

    return pa.cross(pb).dot(pc)/6.0f;
}

// the sum of 4 triangles with a virtual point at the centroid.
inline Vector3 Quad::normal(bool normalize) const
{
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];
    const Vector3& pd = soup->positions[indices[3]];
    const Vector3& cen = centroid();

    Vector3 v1 = cen - pa;
    Vector3 v2 = cen - pb;
    Vector3 v3 = cen - pc;
    Vector3 v4 = cen - pd;

    Vector3 n1 = v1.cross(v2);
    Vector3 n2 = v2.cross(v3);
    Vector3 n3 = v3.cross(v4);
    Vector3 n4 = v4.cross(v1);
    Vector3 n = (n1 + n2 + n3 + n4);
    return normalize ? n.normalized() : n;
}

inline Vector3 Quad::normal(int vIndex, int eIndex) const
{
    if (soup->vNormals.size() > 0 && vIndex >= 0) {
        return soup->vNormals[indices[vIndex]];
    }

    if (soup->eNormals.size() > 0 && eIndex >= 0) {
        return soup->eNormals[soup->eIndices[4*pIndex + eIndex]];
    }

    return normal(true);
}

// NEVER CALLED?
inline Vector3 Quad::normal(const Vector2& uv) const
{
    // REWRITE
    return normal(true);
}

// NEVER CALLED?
inline Vector2 Quad::barycentricCoordinates(const Vector3& p) const
{
    // REWRITE
    return Vector2(-1.0f, -1.0f);
}

inline float Quad::samplePoint(const Vector3& randNums, Vector2& uv, Vector3& p, Vector3& n) const
{
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];
    const Vector3& pd = soup->positions[indices[3]];

    Vector3 n1 = (pb - pa).cross(pc - pa);
    Vector3 n2 = (pc - pa).cross(pd - pa);
    n = n1 + n2;
    float area = n.norm();
    uv = Vector2(randNums[0], randNums[1]);
    p = Puv(pa, pb, pc, pd, uv);
    n /= area;

    return 2.0f/area;
}

// NEVER CALLED?
inline Vector2 Quad::textureCoordinates(const Vector2& uv) const
{
    // REWRITE
    return Vector2(-1, -1);
}

// NEVER CALLED?
inline float Quad::angle(int vIndex) const
{
    //REWRITE
    return -1.0f;
}

// NEVER CALLED?
inline void Quad::split(int dim, float splitCoord, BoundingBox<3>& boxLeft,
                            BoundingBox<3>& boxRight) const
{
    // REWRITE
    return;
}

// given 2 lines, find the T value for the point on lne 1 closest to skew line 2
inline float skewClosestT(const Vector3& line1A, const Vector3& line1B, const Vector3& line2A, const Vector3& line2B) {
    Vector3 l1Vec = line1B-line1A;
    float l1Mag = l1Vec.norm();
    Vector3 l1Dir = l1Vec/l1Mag;
    Vector3 l2Dir = (line2B-line2A).normalized();
    if (l1Dir.dot(l2Dir) <= minFloat) return 0.0f; // parallell
    Vector3 N = l1Dir.cross(l2Dir).normalized();
    Vector3 N2 = l2Dir.cross(N).normalized();
    float T = (line2A-line1A).dot(N2) / (l1Dir.dot(N2) *l1Mag);
    return T;
}

inline float intersectQuad(const Vector3& Pa, const Vector3& Pb, const Vector3& Pc, const Vector3& Pd, const Ray<3>& r, Vector3& pt, Vector2& st) {
    st[0] = .5;
    st[1] = .5;
    Vector3 A, B;
    Vector3 rA = r.o;
    Vector3 rB = r.o + r.d;
    for (int i = 0; i < 3; ++i) {
        A = lerp(Pa, Pd, st[0]);
        B = lerp(Pb, Pc, st[0]);
        st[1] =  skewClosestT(A, B, rA, rB);
        A = lerp(Pa, Pb, st[1]);
        B = lerp(Pd, Pc, st[1]);
        st[0] =  skewClosestT(A, B, rA, rB);
    }
    if ((st[0] <= 0.0f) || (st[0] >= 1.0f)) {
        return -1.0f;
    }
    if ((st[1] <= 0.0f) || (st[1] >= 1.0f)) {
        return -1.0f;
    }
    pt = Puv(Pa, Pb, Pc, Pd, st);

    return (r.o - pt).norm();
}

// intersect ray
inline bool Quad::intersect(const Ray<3>& r, Interaction<3>& i, bool checkForOcclusion) const
{
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];
    const Vector3& pd = soup->positions[indices[3]];

    Vector3 hitloc;
    Vector2 st;
    float d = intersectQuad(pa, pb, pc, pd, r, hitloc, st);
    if (d > 0.0f && d <= r.tMax) {

       if (checkForOcclusion) return true;
        i.d = d;
        i.p = hitloc;
        i.n = normal().normalized();
        i.uv = st;
        i.primitiveIndex = pIndex;

        return true;
    }
    return false;
}

// intersect ray
inline int Quad::intersect(const Ray<3>& r, std::vector<Interaction<3>>& is,
                               bool checkForOcclusion, bool recordAllHits) const
{  
    is.clear();
    Interaction<3> i;
    bool hit = intersect(r, i, checkForOcclusion);

    if (hit) {
        if (checkForOcclusion) return 1;

        is.emplace_back(i);
        return 1;
    }

    return 0;
}

inline float lineClosestT(const Vector3& A, const Vector3& B, const Vector3& Q) 
{
    Vector3 line = (B-A);
    float mag = line.norm();
    Vector3 W = Q - A;
    float T = ((line/mag).dot(W))/mag;
    return std::max(0.0f, std::min(1.0f, T));
}

inline float findClosestPointQuad(const Vector3& Pa, const Vector3& Pb, const Vector3& Pc, const Vector3& Pd,
                                      const Vector3& x, Vector3& pt, Vector2& st)
{
    st[0] = .5;
    st[1] = .5;
    Vector3 A, B;
    for (int i = 0; i < 3; ++i) {
        A = lerp(Pa, Pd, st[0]);
        B = lerp(Pb, Pc, st[0]);
        st[1] = lineClosestT(A, B, x);
        A = lerp(Pa, Pb, st[1]);
        B = lerp(Pd, Pc, st[1]);
        st[0] = lineClosestT(A, B, x);
    }
    pt = Puv(Pa, Pb, Pc, Pd, st);
    return (x - pt).norm();
}

// intersect sphere
inline bool Quad::intersect(const BoundingSphere<3>& s, Interaction<3>& i,
                                bool recordSurfaceArea) const
{
    bool found = findClosestPoint(s, i);
    if (found) {
        i.d = recordSurfaceArea ? surfaceArea() : 1.0f;
        return true;
    }

    return false;
}

inline bool Quad::findClosestPoint(const BoundingSphere<3>& s, Interaction<3>& i) const
{
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];
    const Vector3& pd = soup->positions[indices[3]];

    float d = findClosestPointQuad(pa, pb, pc, pd, s.c, i.p, i.uv);

    if (d*d <= s.r2) {
        i.d = d;
        i.primitiveIndex = pIndex;

        return true;
    }

    return false;
}

} // namespace fcpw
