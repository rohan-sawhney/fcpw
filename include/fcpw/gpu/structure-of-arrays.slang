implementing fcpw;
__include geometry;
__include bvh_node;

public struct Rays
{
    public StructuredBuffer<float> ox, oy, oz;
    public StructuredBuffer<float> dx, dy, dz;
    public StructuredBuffer<float> tMax;

    public __subscript(int i) -> Ray
    {
        get {
            float3 o = float3(ox[i], oy[i], oz[i]);
            float3 d = float3(dx[i], dy[i], dz[i]);

            return Ray(o, d, tMax[i]);
        }
    }
};

public struct BoundingSpheres
{
    public StructuredBuffer<float> cx, cy, cz;
    public StructuredBuffer<float> r2;

    public __subscript(int i) -> BoundingSphere
    {
        get {
            float3 c = float3(cx[i], cy[i], cz[i]);

            return BoundingSphere(c, r2[i]);
        }
    }
};

public struct RandNums
{
    public StructuredBuffer<float> u0, u1, u2;

    public __subscript(int i) -> float3
    {
        get {
            return float3(u0[i], u1[i], u2[i]);
        }
    }
};

public struct Interactions
{
    public RWStructuredBuffer<float> px, py, pz;
    public RWStructuredBuffer<float> nx, ny, nz;
    public RWStructuredBuffer<float> uvx, uvy;
    public RWStructuredBuffer<float> d;
    public RWStructuredBuffer<uint> indices;

    public __subscript(int i) -> Interaction
    {
        get {
            Interaction interaction;
            interaction.p[0] = px[i];
            interaction.p[1] = py[i];
            interaction.p[2] = pz[i];
            interaction.n[0] = nx[i];
            interaction.n[1] = ny[i];
            interaction.n[2] = nz[i];
            interaction.uv[0] = uvx[i];
            interaction.uv[1] = uvy[i];
            interaction.d = d[i];
            interaction.index = indices[i];

            return interaction;
        }

        set(Interaction interaction) {
            px[i] = interaction.p[0];
            py[i] = interaction.p[1];
            pz[i] = interaction.p[2];
            nx[i] = interaction.n[0];
            ny[i] = interaction.n[1];
            nz[i] = interaction.n[2];
            uvx[i] = interaction.uv[0];
            uvy[i] = interaction.uv[1];
            d[i] = interaction.d;
            indices[i] = interaction.index;
        }
    }
};

public interface IBvhNodes
{
    associatedtype NodeType : IBvhNode;

    public __subscript(int i) -> NodeType { get; }
};

public struct BvhNodes : IBvhNodes
{
    public typealias NodeType = BvhNode;
    public StructuredBuffer<float> pxMin, pyMin, pzMin;
    public StructuredBuffer<float> pxMax, pyMax, pzMax;
    public StructuredBuffer<uint> nPrimitives;
    public StructuredBuffer<uint> offsets;

    public __subscript(int i) -> NodeType
    {
        get {
            float3 pMin = float3(pxMin[i], pyMin[i], pzMin[i]);
            float3 pMax = float3(pxMax[i], pyMax[i], pzMax[i]);

            NodeType node;
            node.box = BoundingBox(pMin, pMax);
            node.nPrimitives = nPrimitives[i];
            node.offset = offsets[i];

            return node;
        }
    }
};

public struct SnchNodes : IBvhNodes
{
    public typealias NodeType = SnchNode;
    public StructuredBuffer<float> pxMin, pyMin, pzMin;
    public StructuredBuffer<float> pxMax, pyMax, pzMax;
    public StructuredBuffer<float> axesx, axesy, axesz;
    public StructuredBuffer<float> halfAngles;
    public StructuredBuffer<float> radii;
    public StructuredBuffer<uint> nPrimitives;
    public StructuredBuffer<uint> offsets;
    public StructuredBuffer<uint> nSilhouettes;
    public StructuredBuffer<uint> silhouetteOffsets;

    public __subscript(int i) -> NodeType
    {
        get {
            float3 pMin = float3(pxMin[i], pyMin[i], pzMin[i]);
            float3 pMax = float3(pxMax[i], pyMax[i], pzMax[i]);
            float3 axis = float3(axesx[i], axesy[i], axesz[i]);

            NodeType node;
            node.box = BoundingBox(pMin, pMax);
            node.cone = BoundingCone(axis, halfAngles[i], radii[i]);
            node.nPrimitives = nPrimitives[i];
            node.offset = offsets[i];
            node.nSilhouettes = nSilhouettes[i];
            node.silhouetteOffset = silhouetteOffsets[i];

            return node;
        }
    }
};

public interface IPrimitives
{
    associatedtype PrimitiveType : IPrimitive;

    public __subscript(int i) -> PrimitiveType { get; }
};

public struct LineSegments : IPrimitives
{
    public typealias PrimitiveType = LineSegment;
    public StructuredBuffer<float> pax, pay, paz;
    public StructuredBuffer<float> pbx, pby, pbz;
    public StructuredBuffer<uint> indices;

    public __subscript(int i) -> PrimitiveType
    {
        get {
            PrimitiveType primitive;
            primitive.pa = float3(pax[i], pay[i], paz[i]);
            primitive.pb = float3(pbx[i], pby[i], pbz[i]);
            primitive.index = indices[i];

            return primitive;
        }
    }
};

public struct Triangles : IPrimitives
{
    public typealias PrimitiveType = Triangle;
    public StructuredBuffer<float> pax, pay, paz;
    public StructuredBuffer<float> pbx, pby, pbz;
    public StructuredBuffer<float> pcx, pcy, pcz;
    public StructuredBuffer<uint> indices;

    public __subscript(int i) -> PrimitiveType
    {
        get {
            PrimitiveType primitive;
            primitive.pa = float3(pax[i], pay[i], paz[i]);
            primitive.pb = float3(pbx[i], pby[i], pbz[i]);
            primitive.pc = float3(pcx[i], pcy[i], pcz[i]);
            primitive.index = indices[i];

            return primitive;
        }
    }
};

public interface ISilhouettes
{
    associatedtype SilhouetteType : ISilhouette;

    public __subscript(int i) -> SilhouetteType { get; }
};

public struct NoSilhouettes : ISilhouettes
{
    public typealias SilhouetteType = NoSilhouette;

    public __subscript(int i) -> SilhouetteType
    {
        get {
            SilhouetteType silhouette;
            silhouette.index = UINT_MAX;

            return silhouette;
        }
    }
};

public struct Vertices : ISilhouettes
{
    public typealias SilhouetteType = Vertex;
    public StructuredBuffer<float> px, py, pz;
    public StructuredBuffer<float> n0x, n0y, n0z;
    public StructuredBuffer<float> n1x, n1y, n1z;
    public StructuredBuffer<uint> indices;
    public StructuredBuffer<uint> hasOneAdjacentFace;

    public __subscript(int i) -> SilhouetteType
    {
        get {
            SilhouetteType silhouette;
            silhouette.p = float3(px[i], py[i], pz[i]);
            silhouette.n0 = float3(n0x[i], n0y[i], n0z[i]);
            silhouette.n1 = float3(n1x[i], n1y[i], n1z[i]);
            silhouette.index = indices[i];
            silhouette.hasOneAdjacentFace = hasOneAdjacentFace[i];

            return silhouette;
        }
    }
};

public struct Edges : ISilhouettes
{
    public typealias SilhouetteType = Edge;
    public StructuredBuffer<float> pax, pay, paz;
    public StructuredBuffer<float> pbx, pby, pbz;
    public StructuredBuffer<float> n0x, n0y, n0z;
    public StructuredBuffer<float> n1x, n1y, n1z;
    public StructuredBuffer<uint> indices;
    public StructuredBuffer<uint> hasOneAdjacentFace;

    public __subscript(int i) -> SilhouetteType
    {
        get {
            SilhouetteType silhouette;
            silhouette.pa = float3(pax[i], pay[i], paz[i]);
            silhouette.pb = float3(pbx[i], pby[i], pbz[i]);
            silhouette.n0 = float3(n0x[i], n0y[i], n0z[i]);
            silhouette.n1 = float3(n1x[i], n1y[i], n1z[i]);
            silhouette.index = indices[i];
            silhouette.hasOneAdjacentFace = hasOneAdjacentFace[i];

            return silhouette;
        }
    }
};
