implementing fcpw;
__include geometry;
__include bvh_node;

public interface IBvhNodes
{
    associatedtype NodeType : IBvhNode;

    BoundingBox getBoundingBox(uint i);
    [mutating]
    void setBoundingBox(uint i, BoundingBox box);
    bool hasBoundingCone(uint i);
    BoundingCone getBoundingCone(uint i);
    [mutating]
    void setBoundingCone(uint i, BoundingCone cone);
    bool isLeaf(uint i);
    uint getRightChildOffset(uint i);
    uint getNumPrimitives(uint i);
    uint getPrimitiveOffset(uint i);
    uint getNumSilhouettes(uint i);
    uint getSilhouetteOffset(uint i);
    __subscript(uint i) -> NodeType { get; set; }
};

public struct BvhNodes : IBvhNodes
{
    public typealias NodeType = BvhNode;
    public RWStructuredBuffer<float> pxMin, pyMin, pzMin;
    public RWStructuredBuffer<float> pxMax, pyMax, pzMax;
    public StructuredBuffer<uint> nPrimitives;
    public StructuredBuffer<uint> offsets;

    public BoundingBox getBoundingBox(uint i)
    {
        float3 pMin = float3(pxMin[i], pyMin[i], pzMin[i]);
        float3 pMax = float3(pxMax[i], pyMax[i], pzMax[i]);

        return BoundingBox(pMin, pMax);
    }

    [mutating]
    public void setBoundingBox(uint i, BoundingBox box)
    {
        pxMin[i] = box.pMin[0];
        pyMin[i] = box.pMin[1];
        pzMin[i] = box.pMin[2];
        pxMax[i] = box.pMax[0];
        pyMax[i] = box.pMax[1];
        pzMax[i] = box.pMax[2];
    }

    public bool hasBoundingCone(uint i)
    {
        return false;
    }

    public BoundingCone getBoundingCone(uint i)
    {
        return BoundingCone();
    }

    [mutating]
    public void setBoundingCone(uint i, BoundingCone cone)
    {
        // do nothing
    }

    public bool isLeaf(uint i)
    {
        return nPrimitives[i] > 0;
    }

    public uint getRightChildOffset(uint i)
    {
        return offsets[i];
    }

    public uint getNumPrimitives(uint i)
    {
        return nPrimitives[i];
    }

    public uint getPrimitiveOffset(uint i)
    {
        return offsets[i];
    }

    public uint getNumSilhouettes(uint i)
    {
        return 0;
    }

    public uint getSilhouetteOffset(uint i)
    {
        return 0;
    }

    public __subscript(uint i) -> NodeType
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

        set(NodeType node) {
            BoundingBox box = node.getBoundingBox();
            pxMin[i] = box.pMin[0];
            pyMin[i] = box.pMin[1];
            pzMin[i] = box.pMin[2];
            pxMax[i] = box.pMax[0];
            pyMax[i] = box.pMax[1];
            pzMax[i] = box.pMax[2];
        }
    }
};

public struct SnchNodes : IBvhNodes
{
    public typealias NodeType = SnchNode;
    public RWStructuredBuffer<float> pxMin, pyMin, pzMin;
    public RWStructuredBuffer<float> pxMax, pyMax, pzMax;
    public RWStructuredBuffer<float> axesx, axesy, axesz;
    public RWStructuredBuffer<float> halfAngles;
    public RWStructuredBuffer<float> radii;
    public StructuredBuffer<uint> nPrimitives;
    public StructuredBuffer<uint> offsets;
    public StructuredBuffer<uint> nSilhouettes;
    public StructuredBuffer<uint> silhouetteOffsets;

    public BoundingBox getBoundingBox(uint i)
    {
        float3 pMin = float3(pxMin[i], pyMin[i], pzMin[i]);
        float3 pMax = float3(pxMax[i], pyMax[i], pzMax[i]);

        return BoundingBox(pMin, pMax);
    }

    [mutating]
    public void setBoundingBox(uint i, BoundingBox box)
    {
        pxMin[i] = box.pMin[0];
        pyMin[i] = box.pMin[1];
        pzMin[i] = box.pMin[2];
        pxMax[i] = box.pMax[0];
        pyMax[i] = box.pMax[1];
        pzMax[i] = box.pMax[2];
    }

    public bool hasBoundingCone(uint i)
    {
        return true;
    }

    public BoundingCone getBoundingCone(uint i)
    {
        float3 axis = float3(axesx[i], axesy[i], axesz[i]);

        return BoundingCone(axis, halfAngles[i], radii[i]);
    }

    [mutating]
    public void setBoundingCone(uint i, BoundingCone cone)
    {
        axesx[i] = cone.axis[0];
        axesy[i] = cone.axis[1];
        axesz[i] = cone.axis[2];
        halfAngles[i] = cone.halfAngle;
        radii[i] = cone.radius;
    }

    public bool isLeaf(uint i)
    {
        return nPrimitives[i] > 0;
    }

    public uint getRightChildOffset(uint i)
    {
        return offsets[i];
    }

    public uint getNumPrimitives(uint i)
    {
        return nPrimitives[i];
    }

    public uint getPrimitiveOffset(uint i)
    {
        return offsets[i];
    }

    public uint getNumSilhouettes(uint i)
    {
        return nSilhouettes[i];
    }

    public uint getSilhouetteOffset(uint i)
    {
        return silhouetteOffsets[i];
    }

    public __subscript(uint i) -> NodeType
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

        set(NodeType node) {
            BoundingBox box = node.getBoundingBox();
            pxMin[i] = box.pMin[0];
            pyMin[i] = box.pMin[1];
            pzMin[i] = box.pMin[2];
            pxMax[i] = box.pMax[0];
            pyMax[i] = box.pMax[1];
            pzMax[i] = box.pMax[2];

            BoundingCone cone = node.getBoundingCone();
            axesx[i] = cone.axis[0];
            axesy[i] = cone.axis[1];
            axesz[i] = cone.axis[2];
            halfAngles[i] = cone.halfAngle;
            radii[i] = cone.radius;
        }
    }
};

public interface IPrimitives
{
    associatedtype PrimitiveType : IPrimitive;

    __subscript(uint i) -> PrimitiveType { get; }
};

public struct LineSegments : IPrimitives
{
    public typealias PrimitiveType = LineSegment;
    public StructuredBuffer<float> pax, pay;
    public StructuredBuffer<float> pbx, pby;
    public StructuredBuffer<uint> indices;

    public __subscript(uint i) -> PrimitiveType
    {
        get {
            PrimitiveType primitive;
            primitive.pa = float3(pax[i], pay[i], 0.0);
            primitive.pb = float3(pbx[i], pby[i], 0.0);
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

    public __subscript(uint i) -> PrimitiveType
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

    __subscript(uint i) -> SilhouetteType { get; }
};

public struct NoSilhouettes : ISilhouettes
{
    public typealias SilhouetteType = NoSilhouette;

    public __subscript(uint i) -> SilhouetteType
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
    public StructuredBuffer<float> px, py;
    public StructuredBuffer<float> n0x, n0y;
    public StructuredBuffer<float> n1x, n1y;
    public StructuredBuffer<uint> indices;
    public StructuredBuffer<uint> hasOneAdjacentFace;

    public __subscript(uint i) -> SilhouetteType
    {
        get {
            SilhouetteType silhouette;
            silhouette.p = float3(px[i], py[i], 0.0);
            silhouette.n0 = float3(n0x[i], n0y[i], 0.0);
            silhouette.n1 = float3(n1x[i], n1y[i], 0.0);
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

    public __subscript(uint i) -> SilhouetteType
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

public struct Rays
{
    public StructuredBuffer<float> ox, oy, oz;
    public StructuredBuffer<float> dx, dy, dz;
    public StructuredBuffer<float> tMax;

    public __subscript(uint i) -> Ray
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

    public __subscript(uint i) -> BoundingSphere
    {
        get {
            float3 c = float3(cx[i], cy[i], cz[i]);

            return BoundingSphere(c, r2[i]);
        }
    }
};

public struct Float3List
{
    public StructuredBuffer<float> x, y, z;

    public __subscript(uint i) -> float3
    {
        get {
            return float3(x[i], y[i], z[i]);
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

    public __subscript(uint i) -> Interaction
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
