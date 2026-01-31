implementing fcpw;

public struct Ray
{
    public float3 o;    // ray origin
    public float3 d;    // ray direction
    public float3 dInv; // 1 over ray direction (coordinate-wise)
    public float tMax;  // max ray distance

    // constructor
    public __init(float3 o_, float3 d_, float tMax_)
    {
        o = o_;
        d = d_;
        dInv = float3(1.0, 1.0, 1.0) / d_;
        tMax = tMax_;
    }
};

// Watertight ray data for conservative BVH traversal and triangle intersection
// Based on: Woop, Benthin, Wald. "Watertight Ray/Triangle Intersection" JCGT 2013
public struct WatertightRayData
{
    // Axis permutation indices (kz is the dominant axis)
    public int kx, ky, kz;

    // Shear constants for ray-triangle intersection
    public float Sx, Sy, Sz;

    // Conservative reciprocal direction for near/far plane calculations
    public float3 rdirNear, rdirFar;

    // Near/far axis indices for box intersection (0-2 for pMin components, 3-5 for pMax)
    public int nearX, nearY, nearZ;
    public int farX, farY, farZ;

    // Compute watertight data from ray
    public __init(Ray r)
    {
        // Calculate dimension where the ray direction is maximal
        float3 absD = abs(r.d);
        if (absD.x > absD.y)
        {
            kz = (absD.x > absD.z) ? 0 : 2;
        }
        else
        {
            kz = (absD.y > absD.z) ? 1 : 2;
        }

        kx = (kz + 1) % 3;
        ky = (kx + 1) % 3;

        // Swap kx and ky to preserve winding direction of triangles
        if (r.d[kz] < 0.0)
        {
            int tmp = kx;
            kx = ky;
            ky = tmp;
        }

        // Calculate shear constants
        Sx = r.d[kx] / r.d[kz];
        Sy = r.d[ky] / r.d[kz];
        Sz = 1.0 / r.d[kz];

        // Conservative rounding constants
        // p = 1 + 2^-23, m = 1 - 2^-23
        const float p = 1.0 + 1.1920929e-7;  // 1 + 2^-23
        const float m = 1.0 - 1.1920929e-7;  // 1 - 2^-23

        // Setup near/far plane indices
        // For a box stored as [pMin.x, pMin.y, pMin.z, pMax.x, pMax.y, pMax.z]
        nearX = kx;
        nearY = ky;
        nearZ = kz;
        farX = 3 + kx;
        farY = 3 + ky;
        farZ = 3 + kz;

        if (r.d[kx] < 0.0) { int tmp = nearX; nearX = farX; farX = tmp; }
        if (r.d[ky] < 0.0) { int tmp = nearY; nearY = farY; farY = tmp; }
        if (r.d[kz] < 0.0) { int tmp = nearZ; nearZ = farZ; farZ = tmp; }

        // Compute conservative reciprocal directions
        // Dn(Dn(x)) where Dn(a) = a > 0 ? a*m : a*p
        float3 invD = r.dInv;
        rdirNear.x = invD[kx] > 0.0 ? invD[kx] * m * m : invD[kx] * p * p;
        rdirNear.y = invD[ky] > 0.0 ? invD[ky] * m * m : invD[ky] * p * p;
        rdirNear.z = invD[kz] > 0.0 ? invD[kz] * m * m : invD[kz] * p * p;

        // Up(Up(x)) where Up(a) = a > 0 ? a*p : a*m
        rdirFar.x = invD[kx] > 0.0 ? invD[kx] * p * p : invD[kx] * m * m;
        rdirFar.y = invD[ky] > 0.0 ? invD[ky] * p * p : invD[ky] * m * m;
        rdirFar.z = invD[kz] > 0.0 ? invD[kz] * p * p : invD[kz] * m * m;
    }
};
