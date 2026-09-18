// Double-precision vector/quaternion types used by the dynamics (Unity's float Vector3 is only used at the
// engine boundary), the Conrad <-> Unity frame map, seeded randomness and the integer-ns simulation clock.
using System;
using UnityEngine;

namespace Conrad.UnityV2.Core
{
    public struct Vec3d
    {
        public double X, Y, Z;

        public Vec3d(double x, double y, double z) { X = x; Y = y; Z = z; }

        public static readonly Vec3d Zero = new Vec3d(0, 0, 0);

        public static Vec3d operator +(Vec3d a, Vec3d b) => new Vec3d(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        public static Vec3d operator -(Vec3d a, Vec3d b) => new Vec3d(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        public static Vec3d operator -(Vec3d a) => new Vec3d(-a.X, -a.Y, -a.Z);
        public static Vec3d operator *(Vec3d a, double s) => new Vec3d(a.X * s, a.Y * s, a.Z * s);
        public static Vec3d operator *(double s, Vec3d a) => a * s;
        public static Vec3d operator /(Vec3d a, double s) => new Vec3d(a.X / s, a.Y / s, a.Z / s);

        public static double Dot(Vec3d a, Vec3d b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

        public static Vec3d Cross(Vec3d a, Vec3d b) =>
            new Vec3d(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);

        /// <summary>Component-wise product (diagonal matrix times vector).</summary>
        public static Vec3d Mul(Vec3d a, Vec3d b) => new Vec3d(a.X * b.X, a.Y * b.Y, a.Z * b.Z);

        public double Norm => Math.Sqrt(X * X + Y * Y + Z * Z);

        public bool IsFinite => !(double.IsNaN(X) || double.IsNaN(Y) || double.IsNaN(Z) ||
                                  double.IsInfinity(X) || double.IsInfinity(Y) || double.IsInfinity(Z));

        public override string ToString() => "(" + X + ", " + Y + ", " + Z + ")";
    }

    /// <summary>Unit quaternion, (w, x, y, z) order as on the Conrad wire. Rotates body -> parent.</summary>
    public struct Quatd
    {
        public double W, X, Y, Z;

        public Quatd(double w, double x, double y, double z) { W = w; X = x; Y = y; Z = z; }

        public static readonly Quatd Identity = new Quatd(1, 0, 0, 0);

        public static Quatd operator *(Quatd a, Quatd b) => new Quatd(
            a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z,
            a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
            a.W * b.Y - a.X * b.Z + a.Y * b.W + a.Z * b.X,
            a.W * b.Z + a.X * b.Y - a.Y * b.X + a.Z * b.W);

        public Quatd Conjugate => new Quatd(W, -X, -Y, -Z);

        public Quatd Normalized
        {
            get
            {
                double n = Math.Sqrt(W * W + X * X + Y * Y + Z * Z);
                if (n < 1e-12) throw new InvalidOperationException("zero-norm quaternion");
                double s = W >= 0 ? 1.0 / n : -1.0 / n; // canonical sign: w >= 0
                return new Quatd(W * s, X * s, Y * s, Z * s);
            }
        }

        public Vec3d Rotate(Vec3d v)
        {
            var p = this * new Quatd(0, v.X, v.Y, v.Z) * Conjugate;
            return new Vec3d(p.X, p.Y, p.Z);
        }

        public Vec3d InverseRotate(Vec3d v) => Conjugate.Rotate(v);

        public static Quatd FromEuler(double roll, double pitch, double yaw)
        {
            double cr = Math.Cos(roll / 2), sr = Math.Sin(roll / 2);
            double cp = Math.Cos(pitch / 2), sp = Math.Sin(pitch / 2);
            double cy = Math.Cos(yaw / 2), sy = Math.Sin(yaw / 2);
            return new Quatd(cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                             cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy).Normalized;
        }
    }

    /// <summary>
    /// The single documented axis map between Conrad (right-handed, +X forward, +Y left, +Z up) and Unity
    /// (left-handed, +X right, +Y up, +Z forward). Identical to conrad/adapters/unity/frames.py:
    ///   point/vector: (x, y, z)_c -> (-y, z, x)_u
    ///   axial vector: (x, y, z)_c -> ( y, -z, -x)_u      (det = -1)
    ///   quaternion  : (w, x, y, z)_c -> (w, y, -z, -x)_u
    /// </summary>
    public static class ConradFrames
    {
        public const string WireConvention = "UNITY_LH_X_RIGHT_Y_UP_Z_FORWARD";
        public const string BodyConvention = "CONRAD_BODY_RH_X_FORWARD_Y_LEFT_Z_UP";

        public static Vec3d PointToUnity(Vec3d c) => new Vec3d(-c.Y, c.Z, c.X);
        public static Vec3d PointToConrad(Vec3d u) => new Vec3d(u.Z, -u.X, u.Y);
        public static Vec3d AxialToUnity(Vec3d c) => new Vec3d(c.Y, -c.Z, -c.X);
        public static Vec3d AxialToConrad(Vec3d u) => new Vec3d(-u.Z, u.X, -u.Y);
        public static Quatd QuatToUnity(Quatd c) => new Quatd(c.W, c.Y, -c.Z, -c.X);
        public static Quatd QuatToConrad(Quatd u) => new Quatd(u.W, -u.Z, u.X, -u.Y);

        public static Vector3 ToEngine(Vec3d unityConvention) =>
            new Vector3((float)unityConvention.X, (float)unityConvention.Y, (float)unityConvention.Z);

        public static Vec3d FromEngine(Vector3 v) => new Vec3d(v.x, v.y, v.z);

        public static Quaternion ToEngine(Quatd unityConvention) =>
            new Quaternion((float)unityConvention.X, (float)unityConvention.Y, (float)unityConvention.Z, (float)unityConvention.W);

        public static Quatd FromEngine(Quaternion q) => new Quatd(q.w, q.x, q.y, q.z);
    }

    /// <summary>SplitMix64-seeded xoshiro256** stream; one independent stream per stochastic source.</summary>
    public sealed class SeededRandom
    {
        private ulong _s0, _s1, _s2, _s3;
        private double? _spare;
        public string StreamName { get; }
        public ulong Seed { get; }

        public SeededRandom(ulong seed, string streamName)
        {
            Seed = seed;
            StreamName = streamName;
            ulong x = seed ^ Fnv1a(streamName);
            _s0 = SplitMix(ref x); _s1 = SplitMix(ref x); _s2 = SplitMix(ref x); _s3 = SplitMix(ref x);
        }

        private static ulong Fnv1a(string s)
        {
            ulong h = 14695981039346656037UL;
            foreach (char c in s) { h ^= c; h *= 1099511628211UL; }
            return h;
        }

        private static ulong SplitMix(ref ulong x)
        {
            ulong z = x += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        private static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));

        public ulong NextULong()
        {
            ulong result = Rotl(_s1 * 5, 7) * 9;
            ulong t = _s1 << 17;
            _s2 ^= _s0; _s3 ^= _s1; _s1 ^= _s2; _s0 ^= _s3; _s2 ^= t; _s3 = Rotl(_s3, 45);
            return result;
        }

        /// <summary>Uniform in [0, 1).</summary>
        public double NextDouble() => (NextULong() >> 11) * (1.0 / 9007199254740992.0);

        public double Gaussian(double sigma)
        {
            if (sigma <= 0) return 0.0;
            if (_spare.HasValue) { double s = _spare.Value; _spare = null; return s * sigma; }
            double u, v, r;
            do { u = 2 * NextDouble() - 1; v = 2 * NextDouble() - 1; r = u * u + v * v; } while (r >= 1 || r == 0);
            double f = Math.Sqrt(-2 * Math.Log(r) / r);
            _spare = v * f;
            return u * f * sigma;
        }
    }

    /// <summary>Integer-nanosecond simulation clock. Advanced only by the experiment runtime.</summary>
    public sealed class SimClock
    {
        public const long NsPerSecond = 1_000_000_000L;
        public string Domain { get; }
        public long NowNs { get; private set; }

        public SimClock(string domain) { Domain = domain; }

        public void Advance(long dtNs)
        {
            if (dtNs <= 0) throw new ArgumentOutOfRangeException(nameof(dtNs), "time only moves forward");
            NowNs += dtNs;
        }

        public void Reset() { NowNs = 0; }
        public double Seconds => NowNs / (double)NsPerSecond;
    }
}
