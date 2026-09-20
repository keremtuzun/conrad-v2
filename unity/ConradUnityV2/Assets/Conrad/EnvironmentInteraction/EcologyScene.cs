// Twin 2E -> Unity (ch21 "Unity <-> Twin 2E"; gate I6). Two scene-description extensions of CONFIGURE_SCENE:
//   optics_grid  {"kind":"optics_grid","id","origin_m":[3],"spacing_m":[3],"shape":[nx,ny,nz],
//                 "beam_attenuation_per_m":[nx*ny*nz] (index (i*ny + j)*nz + k, cell centre origin + (i,j,k)*spacing),
//                 "water_rgb":[3] (optional)}
//                Beam attenuation c (1/m) of the water, sampled by the Python side from the Twin 2E turbidity field with
//                Twin 2E's own optical constants. The RGB camera blends every pixel toward the water colour with the
//                transmission exp(-integral of c along the pixel ray). Trilinear inside the grid, clamped at its faces.
//                A later CONFIGURE_SCENE (replace=false) with an optics_grid replaces the current one (the field evolves).
//   fouling_cover (optional key on box/capsule primitives, 0..1) + fouling_rgb (optional [3]): the Twin 2E biofouling
//                cover on that structure, shown as a colour on its visual only. Colliders are unchanged: biofouling does
//                not change geometry in the Python kernel either.
// Geometry sensors (range imager, sonar) are NOT attenuated: the kernel applies Twin 2E only to the structural and
// ecological payload channels, and the Unity path must match it.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using UnityEngine;

namespace Conrad.UnityV2.EnvironmentInteraction
{
    public sealed class OpticsField
    {
        public const int MaxCells = 1 << 20;
        public static OpticsField Current { get; private set; }

        public string Id { get; }
        public Vec3d Origin { get; }
        public Vec3d Spacing { get; }
        public int Nx { get; }
        public int Ny { get; }
        public int Nz { get; }
        public double[] WaterRgb { get; }
        private readonly float[] _c;

        public OpticsField(string id, Vec3d origin, Vec3d spacing, int nx, int ny, int nz, float[] c, double[] waterRgb)
        {
            if (nx < 1 || ny < 1 || nz < 1 || (long)nx * ny * nz > MaxCells) throw new JsonException("optics_grid shape out of range");
            if (!(spacing.X > 0 && spacing.Y > 0 && spacing.Z > 0)) throw new JsonException("optics_grid spacing must be positive");
            if (c == null || c.Length != nx * ny * nz) throw new JsonException("optics_grid beam_attenuation_per_m has the wrong length");
            foreach (float v in c) if (!(v >= 0) || float.IsInfinity(v)) throw new JsonException("optics_grid attenuation must be finite and >= 0");
            Id = id; Origin = origin; Spacing = spacing; Nx = nx; Ny = ny; Nz = nz; _c = c;
            WaterRgb = waterRgb ?? new[] { 0.05, 0.25, 0.30 };
        }

        public static void Set(OpticsField field) => Current = field;

        private double Cell(int i, int j, int k) => _c[(i * Ny + j) * Nz + k];

        private static void Axis(double u, int n, out int i0, out int i1, out double w)
        {
            if (n == 1 || u <= 0) { i0 = 0; i1 = 0; w = 0; return; }
            if (u >= n - 1) { i0 = n - 1; i1 = n - 1; w = 0; return; }
            i0 = (int)Math.Floor(u); i1 = i0 + 1; w = u - i0;
        }

        /// <summary>Beam attenuation (1/m) at a Conrad WORLD point: trilinear on cell centres, clamped at the faces.</summary>
        public double Attenuation(Vec3d p)
        {
            Axis((p.X - Origin.X) / Spacing.X, Nx, out int i0, out int i1, out double wx);
            Axis((p.Y - Origin.Y) / Spacing.Y, Ny, out int j0, out int j1, out double wy);
            Axis((p.Z - Origin.Z) / Spacing.Z, Nz, out int k0, out int k1, out double wz);
            double c00 = Cell(i0, j0, k0) * (1 - wx) + Cell(i1, j0, k0) * wx;
            double c10 = Cell(i0, j1, k0) * (1 - wx) + Cell(i1, j1, k0) * wx;
            double c01 = Cell(i0, j0, k1) * (1 - wx) + Cell(i1, j0, k1) * wx;
            double c11 = Cell(i0, j1, k1) * (1 - wx) + Cell(i1, j1, k1) * wx;
            double c0 = c00 * (1 - wy) + c10 * wy, c1 = c01 * (1 - wy) + c11 * wy;
            return c0 * (1 - wz) + c1 * wz;
        }

        /// <summary>Optical depth along a Conrad WORLD ray (midpoint rule, step at most half the smallest spacing).</summary>
        public double OpticalDepth(Vec3d origin, Vec3d unitDir, double length)
        {
            if (!(length > 0)) return 0;
            double h = 0.5 * Math.Min(Spacing.X, Math.Min(Spacing.Y, Spacing.Z));
            int n = Math.Max(4, (int)Math.Ceiling(length / h));
            double ds = length / n, tau = 0;
            for (int s = 0; s < n; s++) tau += Attenuation(origin + unitDir * ((s + 0.5) * ds));
            return tau * ds;
        }
    }

    public static class EcologySceneBuilder
    {
        public static readonly double[] DefaultFoulingRgb = { 0.25, 0.45, 0.12 };
        public static readonly double[] DefaultStructureRgb = { 0.55, 0.55, 0.55 };

        private static double[] Rgb(Dictionary<string, object> o, string key, double[] fallback)
        {
            if (!J.Has(o, key)) return fallback;
            var a = J.Arr(o, key);
            if (a.Count != 3) throw new JsonException(key + " must have 3 components");
            var rgb = new double[3];
            for (int i = 0; i < 3; i++)
            {
                rgb[i] = J.Num(a[i], key);
                if (!(rgb[i] >= 0 && rgb[i] <= 1)) throw new JsonException(key + " components must be in [0, 1]");
            }
            return rgb;
        }

        private static Vec3d V3(Dictionary<string, object> o, string key)
        {
            var a = J.Arr(o, key);
            if (a.Count != 3) throw new JsonException(key + " must have 3 components");
            var v = new Vec3d(J.Num(a[0], key), J.Num(a[1], key), J.Num(a[2], key));
            if (!v.IsFinite) throw new JsonException(key + " must be finite");
            return v;
        }

        public static OpticsField ParseOptics(string id, Dictionary<string, object> p)
        {
            var shape = J.Arr(p, "shape");
            if (shape.Count != 3) throw new JsonException("optics_grid shape must be [nx, ny, nz]");
            int nx = (int)J.Num(shape[0], "shape"), ny = (int)J.Num(shape[1], "shape"), nz = (int)J.Num(shape[2], "shape");
            var values = J.Arr(p, "beam_attenuation_per_m");
            var c = new float[values.Count];
            for (int i = 0; i < c.Length; i++) c[i] = (float)J.Num(values[i], "beam_attenuation_per_m");
            return new OpticsField(id, V3(p, "origin_m"), V3(p, "spacing_m"), nx, ny, nz, c, Rgb(p, "water_rgb", null));
        }

        /// <summary>Colour of a structure visual from its optional fouling_cover (0..1). The collider is untouched.</summary>
        public static void ApplyFouling(GameObject go, Dictionary<string, object> p)
        {
            if (!J.Has(p, "fouling_cover")) return;
            double cover = J.Num(p, "fouling_cover");
            if (!(cover >= 0 && cover <= 1)) throw new JsonException("fouling_cover must be in [0, 1]");
            double[] f = Rgb(p, "fouling_rgb", DefaultFoulingRgb), s = DefaultStructureRgb;
            var renderer = go.GetComponent<Renderer>();
            if (renderer == null) return;
            renderer.material.color = new Color(
                (float)(s[0] + (f[0] - s[0]) * cover), (float)(s[1] + (f[1] - s[1]) * cover), (float)(s[2] + (f[2] - s[2]) * cover));
        }
    }
}
