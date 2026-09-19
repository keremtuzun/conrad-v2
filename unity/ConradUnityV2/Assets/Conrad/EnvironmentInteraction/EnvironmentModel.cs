// Environment interaction (ch21 Currents, Environmental fields, Unity <-> Twin 2E).
// Unity CONSUMES the environment from the shared scenario; it does not evolve ecological/structural truth.
// All vectors in the Conrad WORLD frame (+Z up); the water surface is z = SurfaceZ.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;

namespace Conrad.UnityV2.EnvironmentInteraction
{
    public interface ICurrentField
    {
        /// <summary>Water velocity (m/s, Conrad WORLD) at position p (m, Conrad WORLD) and time t (s).</summary>
        Vec3d VelocityAt(Vec3d p, double tS);
    }

    public sealed class ConstantCurrent : ICurrentField
    {
        private readonly Vec3d _v;
        public ConstantCurrent(Vec3d v) { _v = v; }
        public Vec3d VelocityAt(Vec3d p, double tS) => _v;
    }

    /// <summary>Linear depth profile between surface and reference depth, plus optional spatial gradient.</summary>
    public sealed class DepthProfileCurrent : ICurrentField
    {
        private readonly Vec3d _surface, _atReference, _gradientPerM;
        private readonly double _referenceDepthM, _surfaceZ;

        public DepthProfileCurrent(Vec3d surface, Vec3d atReference, double referenceDepthM, double surfaceZ, Vec3d gradientPerM)
        {
            if (!(referenceDepthM > 0)) throw new ArgumentException("reference depth must be positive");
            _surface = surface; _atReference = atReference; _referenceDepthM = referenceDepthM;
            _surfaceZ = surfaceZ; _gradientPerM = gradientPerM;
        }

        public Vec3d VelocityAt(Vec3d p, double tS)
        {
            double s = Math.Max(0.0, Math.Min(1.0, (_surfaceZ - p.Z) / _referenceDepthM));
            Vec3d v = _surface * (1 - s) + _atReference * s;
            return v + new Vec3d(_gradientPerM.X * p.X, _gradientPerM.Y * p.Y, 0.0);
        }
    }

    /// <summary>Base field + sinusoidal time variation + seeded gust / turbulence approximation.</summary>
    public sealed class TimeVaryingCurrent : ICurrentField
    {
        private readonly ICurrentField _base;
        private readonly Vec3d _amplitude;
        private readonly double _periodS, _turbulenceSigma;
        private readonly SeededRandom _rng;
        private long _lastSampleStep = -1;
        private Vec3d _turbulence = Vec3d.Zero;
        private readonly double _turbulenceUpdateS;
        public Vec3d GustVelocity = Vec3d.Zero;
        public double GustUntilS = -1;

        public TimeVaryingCurrent(ICurrentField baseField, Vec3d amplitude, double periodS, double turbulenceSigma,
                                  double turbulenceUpdateS, SeededRandom rng)
        {
            _base = baseField; _amplitude = amplitude; _periodS = periodS;
            _turbulenceSigma = turbulenceSigma; _turbulenceUpdateS = Math.Max(turbulenceUpdateS, 1e-3); _rng = rng;
        }

        public Vec3d VelocityAt(Vec3d p, double tS)
        {
            Vec3d v = _base.VelocityAt(p, tS);
            if (_periodS > 0) v += _amplitude * Math.Sin(2 * Math.PI * tS / _periodS);
            long step = (long)Math.Floor(tS / _turbulenceUpdateS);
            if (step != _lastSampleStep && _turbulenceSigma > 0)
            {
                // Draws happen at deterministic simulation instants, so replays reproduce them exactly.
                _turbulence = new Vec3d(_rng.Gaussian(_turbulenceSigma), _rng.Gaussian(_turbulenceSigma), _rng.Gaussian(_turbulenceSigma));
                _lastSampleStep = step;
            }
            if (tS < GustUntilS) v += GustVelocity;
            return v + _turbulence;
        }
    }

    /// <summary>Scenario-declared current gusts: extra uniform velocity during [start_s, start_s + duration_s).</summary>
    public sealed class GustScheduleCurrent : ICurrentField
    {
        private readonly ICurrentField _base;
        private readonly List<(double Start, double End, Vec3d V)> _gusts;

        public GustScheduleCurrent(ICurrentField baseField, List<(double, double, Vec3d)> gusts)
        {
            _base = baseField ?? throw new ArgumentNullException(nameof(baseField));
            _gusts = gusts ?? throw new ArgumentNullException(nameof(gusts));
        }

        public Vec3d VelocityAt(Vec3d p, double tS)
        {
            Vec3d v = _base.VelocityAt(p, tS);
            foreach (var g in _gusts) if (tS >= g.Start && tS < g.End) v += g.V;
            return v;
        }
    }

    /// <summary>Scenario-supplied environment. Missing required values (e.g. water density) are errors.</summary>
    public sealed class EnvironmentModel
    {
        public double WaterDensityKgM3 { get; }
        public double SurfaceZ { get; }
        public double Turbidity { get; }          // 0..1 visibility proxy consumed by camera degradation
        public ICurrentField Current { get; }

        public EnvironmentModel(double waterDensityKgM3, double surfaceZ, double turbidity, ICurrentField current)
        {
            if (!(waterDensityKgM3 > 0)) throw new ArgumentException("scenario environment.water_density_kgm3 is required");
            WaterDensityKgM3 = waterDensityKgM3;
            SurfaceZ = surfaceZ;
            Turbidity = Math.Max(0, Math.Min(1, turbidity));
            Current = current ?? throw new ArgumentNullException(nameof(current));
        }

        public double DepthBelowSurface(Vec3d positionWorld) => SurfaceZ - positionWorld.Z;

        /// <summary>Parse scenario.environment (Conrad WORLD, SI). See docs/UNITY.md for the schema.</summary>
        public static EnvironmentModel FromScenario(Dictionary<string, object> env, ulong seed)
        {
            double rho = J.Num(env, "water_density_kgm3");
            double surfaceZ = J.Has(env, "surface_z_m") ? J.Num(env, "surface_z_m") : 0.0;
            double turbidity = J.Has(env, "turbidity") ? J.Num(env, "turbidity") : 0.0;
            ICurrentField field = new ConstantCurrent(Vec3d.Zero);
            if (J.Has(env, "current"))
            {
                var c = J.Obj(env, "current");
                string kind = J.Str(c, "kind");
                switch (kind)
                {
                    case "constant":
                        field = new ConstantCurrent(Vec(c, "velocity_mps"));
                        break;
                    case "depth_profile":
                        field = new DepthProfileCurrent(Vec(c, "surface_velocity_mps"), Vec(c, "reference_velocity_mps"),
                            J.Num(c, "reference_depth_m"), surfaceZ,
                            J.Has(c, "gradient_per_m") ? Vec(c, "gradient_per_m") : Vec3d.Zero);
                        break;
                    default:
                        throw new JsonException("unknown current kind '" + kind + "'");
                }
                if (J.Has(c, "time_variation"))
                {
                    var tv = J.Obj(c, "time_variation");
                    field = new TimeVaryingCurrent(field, Vec(tv, "amplitude_mps"), J.Num(tv, "period_s"),
                        J.Has(tv, "turbulence_sigma_mps") ? J.Num(tv, "turbulence_sigma_mps") : 0.0,
                        J.Has(tv, "turbulence_update_s") ? J.Num(tv, "turbulence_update_s") : 0.5,
                        new SeededRandom(seed, "current.turbulence"));
                }
                if (J.Has(c, "gusts"))
                {
                    var gusts = new List<(double, double, Vec3d)>();
                    foreach (object o in J.Arr(c, "gusts"))
                    {
                        var g = o as Dictionary<string, object> ?? throw new JsonException("current gust must be an object");
                        double start = J.Num(g, "start_s"), duration = J.Num(g, "duration_s");
                        if (!(start >= 0) || !(duration > 0)) throw new JsonException("current gust needs start_s >= 0 and duration_s > 0");
                        gusts.Add((start, start + duration, Vec(g, "velocity_mps")));
                    }
                    field = new GustScheduleCurrent(field, gusts);
                }
            }
            return new EnvironmentModel(rho, surfaceZ, turbidity, field);
        }

        private static Vec3d Vec(Dictionary<string, object> o, string key)
        {
            var a = J.Arr(o, key);
            if (a.Count != 3) throw new JsonException(key + " must have 3 components");
            return new Vec3d(J.Num(a[0], key), J.Num(a[1], key), J.Num(a[2], key));
        }
    }
}
