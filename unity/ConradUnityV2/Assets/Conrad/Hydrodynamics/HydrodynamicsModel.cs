// Hydrodynamic and hydrostatic terms of  M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + tau_env  (ch20/ch21).
// All quantities in the Conrad BODY frame (+X forward, +Y left, +Z up), SI, double precision.
// Engineering-level model (validity L1 until identified): diagonal M_A, diagonal linear + quadratic damping,
// damping and added-mass Coriolis act on the velocity RELATIVE to the water (v_r = v - v_c).
using System;
using Conrad.UnityV2.Core;

namespace Conrad.UnityV2.Hydrodynamics
{
    public struct Wrench
    {
        public Vec3d Force;  // N, body frame
        public Vec3d Torque; // N*m, body frame about the body origin

        public Wrench(Vec3d f, Vec3d t) { Force = f; Torque = t; }
        public static Wrench operator +(Wrench a, Wrench b) => new Wrench(a.Force + b.Force, a.Torque + b.Torque);
        public static readonly Wrench Zero = new Wrench(Vec3d.Zero, Vec3d.Zero);
    }

    public sealed class HydrodynamicsModel
    {
        public const double StandardGravity = 9.80665;

        private readonly RobotParameters _p;
        public double WaterDensity { get; }

        public HydrodynamicsModel(RobotParameters parameters, double waterDensityKgM3)
        {
            if (!(waterDensityKgM3 > 0)) throw new ArgumentException("water density must come from the scenario");
            _p = parameters;
            WaterDensity = waterDensityKgM3;
        }

        public Vec3d AddedMassLinear => new Vec3d(_p.AddedMass[0], _p.AddedMass[1], _p.AddedMass[2]);
        public Vec3d AddedMassAngular => new Vec3d(_p.AddedMass[3], _p.AddedMass[4], _p.AddedMass[5]);

        /// <summary>-D(v_r) v_r with separate linear and quadratic coefficients per DOF.</summary>
        public Wrench Damping(Vec3d vRel, Vec3d wRel)
        {
            double[] d1 = _p.LinearDrag, d2 = _p.QuadraticDrag;
            var f = new Vec3d(
                -(d1[0] + d2[0] * Math.Abs(vRel.X)) * vRel.X,
                -(d1[1] + d2[1] * Math.Abs(vRel.Y)) * vRel.Y,
                -(d1[2] + d2[2] * Math.Abs(vRel.Z)) * vRel.Z);
            var t = new Vec3d(
                -(d1[3] + d2[3] * Math.Abs(wRel.X)) * wRel.X,
                -(d1[4] + d2[4] * Math.Abs(wRel.Y)) * wRel.Y,
                -(d1[5] + d2[5] * Math.Abs(wRel.Z)) * wRel.Z);
            return new Wrench(f, t);
        }

        /// <summary>-C_A(v_r) v_r for diagonal added mass (Fossen): p = M_A,lin v_r, h = M_A,ang w.</summary>
        public Wrench AddedMassCoriolis(Vec3d vRel, Vec3d w)
        {
            Vec3d p = Vec3d.Mul(AddedMassLinear, vRel);
            Vec3d h = Vec3d.Mul(AddedMassAngular, w);
            return new Wrench(-Vec3d.Cross(w, p), -(Vec3d.Cross(vRel, p) + Vec3d.Cross(w, h)));
        }

        /// <summary>
        /// Weight at the CoG and buoyancy at the CoB (restoring forces and moments, -g(eta)).
        /// <paramref name="submergedFraction"/> in [0, 1] scales buoyancy near the surface.
        /// </summary>
        public Wrench Restoring(Quatd worldFromBody, double submergedFraction)
        {
            Vec3d upBody = worldFromBody.InverseRotate(new Vec3d(0, 0, 1));
            double weight = _p.MassKg * StandardGravity;
            double buoyancy = WaterDensity * StandardGravity * _p.DisplacedVolumeM3 * Math.Max(0.0, Math.Min(1.0, submergedFraction));
            Vec3d fw = upBody * -weight;
            Vec3d fb = upBody * buoyancy;
            Vec3d torque = Vec3d.Cross(_p.CenterOfMassBody, fw) + Vec3d.Cross(_p.CenterOfBuoyancyBody, fb);
            return new Wrench(fw + fb, torque);
        }

        /// <summary>Smooth submergence proxy: vehicle treated as a vertical extent of dimensions.Z about its origin.</summary>
        public double SubmergedFraction(double depthBelowSurfaceM)
        {
            double h = Math.Max(_p.Dimensions.Z, 1e-3);
            return Math.Max(0.0, Math.Min(1.0, depthBelowSurfaceM / h + 0.5));
        }
    }
}
