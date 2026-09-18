// IMU and pressure-depth sensor models (ch20/ch21):
//   a_m = a + b_a + n_a,  omega_m = omega + b_g + n_g   (bias random walk via drift, white noise, latency, dropout)
//   z_m = z + b + n                                    (bias, drift, noise, latency, failure)
// Payload layouts match conrad/adapters/unity/vocabulary.py (imu_v1, depth_v1). Spatial values on the wire are
// in the UNITY convention; the Python adapter converts them back with the same documented axis map.
// Lever-arm effects of the mount offset are neglected at V2.1 (documented assumption).
using System;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.VehicleDynamics;

namespace Conrad.UnityV2.SensorSimulation
{
    public sealed class ImuSensor : SimSensorBase
    {
        private Vec3d _accelBiasWalk = Vec3d.Zero, _gyroBiasWalk = Vec3d.Zero;
        private readonly double _gyroNoiseStd, _biasWalkSigma;
        private readonly bool _reportOrientation;
        private double _lastTs;

        public override string Modality => "IMU";

        public ImuSensor(SensorParameters p, ulong seed) : base(p, seed)
        {
            _gyroNoiseStd = p.Extra.TryGetValue("gyro_noise_std", out object g) ? J.Num(g, "gyro_noise_std") : p.NoiseStd;
            _biasWalkSigma = p.Extra.TryGetValue("bias_random_walk", out object w) ? J.Num(w, "bias_random_walk") : 0.0;
            _reportOrientation = p.Extra.TryGetValue("report_orientation", out object o) && o is bool b && b;
        }

        protected override SensorPacketData Measure(TruthSnapshot truth, double tS)
        {
            double dt = Math.Max(tS - _lastTs, 0);
            _lastTs = tS;
            if (_biasWalkSigma > 0 && dt > 0)
            {
                double s = _biasWalkSigma * Math.Sqrt(dt);
                _accelBiasWalk += new Vec3d(Rng.Gaussian(s), Rng.Gaussian(s), Rng.Gaussian(s));
                _gyroBiasWalk += new Vec3d(Rng.Gaussian(s), Rng.Gaussian(s), Rng.Gaussian(s));
            }
            Quatd mount = P.MountRotationBody; // always set: the loader refuses OPEN mounts
            Vec3d f = mount.InverseRotate(truth.SpecificForceBody) + _accelBiasWalk;
            Vec3d w = mount.InverseRotate(truth.AngularVelocityBody) + _gyroBiasWalk;
            f = new Vec3d(Corrupt(f.X, tS), Corrupt(f.Y, tS), Corrupt(f.Z, tS));
            double ng = _gyroNoiseStd * Fault.NoiseScale;
            w = new Vec3d(w.X + Rng.Gaussian(ng), w.Y + Rng.Gaussian(ng), w.Z + Rng.Gaussian(ng));
            Vec3d fu = ConradFrames.PointToUnity(f);
            Vec3d wu = ConradFrames.AxialToUnity(w);
            double[] q = { double.NaN, double.NaN, double.NaN, double.NaN };
            if (_reportOrientation)
            {
                Quatd qu = ConradFrames.QuatToUnity((truth.WorldFromBody * mount).Normalized);
                q = new[] { qu.W, qu.X, qu.Y, qu.Z };
            }
            return new SensorPacketData
            {
                Layout = "imu_v1", Encoding = PayloadEncoding.f64le, Units = "m/s^2,rad/s,quat_wxyz",
                Shape = new[] { 10 },
                Payload = Pack(new[] { fu.X, fu.Y, fu.Z, wu.X, wu.Y, wu.Z, q[0], q[1], q[2], q[3] }),
            };
        }

        public override void Reset()
        {
            base.Reset();
            _accelBiasWalk = Vec3d.Zero;
            _gyroBiasWalk = Vec3d.Zero;
            _lastTs = 0;
        }
    }

    public sealed class DepthSensor : SimSensorBase
    {
        private readonly Func<Vec3d, double> _depthBelowSurface;

        public override string Modality => "PRESSURE_DEPTH";

        public DepthSensor(SensorParameters p, ulong seed, Func<Vec3d, double> depthBelowSurface) : base(p, seed)
        {
            _depthBelowSurface = depthBelowSurface ?? throw new ArgumentNullException(nameof(depthBelowSurface));
        }

        protected override SensorPacketData Measure(TruthSnapshot truth, double tS)
        {
            Vec3d sensorWorld = truth.PositionWorld + truth.WorldFromBody.Rotate(P.MountPositionBody);
            double z = Corrupt(_depthBelowSurface(sensorWorld), tS);
            return new SensorPacketData
            {
                Layout = "depth_v1", Encoding = PayloadEncoding.f64le, Units = "m", Shape = new[] { 1 },
                Payload = Pack(new[] { z }),
            };
        }
    }
}
