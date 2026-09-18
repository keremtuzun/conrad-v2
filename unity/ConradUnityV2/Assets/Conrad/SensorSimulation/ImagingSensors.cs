// RGB camera and imaging sonar (ch20/ch21). Statistically useful behaviour over cinematic visuals:
// camera = rendered image + attenuation/backscatter toward a water colour (turbidity), fouling (contrast loss),
// white noise; sonar = geometric ray casts -> range/intensity proxy with attenuation, incidence, noise,
// false returns and dropout. Simplified sonar is NOT physically exact sonar (stated in capabilities context).
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using Conrad.UnityV2.VehicleDynamics;
using UnityEngine;

namespace Conrad.UnityV2.SensorSimulation
{
    public sealed class CameraSensor : SimSensorBase
    {
        private readonly Camera _camera;
        private readonly RenderTexture _target;
        private readonly Texture2D _readback;
        private readonly int _width, _height;
        private readonly Func<double> _turbidity;
        private readonly double[] _waterRgb;

        public override string Modality => "RGB";

        public CameraSensor(SensorParameters p, ulong seed, Camera camera, Func<double> turbidity) : base(p, seed, 8)
        {
            _camera = camera != null ? camera : throw new ArgumentNullException(nameof(camera));
            _turbidity = turbidity ?? throw new ArgumentNullException(nameof(turbidity));
            _width = (int)J.Num(p.Extra, "width_px");
            _height = (int)J.Num(p.Extra, "height_px");
            if (_width <= 0 || _height <= 0) throw new ArgumentException("camera width_px/height_px must be positive");
            _waterRgb = new[] { 0.05, 0.25, 0.30 };
            if (p.Extra.TryGetValue("water_rgb", out object wc) && wc is List<object> l && l.Count == 3)
                _waterRgb = new[] { J.Num(l[0], "water_rgb"), J.Num(l[1], "water_rgb"), J.Num(l[2], "water_rgb") };
            _target = new RenderTexture(_width, _height, 24, RenderTextureFormat.ARGB32);
            _readback = new Texture2D(_width, _height, TextureFormat.RGB24, false);
            _camera.targetTexture = _target;
            _camera.enabled = false; // rendered on demand at the sensor rate, in step with simulation time
            if (p.Extra.TryGetValue("vertical_fov_deg", out object fov)) _camera.fieldOfView = (float)J.Num(fov, "vertical_fov_deg");
        }

        protected override SensorPacketData Measure(TruthSnapshot truth, double tS)
        {
            _camera.Render();
            RenderTexture previous = RenderTexture.active;
            RenderTexture.active = _target;
            _readback.ReadPixels(new Rect(0, 0, _width, _height), 0, 0, false);
            _readback.Apply(false);
            RenderTexture.active = previous;
            byte[] raw = _readback.GetRawTextureData(); // bottom row first
            var hwc = new byte[_width * _height * 3];
            double k = Math.Max(0, Math.Min(1, _turbidity())) * 0.8 + Fault.Fouling * 0.6;
            k = Math.Min(k, 0.95);
            double sigma = P.NoiseStd * Fault.NoiseScale;
            int row = _width * 3;
            for (int y = 0; y < _height; y++)
            {
                int src = (_height - 1 - y) * row, dst = y * row;
                for (int x = 0; x < row; x++)
                {
                    double v = raw[src + x] / 255.0;
                    v = (1 - k) * v + k * _waterRgb[x % 3] + P.Bias + Rng.Gaussian(sigma);
                    hwc[dst + x] = (byte)Math.Max(0, Math.Min(255, Math.Round(v * 255.0)));
                }
            }
            double fy = 0.5 * _height / Math.Tan(0.5 * _camera.fieldOfView * Math.PI / 180.0);
            return new SensorPacketData
            {
                Layout = "rgb8_hwc_v1", Encoding = PayloadEncoding.u8, Units = "dn",
                Shape = new[] { _height, _width, 3 }, Payload = hwc,
                Context = new Dictionary<string, object>
                {
                    ["fx"] = fy, ["fy"] = fy, ["cx"] = 0.5 * _width, ["cy"] = 0.5 * _height,
                    ["turbidity_applied"] = k, ["renderer"] = "unity_camera",
                },
            };
        }
    }

    public sealed class SonarSensor : SimSensorBase
    {
        private readonly Transform _mount;
        private readonly int _beams, _bins, _layerMask;
        private readonly double _rangeMax, _hFov, _attenuation, _falseReturnProb;

        public override string Modality => "SONAR";

        public SonarSensor(SensorParameters p, ulong seed, Transform mount) : base(p, seed, 16)
        {
            _mount = mount != null ? mount : throw new ArgumentNullException(nameof(mount));
            _beams = (int)J.Num(p.Extra, "beams");
            _bins = (int)J.Num(p.Extra, "bins");
            _rangeMax = J.Num(p.Extra, "range_max_m");
            _hFov = J.Num(p.Extra, "horizontal_fov_rad");
            _attenuation = p.Extra.TryGetValue("attenuation_per_m", out object a) ? J.Num(a, "attenuation_per_m") : 0.0;
            _falseReturnProb = p.Extra.TryGetValue("false_return_prob", out object f) ? J.Num(f, "false_return_prob") : 0.0;
            _layerMask = Physics.DefaultRaycastLayers;
            if (_beams <= 0 || _bins <= 0 || !(_rangeMax > 0) || !(_hFov > 0))
                throw new ArgumentException("sonar needs beams, bins, range_max_m and horizontal_fov_rad");
        }

        protected override SensorPacketData Measure(TruthSnapshot truth, double tS)
        {
            var img = new float[_beams * _bins];
            double binSize = _rangeMax / _bins;
            for (int b = 0; b < _beams; b++)
            {
                double bearing = _beams == 1 ? 0 : -0.5 * _hFov + _hFov * b / (_beams - 1);
                // Conrad bearing is positive to the LEFT (+Y); Unity yaw about +Y is positive to the RIGHT.
                Vector3 dir = _mount.rotation * Quaternion.AngleAxis((float)(-bearing * 180.0 / Math.PI), Vector3.up) * Vector3.forward;
                if (Physics.Raycast(_mount.position, dir, out RaycastHit hit, (float)_rangeMax, _layerMask, QueryTriggerInteraction.Ignore))
                {
                    double r = hit.distance;
                    double incidence = Math.Abs(Vector3.Dot(-dir, hit.normal));
                    double intensity = incidence * Math.Exp(-2.0 * _attenuation * r);
                    int bin = Math.Min(_bins - 1, (int)(r / binSize));
                    img[b * _bins + bin] += (float)intensity;
                }
                if (_falseReturnProb > 0 && Rng.NextDouble() < _falseReturnProb)
                    img[b * _bins + (int)(Rng.NextDouble() * _bins)] += (float)(0.5 * Rng.NextDouble());
            }
            double sigma = P.NoiseStd * Fault.NoiseScale;
            for (int i = 0; i < img.Length; i++)
                img[i] = (float)Math.Max(0.0, img[i] + P.Bias + Rng.Gaussian(sigma));
            return new SensorPacketData
            {
                Layout = "sonar_beam_bin_v1", Encoding = PayloadEncoding.f32le, Units = "intensity_proxy",
                Shape = new[] { _beams, _bins }, Payload = Pack(img),
                Context = new Dictionary<string, object>
                {
                    ["range_max_m"] = _rangeMax, ["horizontal_fov_rad"] = _hFov,
                    ["model"] = "geometric_raycast_v1_not_acoustic_physics",
                },
            };
        }
    }
}
