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
    public sealed class CameraSensor : SimSensorBase, IDisposable
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
            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
                throw new InvalidOperationException("camera sensor " + p.Name + " needs a graphics device; do not start the player with -nographics");
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

        /// <summary>Releases the GPU render target (called when the runtime rebuilds its sensors on reset).</summary>
        public void Dispose()
        {
            if (_camera != null && _camera.targetTexture == _target) _camera.targetTexture = null;
            if (_target != null) { _target.Release(); UnityEngine.Object.Destroy(_target); }
            if (_readback != null) UnityEngine.Object.Destroy(_readback);
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
                    ["graphics_device"] = SystemInfo.graphicsDeviceType.ToString(),
                },
            };
        }
    }

    /// <summary>
    /// Range imager (modality DEPTH_RANGE): geometric ray casts on a pinhole grid, range along each pixel ray.
    /// Sensor frame is the Conrad convention (+X boresight, +Y left, +Z up); image u grows right (-Y), v grows down
    /// (-Z), row-major, top row first: exactly the pinhole model of conrad.twins.twin2s.raycast / Model2S.
    /// NaN = no return (nothing inside range_max_m, below range_min_m, or a dropout). Not an optical depth camera.
    /// </summary>
    public sealed class RangeImageSensor : SimSensorBase
    {
        private readonly Transform _mount;
        private readonly int _width, _height, _layerMask;
        private readonly double _rangeMax, _rangeMin, _dropoutProb;
        private readonly Vector3[] _localDirs;

        public override string Modality => "DEPTH_RANGE";

        public RangeImageSensor(SensorParameters p, ulong seed, Transform mount) : base(p, seed, 16)
        {
            _mount = mount != null ? mount : throw new ArgumentNullException(nameof(mount));
            _width = (int)J.Num(p.Extra, "width_px");
            _height = (int)J.Num(p.Extra, "height_px");
            double hFovDeg = J.Num(p.Extra, "hfov_deg");
            _rangeMax = J.Num(p.Extra, "max_range_m");
            _rangeMin = p.Extra.TryGetValue("min_range_m", out object mn) ? J.Num(mn, "min_range_m") : 0.0;
            _dropoutProb = p.Extra.TryGetValue("dropout_prob", out object d) ? J.Num(d, "dropout_prob") : 0.0;
            if (_width <= 0 || _height <= 0 || !(_rangeMax > 0) || !(hFovDeg > 0 && hFovDeg < 180))
                throw new ArgumentException("range imager needs width_px, height_px, 0 < hfov_deg < 180 and max_range_m");
            _layerMask = Physics.DefaultRaycastLayers;
            double f = 0.5 * _width / Math.Tan(0.5 * hFovDeg * Math.PI / 180.0);
            _localDirs = new Vector3[_width * _height];
            for (int j = 0; j < _height; j++)
                for (int i = 0; i < _width; i++)
                {
                    double u = (i + 0.5 - 0.5 * _width) / f, v = (j + 0.5 - 0.5 * _height) / f;
                    // Conrad sensor direction (1, -u, -v) -> Unity local (-y, z, x) = (u, -v, 1).
                    _localDirs[j * _width + i] = new Vector3((float)u, (float)-v, 1f).normalized;
                }
        }

        protected override SensorPacketData Measure(TruthSnapshot truth, double tS)
        {
            var img = new float[_width * _height];
            double sigma = P.NoiseStd * Fault.NoiseScale;
            Vector3 origin = _mount.position;
            Quaternion rot = _mount.rotation;
            for (int k = 0; k < img.Length; k++)
            {
                double r = double.NaN;
                if (Physics.Raycast(origin, rot * _localDirs[k], out RaycastHit hit, (float)_rangeMax, _layerMask, QueryTriggerInteraction.Ignore))
                    r = hit.distance + P.Bias + Fault.ExtraBias + Rng.Gaussian(sigma);
                if (_dropoutProb > 0 && Rng.NextDouble() < _dropoutProb) r = double.NaN;
                if (!(r >= _rangeMin && r <= _rangeMax)) r = double.NaN;
                img[k] = (float)r;
            }
            return new SensorPacketData
            {
                Layout = "range_f32_hw_v1", Encoding = PayloadEncoding.f32le, Units = "m",
                Shape = new[] { _height, _width }, Payload = Pack(img),
                Context = new Dictionary<string, object>
                {
                    ["hfov_deg"] = J.Num(P.Extra, "hfov_deg"), ["max_range_m"] = _rangeMax, ["min_range_m"] = _rangeMin,
                    ["missing"] = "NaN", ["model"] = "geometric_raycast_pinhole_v1",
                },
            };
        }
    }

    public sealed class SonarSensor : SimSensorBase
    {
        private readonly Transform _mount;
        private readonly int _beams, _bins, _layerMask, _elevationRays;
        private readonly double _rangeMax, _hFov, _vFov, _attenuation, _falseReturnProb;

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
            // Optional vertical fan: elevation_rays rays per beam spread over vertical_fov_rad (default: one ray).
            _elevationRays = p.Extra.TryGetValue("elevation_rays", out object er) ? (int)J.Num(er, "elevation_rays") : 1;
            _vFov = p.Extra.TryGetValue("vertical_fov_rad", out object vf) ? J.Num(vf, "vertical_fov_rad") : 0.0;
            _layerMask = Physics.DefaultRaycastLayers;
            if (_beams <= 0 || _bins <= 0 || !(_rangeMax > 0) || !(_hFov > 0))
                throw new ArgumentException("sonar needs beams, bins, range_max_m and horizontal_fov_rad");
            if (_elevationRays < 1 || (_elevationRays > 1 && !(_vFov > 0 && _vFov < Math.PI)))
                throw new ArgumentException("sonar elevation_rays > 1 needs 0 < vertical_fov_rad < pi");
        }

        /// <summary>Unity-local direction of (bearing, elevation); Conrad sensor (cos e cos a, cos e sin a, sin e).</summary>
        private Vector3 LocalDir(double bearing, double elevation)
        {
            if (_elevationRays == 1)
                return Quaternion.AngleAxis((float)(-bearing * 180.0 / Math.PI), Vector3.up) * Vector3.forward;
            double ce = Math.Cos(elevation);
            return new Vector3((float)(-ce * Math.Sin(bearing)), (float)Math.Sin(elevation), (float)(ce * Math.Cos(bearing)));
        }

        protected override SensorPacketData Measure(TruthSnapshot truth, double tS)
        {
            var img = new float[_beams * _bins];
            double binSize = _rangeMax / _bins;
            for (int b = 0; b < _beams; b++)
            {
                double bearing = _beams == 1 ? 0 : -0.5 * _hFov + _hFov * b / (_beams - 1);
                // Conrad bearing is positive to the LEFT (+Y); Unity yaw about +Y is positive to the RIGHT.
                for (int e = 0; e < _elevationRays; e++)
                {
                    double elevation = _elevationRays == 1 ? 0 : -0.5 * _vFov + _vFov * e / (_elevationRays - 1);
                    Vector3 dir = _mount.rotation * LocalDir(bearing, elevation);
                    if (Physics.Raycast(_mount.position, dir, out RaycastHit hit, (float)_rangeMax, _layerMask, QueryTriggerInteraction.Ignore))
                    {
                        double r = hit.distance;
                        double incidence = Math.Abs(Vector3.Dot(-dir, hit.normal));
                        double intensity = incidence * Math.Exp(-2.0 * _attenuation * r);
                        int bin = Math.Min(_bins - 1, (int)(r / binSize));
                        img[b * _bins + bin] += (float)intensity;
                    }
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
                    ["vertical_fov_rad"] = _vFov, ["elevation_rays"] = (long)_elevationRays,
                    ["model"] = "geometric_raycast_v1_not_acoustic_physics",
                },
            };
        }
    }
}
