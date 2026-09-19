// FRAME_PROBE: frame-contract diagnostic (ch28 contract tests "unit-axis probes match the documented Unity adapter",
// "round-trip pose preserves position and orientation"). Conrad WORLD poses go through ConradFrames INTO the
// engine (a real Transform), and the engine's own answers (position, rotation, forward/right/up) come back,
// together with the pose converted back to Conrad. No world state is read or changed: this is not truth.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using UnityEngine;

namespace Conrad.UnityV2.ExternalInterfaces
{
    public static class FrameProbe
    {
        public const int MaxPoses = 256;
        private static Transform _probe;

        private static List<object> L(Vector3 v) => new List<object> { (double)v.x, (double)v.y, (double)v.z };
        private static List<object> L(Vec3d v) => new List<object> { v.X, v.Y, v.Z };
        private static List<object> L(Quatd q) => new List<object> { q.W, q.X, q.Y, q.Z };

        public static Dictionary<string, object> Run(Dictionary<string, object> body)
        {
            var poses = J.Arr(body, "poses");
            if (poses.Count == 0 || poses.Count > MaxPoses) throw new ArgumentException("FRAME_PROBE needs 1.." + MaxPoses + " poses");
            if (_probe == null)
            {
                var go = new GameObject("ConradFrameProbe") { hideFlags = HideFlags.HideAndDontSave };
                _probe = go.transform;
            }
            var results = new List<object>();
            foreach (object o in poses)
            {
                var pose = (Dictionary<string, object>)o;
                var p = J.Arr(pose, "position_m");
                var q = J.Arr(pose, "orientation_wxyz");
                if (p.Count != 3 || q.Count != 4) throw new ArgumentException("pose needs position_m[3] and orientation_wxyz[4]");
                var pc = new Vec3d(J.Num(p[0], "x"), J.Num(p[1], "y"), J.Num(p[2], "z"));
                var qc = new Quatd(J.Num(q[0], "w"), J.Num(q[1], "x"), J.Num(q[2], "y"), J.Num(q[3], "z")).Normalized;
                _probe.SetPositionAndRotation(ConradFrames.ToEngine(ConradFrames.PointToUnity(pc)),
                                              ConradFrames.ToEngine(ConradFrames.QuatToUnity(qc)));
                Quaternion r = _probe.rotation;
                Quatd back = ConradFrames.QuatToConrad(ConradFrames.FromEngine(r));
                if (back.W < 0) back = new Quatd(-back.W, -back.X, -back.Y, -back.Z);
                results.Add(new Dictionary<string, object>
                {
                    ["unity_position"] = L(_probe.position),
                    ["unity_rotation_wxyz"] = new List<object> { (double)r.w, (double)r.x, (double)r.y, (double)r.z },
                    ["unity_forward"] = L(_probe.forward),
                    ["unity_right"] = L(_probe.right),
                    ["unity_up"] = L(_probe.up),
                    ["conrad_position"] = L(ConradFrames.PointToConrad(ConradFrames.FromEngine(_probe.position))),
                    ["conrad_orientation_wxyz"] = L(back),
                });
            }
            return new Dictionary<string, object> { ["results"] = results };
        }
    }
}
