// Truth-endpoint geometry query for a simulated structural sensor. It returns
// collider line of sight only; no structural condition or defect value crosses
// the bridge. Continuous healthy coverage is certified separately by the
// kernel's conservative support bound.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using UnityEngine;

namespace Conrad.UnityV2.ExternalInterfaces
{
    public static class SurfaceVisibility
    {
        public const int MaxPoints = 4096;
        private const int VehicleLayer = 2;

        private static Vector3 Point(object value)
        {
            var a = (List<object>)value;
            if (a.Count != 3) throw new ArgumentException("point must have three coordinates");
            return ConradFrames.ToEngine(ConradFrames.PointToUnity(
                new Vec3d(J.Num(a[0], "x"), J.Num(a[1], "y"), J.Num(a[2], "z"))));
        }

        public static Dictionary<string, object> Run(Dictionary<string, object> body)
        {
            Vector3 origin = Point(J.Arr(body, "origin_m"));
            var points = J.Arr(body, "points_m");
            if (points.Count == 0 || points.Count > MaxPoints)
                throw new ArgumentException("SURFACE_VISIBILITY needs 1.." + MaxPoints + " points");
            double tolerance = J.Num(body["tolerance_m"], "tolerance_m");
            if (tolerance < 0.0 || tolerance > 0.5) throw new ArgumentException("invalid visibility tolerance");
            Physics.SyncTransforms();
            var visible = new List<object>();
            var firstHitDistance = new List<object>();
            foreach (object item in points)
            {
                Vector3 target = Point(item);
                Vector3 ray = target - origin;
                float distance = ray.magnitude;
                if (distance < 1e-6f)
                {
                    visible.Add(false);
                    firstHitDistance.Add(0.0);
                    continue;
                }
                bool hit = Physics.Raycast(origin, ray / distance, out RaycastHit info,
                    distance + (float)tolerance, ~(1 << VehicleLayer), QueryTriggerInteraction.Ignore);
                visible.Add(!hit || info.distance >= distance - (float)tolerance);
                firstHitDistance.Add(hit ? (double)info.distance : -1.0);
            }
            return new Dictionary<string, object>
            {
                ["visible"] = visible,
                ["first_hit_distance_m"] = firstHitDistance,
            };
        }
    }
}
