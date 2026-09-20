// Static scene geometry as colliders (ch21 V2.0 "simple world"; V2.3 Twin 2S geometry arrives through this API).
// CONFIGURE_SCENE bodies carry primitives in the Conrad WORLD frame (right-handed, +X forward, +Y left, +Z up, m);
// this builder converts them with the single documented axis map (ConradFrames) and creates static colliders
// (plus a plain visual so the camera can see them). Unity never evolves this geometry: it is scenario input.
//   box         {"kind":"box","id","center_m":[3],"size_m":[3],"orientation_wxyz":[4]}
//   capsule     {"kind":"capsule","id","p0_m":[3],"p1_m":[3],"radius_m"}
//   heightfield {"kind":"heightfield","id","origin_m":[3],"spacing_m":[2],"heights_m":[[nx][ny]]}
//               vertex (i, j) = origin + (i*dx, j*dy, heights[i][j])
//   optics_grid (Twin 2E water optics) and the optional fouling_cover key of box/capsule: see EcologyScene.cs.
using System;
using System.Collections.Generic;
using Conrad.UnityV2.Core;
using UnityEngine;

namespace Conrad.UnityV2.EnvironmentInteraction
{
    public static class SceneGeometryBuilder
    {
        public const string WorldFrame = "WORLD";
        public const int MaxPrimitives = 4096;
        public const int MaxHeightfieldVertices = 1 << 20;

        /// <summary>Rebuild (replace=true) or extend the geometry under <paramref name="root"/>. Returns the child count.</summary>
        public static int Apply(Transform root, Dictionary<string, object> body)
        {
            if (root == null) throw new ArgumentNullException(nameof(root));
            if (J.Str(body, "frame") != WorldFrame) throw new JsonException("scene geometry must be expressed in the Conrad WORLD frame");
            var primitives = J.Arr(body, "primitives");
            if (primitives.Count > MaxPrimitives) throw new JsonException("too many primitives (max " + MaxPrimitives + ")");
            // Validate and build into a detached root first so a bad primitive leaves the old world untouched.
            var staging = new GameObject("SceneGeometryStaging").transform;
            OpticsField optics = null;
            try
            {
                foreach (object o in primitives)
                {
                    var p = o as Dictionary<string, object> ?? throw new JsonException("primitive must be an object");
                    string kind = J.Str(p, "kind");
                    string id = J.Str(p, "id");
                    switch (kind)
                    {
                        case "box": Box(staging, id, p); break;
                        case "capsule": Capsule(staging, id, p); break;
                        case "heightfield": Heightfield(staging, id, p); break;
                        case "optics_grid":
                            if (optics != null) throw new JsonException("at most one optics_grid per request");
                            optics = EcologySceneBuilder.ParseOptics(id, p);
                            new GameObject("optics:" + id).transform.SetParent(staging, false); // counted, no collider
                            break;
                        default: throw new JsonException("unknown primitive kind '" + kind + "'");
                    }
                }
            }
            catch
            {
                UnityEngine.Object.Destroy(staging.gameObject);
                throw;
            }
            bool replace = J.Bool(body, "replace");
            for (int i = root.childCount - 1; i >= 0; i--)
            {
                var child = root.GetChild(i).gameObject;
                // replace: everything goes; an incremental optics update replaces only the previous optics grid
                if (!replace && !(optics != null && child.name.StartsWith("optics:", StringComparison.Ordinal))) continue;
                child.SetActive(false); // colliders leave the physics scene immediately
                UnityEngine.Object.Destroy(child);
            }
            if (optics != null) OpticsField.Set(optics);
            else if (replace) OpticsField.Set(null);
            while (staging.childCount > 0) staging.GetChild(0).SetParent(root, true);
            UnityEngine.Object.Destroy(staging.gameObject);
            Physics.SyncTransforms();
            int active = 0;
            for (int i = 0; i < root.childCount; i++) if (root.GetChild(i).gameObject.activeSelf) active++;
            return active;
        }

        private static Vec3d V3(Dictionary<string, object> o, string key)
        {
            var a = J.Arr(o, key);
            if (a.Count != 3) throw new JsonException(key + " must have 3 components");
            var v = new Vec3d(J.Num(a[0], key), J.Num(a[1], key), J.Num(a[2], key));
            if (!v.IsFinite) throw new JsonException(key + " must be finite");
            return v;
        }

        private static Quatd Q(Dictionary<string, object> o, string key)
        {
            if (!J.Has(o, key)) return Quatd.Identity;
            var a = J.Arr(o, key);
            if (a.Count != 4) throw new JsonException(key + " must have 4 components (w, x, y, z)");
            return new Quatd(J.Num(a[0], key), J.Num(a[1], key), J.Num(a[2], key), J.Num(a[3], key)).Normalized;
        }

        private static GameObject Visual(PrimitiveType type, Transform parent, string name)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.isStatic = false;
            go.transform.SetParent(parent, false);
            return go;
        }

        public static void Box(Transform parent, string id, Dictionary<string, object> p)
        {
            Vec3d size = V3(p, "size_m");
            if (!(size.X > 0 && size.Y > 0 && size.Z > 0)) throw new JsonException("box " + id + " size must be positive");
            var go = Visual(PrimitiveType.Cube, parent, "box:" + id);
            go.transform.SetPositionAndRotation(
                ConradFrames.ToEngine(ConradFrames.PointToUnity(V3(p, "center_m"))),
                ConradFrames.ToEngine(ConradFrames.QuatToUnity(Q(p, "orientation_wxyz"))));
            // Box-local Conrad extents (sx, sy, sz) are Unity-local (sy, sz, sx) under the same axis map.
            go.transform.localScale = new Vector3((float)size.Y, (float)size.Z, (float)size.X);
            EcologySceneBuilder.ApplyFouling(go, p);
        }

        public static void Capsule(Transform parent, string id, Dictionary<string, object> p)
        {
            Vector3 a = ConradFrames.ToEngine(ConradFrames.PointToUnity(V3(p, "p0_m")));
            Vector3 b = ConradFrames.ToEngine(ConradFrames.PointToUnity(V3(p, "p1_m")));
            double r = J.Num(p, "radius_m");
            float length = (b - a).magnitude;
            if (!(r > 0) || !(length > 0)) throw new JsonException("capsule " + id + " needs radius > 0 and distinct end points");
            var go = Visual(PrimitiveType.Capsule, parent, "capsule:" + id);
            go.transform.SetPositionAndRotation(0.5f * (a + b), Quaternion.FromToRotation(Vector3.up, (b - a) / length));
            // The unit capsule primitive: radius 0.5, height 2 along local Y. Scale x/z by diameter; y by half length.
            float d = (float)(2.0 * r);
            go.transform.localScale = new Vector3(d, 0.5f * (length + d), d);
            EcologySceneBuilder.ApplyFouling(go, p);
        }

        public static void Heightfield(Transform parent, string id, Dictionary<string, object> p)
        {
            Vec3d origin = V3(p, "origin_m");
            var spacing = J.Arr(p, "spacing_m");
            if (spacing.Count != 2) throw new JsonException("heightfield spacing_m must be [dx, dy]");
            double dx = J.Num(spacing[0], "dx"), dy = J.Num(spacing[1], "dy");
            if (!(dx > 0 && dy > 0)) throw new JsonException("heightfield spacing must be positive");
            var rows = J.Arr(p, "heights_m");
            int nx = rows.Count;
            int ny = nx > 0 ? ((List<object>)rows[0]).Count : 0;
            if (nx < 2 || ny < 2 || nx * ny > MaxHeightfieldVertices) throw new JsonException("heightfield " + id + " grid must be at least 2x2");
            var verts = new Vector3[nx * ny];
            for (int i = 0; i < nx; i++)
            {
                var row = rows[i] as List<object>;
                if (row == null || row.Count != ny) throw new JsonException("heightfield " + id + " rows must have equal length");
                for (int j = 0; j < ny; j++)
                {
                    var c = new Vec3d(origin.X + i * dx, origin.Y + j * dy, origin.Z + J.Num(row[j], "height"));
                    if (!c.IsFinite) throw new JsonException("heightfield " + id + " heights must be finite");
                    verts[i * ny + j] = ConradFrames.ToEngine(ConradFrames.PointToUnity(c));
                }
            }
            var tris = new int[(nx - 1) * (ny - 1) * 6];
            int k = 0;
            for (int i = 0; i < nx - 1; i++)
                for (int j = 0; j < ny - 1; j++)
                {
                    int v00 = i * ny + j, v10 = (i + 1) * ny + j, v01 = i * ny + j + 1, v11 = (i + 1) * ny + j + 1;
                    // Unity front face normal = cross(b - a, c - a); this winding makes it Conrad +Z (Unity +Y, up).
                    tris[k++] = v00; tris[k++] = v11; tris[k++] = v10;
                    tris[k++] = v00; tris[k++] = v01; tris[k++] = v11;
                }
            var mesh = new Mesh { name = "heightfield:" + id, indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            mesh.vertices = verts;
            mesh.triangles = tris;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            var go = new GameObject("heightfield:" + id);
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var shader = Shader.Find("Standard");
            if (shader != null) go.AddComponent<MeshRenderer>().sharedMaterial = new Material(shader);
            go.AddComponent<MeshCollider>().sharedMesh = mesh;
        }
    }
}
