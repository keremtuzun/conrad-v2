// Shared Scenario loading (ch21 Shared Scenario Loader) and deterministic replay logging (ch20 Deterministic replay).
// Unity reads the SAME ScenarioDefinition JSON the domain twins use (conrad.schemas.world.Scenario, dumped with
// model_dump(mode="json")). Unity reads only: seed, coordinate_system, environment, robots[0].initial_pose,
// mission and the world entities' geometry references. It never evolves twin state.
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Conrad.UnityV2.Core;

namespace Conrad.UnityV2.MissionExperimentRuntime
{
    public sealed class ScenarioDefinition
    {
        public string ScenarioId, ScenarioVersion, FileDigest;
        public long Seed;
        public Dictionary<string, object> Environment;
        public Vec3d InitialPositionWorld;
        public Quatd InitialOrientationWorld = Quatd.Identity;
        public Dictionary<string, object> Mission;
        public List<Dictionary<string, object>> WorldEntities = new List<Dictionary<string, object>>();

        public static string Sha256Hex(byte[] data)
        {
            using (var sha = SHA256.Create())
            {
                var sb = new StringBuilder(64);
                foreach (byte b in sha.ComputeHash(data)) sb.Append(b.ToString("x2"));
                return sb.ToString();
            }
        }

        public static ScenarioDefinition LoadFile(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var s = Parse(Encoding.UTF8.GetString(bytes));
            s.FileDigest = Sha256Hex(bytes);
            return s;
        }

        public static ScenarioDefinition Parse(string json)
        {
            var doc = MiniJson.ParseObject(json);
            var cs = J.Obj(doc, "coordinate_system");
            // The scenario declares its WORLD convention; Unity supports the documented simulation default only.
            if (J.Str(cs, "handedness") != "RIGHT" || J.Str(cs, "up_axis") != "+Z" || J.Str(cs, "forward_axis") != "+X"
                || J.Str(cs, "units") != "m")
                throw new JsonException("Unity V2 requires the RIGHT/+Z up/+X forward/m WORLD convention (ADR-0002)");
            var s = new ScenarioDefinition
            {
                ScenarioId = J.Str(doc, "scenario_id"),
                ScenarioVersion = J.Str(doc, "scenario_version"),
                Seed = J.Long(doc, "seed"),
                Environment = J.Obj(doc, "environment"),
                Mission = doc.TryGetValue("mission", out object m) ? m as Dictionary<string, object> : null,
            };
            foreach (object e in J.Arr(doc, "world_entities")) s.WorldEntities.Add((Dictionary<string, object>)e);
            var robots = J.Arr(doc, "robots");
            if (robots.Count == 0) throw new JsonException("scenario defines no robot");
            var pose = J.Obj((Dictionary<string, object>)robots[0], "initial_pose");
            if (J.Str(pose, "frame_id") != "WORLD") throw new JsonException("robot initial_pose must be in WORLD");
            var p = J.Arr(pose, "position_m");
            s.InitialPositionWorld = new Vec3d(J.Num(p[0], "x"), J.Num(p[1], "y"), J.Num(p[2], "z"));
            if (J.Has(pose, "orientation_wxyz"))
            {
                var q = J.Arr(pose, "orientation_wxyz");
                s.InitialOrientationWorld = new Quatd(J.Num(q[0], "w"), J.Num(q[1], "x"), J.Num(q[2], "y"), J.Num(q[3], "z")).Normalized;
            }
            return s;
        }
    }

    /// <summary>JSONL replay log: header (seeds, versions, digests), every command, every fault, per-step truth summary.</summary>
    public sealed class ReplayLogger : IDisposable
    {
        private readonly StreamWriter _w;
        private readonly int _truthEveryNSteps;

        public ReplayLogger(string path, int truthEveryNSteps)
        {
            string dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            _w = new StreamWriter(path, false, new UTF8Encoding(false)) { NewLine = "\n", AutoFlush = false };
            _truthEveryNSteps = Math.Max(1, truthEveryNSteps);
        }

        public void Write(string kind, long simTimeNs, Dictionary<string, object> data)
        {
            var rec = new Dictionary<string, object> { ["kind"] = kind, ["sim_time_ns"] = simTimeNs, ["data"] = data };
            _w.WriteLine(MiniJson.Serialize(rec));
        }

        public bool ShouldLogTruth(long stepIndex) => stepIndex % _truthEveryNSteps == 0;

        public void Flush() => _w.Flush();

        public void Dispose()
        {
            _w.Flush();
            _w.Dispose();
        }
    }
}
