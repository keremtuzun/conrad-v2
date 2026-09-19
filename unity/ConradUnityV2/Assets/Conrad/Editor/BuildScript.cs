// Editor-only: builds the minimal V2.0 headless scene programmatically and the Windows64 standalone player.
// Batch use (repo root):
//   Unity.exe -batchmode -nographics -quit -projectPath unity/ConradUnityV2 -logFile <log>
//             -executeMethod Conrad.UnityV2.Editor.BuildScript.BuildWindows64Player
// The scene: a Vehicle (Rigidbody + hull BoxCollider + VehicleDynamics6Dof + ExperimentRuntime + TcpBridgeServer
// + ConradHeadless, a child camera and a sonar mount), a directional light, and a World root with a seafloor
// box, a pipe capsule and an obstacle box (static colliders, Conrad WORLD geometry through SceneGeometryBuilder).
// Every physical parameter is loaded at run time from the RobotConfig JSON; nothing physical is baked in here.
using System;
using System.Collections.Generic;
using System.IO;
using Conrad.UnityV2.EnvironmentInteraction;
using Conrad.UnityV2.ExternalInterfaces.Transport;
using Conrad.UnityV2.MissionExperimentRuntime;
using Conrad.UnityV2.VehicleDynamics;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Conrad.UnityV2.Editor
{
    public static class BuildScript
    {
        public const string ScenePath = "Assets/Scenes/ConradHeadless.unity";
        public const string PlayerDirectory = "Builds/Win64";
        public const string PlayerExe = "ConradSim.exe";
        public const int IgnoreRaycastLayer = 2; // the vehicle never occludes its own sonar

        private static Dictionary<string, object> Box(string id, double[] center, double[] size) => new Dictionary<string, object>
        {
            ["kind"] = "box", ["id"] = id, ["center_m"] = new List<object> { center[0], center[1], center[2] },
            ["size_m"] = new List<object> { size[0], size[1], size[2] },
        };

        private static void SetLayerRecursively(GameObject go, int layer)
        {
            go.layer = layer;
            foreach (Transform child in go.transform) SetLayerRecursively(child.gameObject, layer);
        }

        [MenuItem("Conrad/Set Up Headless Scene")]
        public static void SetupHeadlessScene()
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var light = new GameObject("Sun").AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.0f;
            light.transform.rotation = Quaternion.Euler(60f, -30f, 0f);
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.25f, 0.3f, 0.35f);

            // Static world (Conrad WORLD, m): seafloor top at z = -30, a pipe lying on it, one obstacle.
            var world = new GameObject("World").transform;
            SceneGeometryBuilder.Box(world, "seafloor", Box("seafloor", new[] { 0.0, 0.0, -30.5 }, new[] { 400.0, 400.0, 1.0 }));
            SceneGeometryBuilder.Capsule(world, "pipe", new Dictionary<string, object>
            {
                ["kind"] = "capsule", ["id"] = "pipe", ["radius_m"] = 0.3,
                ["p0_m"] = new List<object> { 12.0, -40.0, -29.7 }, ["p1_m"] = new List<object> { 12.0, 40.0, -29.7 },
            });
            SceneGeometryBuilder.Box(world, "obstacle", Box("obstacle", new[] { 60.0, 0.0, -10.0 }, new[] { 2.0, 8.0, 8.0 }));

            var vehicle = new GameObject("Vehicle");
            var rb = vehicle.AddComponent<Rigidbody>();
            rb.useGravity = false;
            vehicle.AddComponent<BoxCollider>();
            var dynamics = vehicle.AddComponent<VehicleDynamics6Dof>();
            var runtime = vehicle.AddComponent<ExperimentRuntime>();
            vehicle.AddComponent<TcpBridgeServer>();
            vehicle.AddComponent<ConradHeadless>();
            runtime.vehicle = dynamics;
            runtime.worldRoot = world;

            // Sensor bindings for configs/robot/sim_reference.yaml ("camera", "sonar"). Mount poses are applied at
            // run time from the RobotConfig; sensors with other names get a mount created automatically.
            var camGo = new GameObject("Camera_camera");
            camGo.transform.SetParent(vehicle.transform, false);
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.05f, 0.25f, 0.30f);
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 200f;
            cam.enabled = false;
            runtime.cameras.Add(new NamedCamera { sensorName = "camera", camera = cam });
            var sonarGo = new GameObject("SonarMount_sonar");
            sonarGo.transform.SetParent(vehicle.transform, false);
            runtime.sonarMounts.Add(new NamedTransform { sensorName = "sonar", mount = sonarGo.transform });
            SetLayerRecursively(vehicle, IgnoreRaycastLayer);

            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            if (!EditorSceneManager.SaveScene(scene, ScenePath)) throw new IOException("could not save " + ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            Debug.Log("[Conrad] headless scene written to " + ScenePath);
        }

        [MenuItem("Conrad/Build Windows64 Player")]
        public static void BuildWindows64Player()
        {
            SetupHeadlessScene();
            PlayerSettings.productName = "ConradSim";
            PlayerSettings.companyName = "Conrad";
            PlayerSettings.runInBackground = true;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 320;
            PlayerSettings.defaultScreenHeight = 240;
            PlayerSettings.resizableWindow = false;
            PlayerSettings.usePlayerLog = true;
            PlayerSettings.SetScriptingBackend(UnityEditor.Build.NamedBuildTarget.Standalone, ScriptingImplementation.Mono2x);

            string output = Path.Combine(PlayerDirectory, PlayerExe);
            var options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = output,
                target = BuildTarget.StandaloneWindows64,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.StrictMode,
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            BuildSummary s = report.summary;
            string info = "{\"unity_version\":\"" + Application.unityVersion + "\",\"result\":\"" + s.result
                          + "\",\"output\":\"" + output.Replace("\\", "/") + "\",\"total_size_bytes\":" + s.totalSize
                          + ",\"total_errors\":" + s.totalErrors + ",\"total_warnings\":" + s.totalWarnings
                          + ",\"build_started_utc\":\"" + s.buildStartedAt.ToUniversalTime().ToString("o")
                          + "\",\"build_seconds\":" + s.totalTime.TotalSeconds.ToString(System.Globalization.CultureInfo.InvariantCulture) + "}";
            Directory.CreateDirectory(PlayerDirectory);
            File.WriteAllText(Path.Combine(PlayerDirectory, "conrad_build_info.json"), info);
            Debug.Log("[Conrad] build " + s.result + ": " + info);
            if (s.result != BuildResult.Succeeded)
                throw new Exception("Windows64 player build failed: " + s.result + " (" + s.totalErrors + " errors)");
        }
    }
}
