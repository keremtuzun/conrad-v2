// Headless player entry point. The built player (Builds/Win64/ConradSim.exe) is started by Python with
//   ConradSim.exe -batchmode -logFile <log> -conradRobotConfig <json> -conradScenario <json> -conradExperiment <json>
//                 [-conradParentPid <pid>] [-conradIdleExitS <seconds>]
// (-nographics is optional: without it the camera renders through the GPU; with it the camera is refused.)
// It never advances time by itself in lock-step mode: TcpBridgeServer serves STEP requests, and ExperimentRuntime
// advances the world only inside them. The player exits with
//   0 on normal quit, 3 when the experiment inputs were refused (e.g. an OPEN RobotConfig parameter),
//   4 when the parent process disappeared, 5 when no control client connected within -conradIdleExitS.
using System;
using System.Diagnostics;
using Conrad.UnityV2.MissionExperimentRuntime;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Conrad.UnityV2.ExternalInterfaces.Transport
{
    [DefaultExecutionOrder(-1000)]
    public sealed class ConradHeadless : MonoBehaviour
    {
        public const int ExitRefusedInputs = 3;
        public const int ExitParentGone = 4;
        public const int ExitIdle = 5;

        private ExperimentRuntime _rt;
        private TcpBridgeServer _server;
        private int _parentPid = -1;
        private double _idleExitS = 300.0;
        private float _lastWatchdog;
        private double _idleSinceS;
        private bool _quitting;

        private static string Arg(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++) if (args[i] == name) return args[i + 1];
            return null;
        }

        private void Awake()
        {
            Application.runInBackground = true;
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
            if (int.TryParse(Arg("-conradParentPid"), out int pid)) _parentPid = pid;
            if (double.TryParse(Arg("-conradIdleExitS"), System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out double idle)) _idleExitS = idle;
        }

        private void Start()
        {
            _rt = GetComponent<ExperimentRuntime>();
            _server = GetComponent<TcpBridgeServer>();
            Debug.Log("[Conrad] headless player: unity=" + Application.unityVersion + " graphics="
                      + SystemInfo.graphicsDeviceType + " batchmode=" + Application.isBatchMode);
            if (_rt == null || !_rt.Ready) Quit(ExitRefusedInputs, "experiment inputs refused: " + (_rt == null ? "no runtime" : _rt.FailureReason));
            _idleSinceS = Time.realtimeSinceStartupAsDouble;
        }

        private void Quit(int code, string why)
        {
            if (_quitting) return;
            _quitting = true;
            if (code == 0) Debug.Log("[Conrad] exiting: " + why);
            else Debug.LogError("[Conrad] exiting with code " + code + ": " + why);
            Application.Quit(code);
        }

        private void Update()
        {
            if (_quitting || Time.realtimeSinceStartup - _lastWatchdog < 1f) return;
            _lastWatchdog = Time.realtimeSinceStartup;
            if (_parentPid > 0)
            {
                bool alive;
                try { alive = !Process.GetProcessById(_parentPid).HasExited; }
                catch (ArgumentException) { alive = false; }
                catch (InvalidOperationException) { alive = false; }
                if (!alive) { Quit(ExitParentGone, "parent process " + _parentPid + " is gone"); return; }
            }
            if (_server != null && _server.HasControlClient) _idleSinceS = Time.realtimeSinceStartupAsDouble;
            else if (_idleExitS > 0 && Time.realtimeSinceStartupAsDouble - _idleSinceS > _idleExitS)
                Quit(ExitIdle, "no control client for " + _idleExitS + " s");
        }
    }
}
