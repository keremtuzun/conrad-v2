# NetMQ binaries (not committed)

The transport assembly `ConradUnityV2.Transport` references three managed DLLs that must be placed in this
folder before the project compiles. They are not vendored in the repository.

| DLL | NuGet package | Tested target |
|---|---|---|
| `NetMQ.dll` | `NetMQ` 4.0.1.x (`lib/netstandard2.0`) | Unity 2022.3, .NET Standard 2.1 API level |
| `AsyncIO.dll` | `AsyncIO` 0.1.69 (`lib/netstandard2.0`) | dependency of NetMQ |
| `NaCl.dll` | `NaCl.Net` 0.1.13 (`lib/netstandard2.0`) | dependency of NetMQ 4.x |

Steps: download the three `.nupkg` files from nuget.org, open each as a zip, copy the `netstandard2.0` DLL
here, and in the Inspector enable "Any Platform" for each. Nothing else in the project depends on NetMQ; the
simulator core (`ConradUnityV2.asmdef`) compiles without it.

"Tested target" above is the intended configuration. This repository's environment could not run the Unity
editor, so the combination has not been compiled here (see ../../../README.md).
