# DualLLM Portal

A Windows desktop app that loads a large language model (a GGUF file) and spreads it across every
place you have to put it: the GPU in this PC, GPUs in other computers on your network, and the RAM
of either. Pick a model, tick the devices to use, press **Launch Chat**, and llama.cpp's web chat opens.

It does not reimplement inference. It drives [llama.cpp](https://github.com/ggml-org/llama.cpp):
`llama-server` runs on this PC and reaches other machines through llama.cpp's RPC backend
(`ggml-rpc-server`), which the portal starts on those machines over ssh and stops again when you are done.

- Pick any `.gguf` from a folder; the portal reads the model's metadata and estimates what context
  size fits on the devices you ticked (it measures free VRAM; it does not guess).
- Add as many GPUs and machines as you like in one config file. Nothing about any particular hardware
  is built in.
- Mixture-of-experts models can keep some of their expert layers in system RAM, on this PC or on a
  worker, while attention and the rest stay on GPUs.
- Optional helper servers (for example MCP bridges) can be switched on per session.
- A small loopback HTTP API lets other programs load and unload models without the UI
  ([docs/CONTROL_API.md](docs/CONTROL_API.md)).

## How it fits together

```
 this PC (Windows)                                 another computer (Linux)
 +-------------------------------+                +--------------------------+
 | DualLlmPortal.exe             |   ssh start    | ggml-rpc-server  GPU 0   |
 |   setup page (WebView2)       |--------------->| ggml-rpc-server  GPU 1   |
 |   launches llama-server  ---- | --rpc over LAN-| ggml-rpc-server  RAM     |
 | llama-server (local GPU/RAM)  |                +--------------------------+
 +-------------------------------+
        |
        +--> chat UI at http://<bindAddress>:<port>/   (default http://127.0.0.1:8080/)
```

Everything runs from one folder, the portal's **home** (the folder `DualLlmPortal.exe` is in; set
`DUALLM_HOME` to use another). It holds your config, the models folder and the llama.cpp build.

## Requirements

On the PC that runs the portal:

- Windows 10 or 11, 64-bit, with the [WebView2 runtime](https://developer.microsoft.com/microsoft-edge/webview2/) (already present on current Windows 11).
- An NVIDIA GPU with a current driver (`nvidia-smi` on the PATH) to use local GPU(s). Without one the portal
  still starts, and you can use RAM and remote devices.
- To build: the [.NET 10 SDK](https://dotnet.microsoft.com/download), git, CMake 3.24+, Visual Studio 2022 (or its Build
  Tools) with the "Desktop development with C++" workload, and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

For each extra computer you want to borrow devices from (optional):

- Linux with an NVIDIA GPU and driver, and the same toolchain minus .NET: git, CMake, a C++ compiler, CUDA Toolkit.
- An ssh server you can log into **with a key** from the portal PC, and a trusted network between the two
  (see [Security](#security)). Details in [docs/MULTI_MACHINE.md](docs/MULTI_MACHINE.md).

## Quick start

```powershell
git clone https://github.com/tur-ky/duallm.git
cd duallm

# 1. Build llama.cpp (clones it into .\llama.cpp and builds llama-server, llama-fit-params, ggml-rpc-server).
.\scripts\build-llamacpp.ps1

# 2. Build the portal and install DualLlmPortal.exe in this folder.
.\scripts\publish.ps1

# 3. Put models in .\llm_models (any *.gguf), then start the portal.
.\DualLlmPortal.exe
```

The first run writes a `duallm.config.json` for the NVIDIA GPUs it finds on this PC. Pick a model, tick
devices, choose a context size and press **Launch Chat**. Use **Open config** on the setup page to add
more machines or change anything else, then **Reload config**; no restart is needed.

`publish.ps1` builds into `.build\publish`, starts that build once to check it comes up and answers its
control API, and only then copies it over the installed exe. A running portal is renamed to
`DualLlmPortal.exe.old` rather than overwritten, so you can update while it is open.

A single-machine setup needs nothing else. To use a second computer, add it to the config
(see below) and follow [docs/MULTI_MACHINE.md](docs/MULTI_MACHINE.md) to build the worker.

## Configuration

`duallm.config.json` lives next to `DualLlmPortal.exe` (override with `DUALLM_CONFIG`). Comments and trailing
commas are allowed. Option names it does not know are reported on the setup page instead of being ignored,
so a typo is visible. If the file is invalid the portal starts with defaults and shows what is wrong.
[duallm.example.json](duallm.example.json) is a complete annotated example with one remote machine.

Relative paths are relative to the home folder.

### Top level

| Option | Default | Meaning |
|---|---|---|
| `modelsDir` | `llm_models` | Folder scanned for `*.gguf` (top level only, not subfolders). |
| `llamaServer` | see below | The local llama.cpp server. |
| `controlApi` | `{ enabled: true, port: 8091 }` | Loopback API for other programs. |
| `hosts` | `[]` | Other computers reached over ssh. |
| `devices` | local GPUs found | Everywhere the model may be placed. |
| `integrations` | `[]` | Optional helper servers. |

### `llamaServer`

| Option | Default | Meaning |
|---|---|---|
| `path` | `llama.cpp\build\bin\Release\llama-server.exe` | Server binary. |
| `fitParamsPath` | `llama.cpp\build\bin\Release\llama-fit-params.exe` | Used to measure what fits. Must be from the same build as `path`. |
| `port` | `8080` | Chat and API port. |
| `bindAddress` | `127.0.0.1` | `127.0.0.1` keeps the chat on this PC. `0.0.0.0` exposes it to your network. |
| `kvCacheType` | `q8_0` | Quantisation of both the K and V cache (`f16`, `q8_0`, `q4_0`, ...). Smaller means more context for the same memory. |
| `cudaPath` | `CUDA_PATH` environment variable | CUDA toolkit whose `bin` is added to the server's PATH. |
| `extraArgs` | none | Extra `llama-server` arguments added to every launch. |

### `hosts[]`: other computers

| Option | Meaning |
|---|---|
| `id` | Short name that devices refer to. |
| `label` | Shown in the UI. |
| `ssh` | ssh destination: `user@host` or an alias from `~/.ssh/config`. |
| `sshOptions` | Extra ssh arguments, for example `["-i", "C:\\keys\\worker", "-p", "2222"]`. |
| `address` | How this PC reaches the worker's RPC ports (IP or DNS name). |
| `rpcServerPath` | Absolute path of `ggml-rpc-server` on the worker. |
| `rpcBindAddress` | Address the worker listens on. Default `0.0.0.0`; use the worker's LAN address to limit it. |
| `cacheDir`, `cacheVolume` | Optional. Where `rpc-server -c` keeps its tensor cache, and the disk it is on; adds cache stats and a Clear button. |
| `startTimeoutSeconds` | How long to wait for a worker to start answering. Default 30. |

### `devices[]`: where the model can go

Devices are listed in the order llama.cpp fills them, so put GPUs before RAM.

| Option | Meaning |
|---|---|
| `id` | Unique name (used by the control API). |
| `type` | `"gpu"` (a CUDA card) or `"ram"` (system memory used as overflow). |
| `host` | Omit for this PC, otherwise the `id` of a host. |
| `label`, `chip` | Full name for messages, and short text for the toggle. |
| `vramGb` | Nominal size, used only for the quick "will it fit" bar. The real limit is measured. |
| `cudaIndex` | Which card on its machine (`nvidia-smi` index). |
| `port` | RPC port of a remote device. Each remote device is its own worker process and needs its own port. |
| `marginMib` | VRAM held back from the fit estimate. Default 768 local, 512 remote. |
| `threads` | CPU threads for a remote RAM worker. Default 4. |
| `extraArgs` | Extra `ggml-rpc-server` arguments, for example `-c` to cache tensors on the worker's disk. |
| `default` | Ticked when the portal starts. |

Rules the portal checks for you: ids are unique; a remote device names a known host and has a port;
two devices on one host cannot share a port or a card; at most one `ram` device per machine.

### `integrations[]`: optional helper servers

Each is a program the portal starts (and stops) with the model, then registers in llama.cpp's web UI
as an MCP server. Fields: `id`, `name`, `description`, `command`, `args`, `url`, `processName`.
`{home}` in `command` and `args` expands to the home folder. Enabling one turns on `--jinja`,
which tool calling needs.

## What gets launched

For a selection of one local GPU plus a remote GPU, the portal runs roughly:

```
llama-server -m model.gguf -ngl 999 -c 32768 -fa on -ctk q8_0 -ctv q8_0
             --rpc 192.168.1.50:50052 --device CUDA0,RPC0
```

and, on the other computer, over ssh:

```
env CUDA_VISIBLE_DEVICES=0 /home/me/llama.cpp/build/bin/ggml-rpc-server -c -H 0.0.0.0 -p 50052
```

The full command line is written to `portal-llama-server.log` in the home folder, with the server's
output. Ordering matters to llama.cpp (`--rpc` must come before `--device`), and the portal takes care of it.

## Security

- **llama.cpp's RPC protocol has no authentication and no encryption.** Anyone who can reach a worker's
  port can use its GPU and read what is sent to it. Use it only on a network you trust, and firewall the
  ports (`devices[].port`) to the portal PC. Set `rpcBindAddress` to the worker's LAN address instead of `0.0.0.0` if it has several interfaces.
- The chat server binds to `127.0.0.1` by default. If you set `bindAddress` to `0.0.0.0`, also set a Windows
  Firewall rule that allows only the devices you want; llama-server has no login unless you add `--api-key`
  through `extraArgs`.
- The control API listens on loopback only and has no authentication, so any program running as any user on
  this PC can load models through it. Turn it off with `"controlApi": { "enabled": false }` if you do not use it.
- ssh runs with `BatchMode=yes`: key-based login only, and the portal never asks for or stores a password.
- `duallm.config.json`, models and build outputs are in `.gitignore`. Your config stays on your machine.

## Troubleshooting

- **"Open config" shows a problems banner.** The text says which option is wrong; fix it and press Reload config.
- **llama-server not found.** Run `scripts\build-llamacpp.ps1`, or point `llamaServer.path` and `fitParamsPath` at your own build.
- **A remote device never comes up.** Run `ssh <host's ssh value> true` by hand first (this also trusts the host key),
  then check the worker's firewall for the device's port. `portal-llama-server.log` shows what failed.
- **Model loads, but output is garbage or the load fails oddly with a remote device.** The worker and the portal PC
  are on different llama.cpp commits, or were built with a different `GGML_MAX_NAME`. Rebuild both from the same
  `-Ref` ([docs/MULTI_MACHINE.md](docs/MULTI_MACHINE.md)).
- **Control API logs "The handle is invalid" and does not start.** The exe is running at Low integrity because its
  folder carries a Low mandatory label (some sandboxed tools add one to the directories they work in, and every
  file created there inherits it). Run the portal from a folder without that label, for example a clone made by
  a normal user process.
- **Windows asks about the firewall on first launch.** That is llama-server listening on its port; allow private networks only.

## Building and testing

```powershell
dotnet build DualLlmPortal.sln -c Release
cd DualLlmPortal.Tests; dotnet run -c Release   # 30 tests: launch-argument building, config validation, worker commands
```

Do not publish by hand with `dotnet publish -o <repo root>`; use `scripts\publish.ps1`, which stages and
checks the build first.

## Layout

```
DualLlmPortal/         WPF app: MainWindow (UI + orchestration), LaunchPlan (argument building),
                       PortalConfig (config), RemoteWorkers (ssh + rpc-server), ControlApi, SetupUi/setup.html
DualLlmPortal.Tests/   xunit tests for the pure logic
scripts/               build-llamacpp.ps1 / .sh, publish.ps1
docs/                  MULTI_MACHINE.md, CONTROL_API.md
duallm.example.json    annotated starting config
```

## License

[MIT](LICENSE). llama.cpp is MIT-licensed too; models have their own licenses.
