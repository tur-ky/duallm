# Using other computers

The portal can borrow GPUs and RAM from other computers on your network. Each borrowed device is a
`ggml-rpc-server` process that the portal starts over ssh; the portal PC's `llama-server` then reaches
it with `--rpc host:port`. Workers can run Linux or Windows ([Windows workers](#windows-workers) has the extra steps). The portal PC is Windows.

## 1. Build the worker

On the worker, with git, CMake 3.24+, a C++ compiler, the NVIDIA driver and the CUDA Toolkit installed:

```bash
./scripts/build-llamacpp.sh <ref>          # or copy the script over; it needs nothing else from this repo
```

The script clones llama.cpp into `~/llama.cpp` (override with `LLAMA_DIR`), builds only `ggml-rpc-server`
with CUDA, and prints the binary's absolute path. `CUDA_ARCH` selects the GPU architectures
(default `native`; use for example `"86;75"` if the worker has cards of different generations).

On the portal PC, build with the **same ref**:

```powershell
.\scripts\build-llamacpp.ps1 -Ref <ref>
```

### The two builds must match

llama.cpp's RPC protocol is not versioned or negotiated, and mismatches fail in ways that do not point at the cause:

- **Same llama.cpp commit on every machine.** Different commits can load a model and then produce garbage, or
  fail partway through loading. Use an explicit `-Ref` (a tag or commit hash) on both sides rather than the default branch, so the two cannot drift.
- **Same `GGML_MAX_NAME`** if you change it from the default. Tensor names are sent over the wire at a fixed
  size. Stable-diffusion.cpp for example needs a larger value, and a worker built for it will not talk to a stock `llama-server`.
- The KV cache type is chosen on the portal PC only, so it needs no matching on workers.

When you update llama.cpp, update every machine.

## 2. Set up ssh

The portal uses your normal `ssh` client with `BatchMode=yes`, so it never prompts: key login only.

1. Create a key if you do not have one: `ssh-keygen -t ed25519`.
2. Install it on the worker: `ssh-copy-id me@worker.local` (or append the `.pub` file to `~/.ssh/authorized_keys`).
3. Log in once by hand, `ssh me@worker.local true`, to accept the host key. The portal does not auto-accept it.
4. Optional: add an alias to `~/.ssh/config` (`C:\Users\<you>\.ssh\config`) and put the alias in `hosts[].ssh`.
   A non-default key or port can go in `hosts[].sshOptions` instead.

The ssh session that starts a worker stays open for that worker's lifetime. Closing the portal, or pressing
**Release GPUs**, stops the workers (`killall`/`pkill` of `ggml-rpc-server` for that host) and frees their VRAM.

> Stopping uses the process name, so a `ggml-rpc-server` you started yourself on the same worker is stopped too.

## 3. Describe it in `duallm.config.json`

```jsonc
"hosts": [
  { "id": "worker", "label": "Linux box", "ssh": "worker",       // alias or user@host
    "address": "192.168.1.50",                                   // what this PC connects to
    "rpcServerPath": "/home/me/llama.cpp/build/bin/ggml-rpc-server" }
],
"devices": [
  // one entry per card; cudaIndex is its nvidia-smi index on that machine, port must be unique per host
  { "id": "worker-gpu0", "type": "gpu", "host": "worker", "label": "Worker GPU 0", "chip": "W-GPU0",
    "vramGb": 8, "cudaIndex": 0, "port": 50052, "extraArgs": "-c", "default": true },
  { "id": "worker-gpu1", "type": "gpu", "host": "worker", "label": "Worker GPU 1", "chip": "W-GPU1",
    "vramGb": 8, "cudaIndex": 1, "port": 50054, "extraArgs": "-c" },
  // the worker's system RAM, served by a CPU-only rpc-server
  { "id": "worker-ram", "type": "ram", "host": "worker", "label": "the worker's RAM", "chip": "Worker RAM",
    "port": 50053, "threads": 8 }
]
```

Then press **Reload config** on the setup page. The new devices appear as toggles.

Notes:

- Each GPU or RAM device is a separate `ggml-rpc-server` process on its own port. Open those ports on the
  worker's firewall to the portal PC only.
- `extraArgs: "-c"` makes the worker cache received tensors on its own disk, so the second load of a model
  skips most of the network transfer. Loading the first time is network-bound: a 1 Gbit link moves roughly
  110 MB/s. Set `hosts[].cacheDir` (the cache folder, `~/.cache/llama.cpp/rpc` by default) and `cacheVolume` (its disk) to see
  the cache size on the setup page and clear it.
- GPUs fill first, in the order of `devices`. Put RAM devices after GPUs.
- A RAM device does not run the model's compute on the GPU; it holds weights (for mixture-of-experts models,
  the expert layers) in memory. Set how many expert layers it takes on the setup page after ticking the RAM device.
  This is slower than VRAM, but lets a model larger than your total VRAM load at all.

## 4. Check it

Tick the remote device, pick a small model, and launch. In `portal-llama-server.log` you should see
llama.cpp list `RPC0` among its devices and place layers on it. On the worker, `nvidia-smi` shows memory in use
and `pgrep -a ggml-rpc-server` shows the process. After **Release GPUs**, both should be empty again.

## Windows workers

A Windows PC or laptop with an NVIDIA card can be a worker too. Set `"os": "windows"` on its host and the
portal sends it PowerShell commands instead of Linux ones. It works whether the laptop's ssh server hands
commands to cmd.exe (the default) or to PowerShell. Everything else in steps 1 to 4 applies, with these
differences. Run the commands below in **PowerShell as administrator on the worker** unless they say otherwise.

### Build

Clone this repository on the worker and build with the **same ref** as the portal PC. It needs the same
toolchain as the portal PC (Visual Studio 2022 or Build Tools with C++, CMake, git, CUDA Toolkit):

```powershell
.\scripts\build-llamacpp.ps1 -Ref <ref>
```

The worker binary is then `<clone>\llama.cpp\build\bin\Release\ggml-rpc-server.exe`. The script also builds
`llama-server`, which the worker does not need but which does no harm.

### Turn on the ssh server

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service sshd -StartupType Automatic
```

Installing it also opens port 22 in Windows Firewall. Your ssh user name is the part after the backslash in
`whoami`.

Where the portal PC's public key goes depends on the worker account:

- **Administrator accounts** (the usual case on a personal laptop): Windows ignores `~\.ssh\authorized_keys`
  for administrators. The key goes in `C:\ProgramData\ssh\administrators_authorized_keys`, and that file must
  be readable only by Administrators and SYSTEM, or sshd ignores it:

  ```powershell
  Add-Content -Path C:\ProgramData\ssh\administrators_authorized_keys -Value '<the line from id_ed25519.pub on the portal PC>'
  icacls.exe C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
  ```

- **Standard accounts**: the key goes in `C:\Users\<you>\.ssh\authorized_keys` on the worker, as on Linux.

Then, **on the portal PC**, log in once to trust the host key: `ssh me@laptop hostname` (use `hostname` rather
than `true`, which cmd.exe does not have).

### Open the worker ports

Windows asks about the firewall only when a program on the desktop starts listening. A worker started over
ssh has no desktop, so nothing asks and the connection is silently blocked. Add the rule yourself, limited to
the portal PC, with one port per device you configure:

```powershell
New-NetFirewallRule -DisplayName "DualLLM RPC workers" -Direction Inbound -Protocol TCP `
  -LocalPort 50052,50053 -RemoteAddress <portal PC's IP> -Profile Private -Action Allow
```

Make sure the worker's network is set to **Private** (Settings, Network & internet, the connection's properties).

### Config

```jsonc
"hosts": [
  { "id": "laptop", "label": "Laptop", "ssh": "me@192.168.1.60", "os": "windows",
    "address": "192.168.1.60",
    "rpcServerPath": "C:\\Users\\me\\duallm\\llama.cpp\\build\\bin\\Release\\ggml-rpc-server.exe" }
],
"devices": [
  { "id": "laptop-gpu", "type": "gpu", "host": "laptop", "label": "Laptop RTX 3050", "chip": "3050",
    "vramGb": 6, "cudaIndex": 0, "port": 50052, "extraArgs": "-c" }
]
```

For the cache controls, llama.cpp keeps the tensor cache under `%LOCALAPPDATA%\llama.cpp\rpc` on Windows
(unless `LLAMA_CACHE` is set), so `"cacheDir": "C:\\Users\\me\\AppData\\Local\\llama.cpp\\rpc"` and
`"cacheVolume": "C:\\"`. Check that the folder exists after a first load with `-c`.

### What is different on Windows

- **The split check reads the whole card.** Windows does not report how much GPU memory each program uses
  (nvidia-smi shows `[N/A]`), so for a Windows host the portal compares the card's total memory in use before
  and after the load. Anything else that grabs GPU memory on the worker during the load, such as a game,
  skews that number.
- **Stopping always ends the process outright**, by name (`ggml-rpc-server`), like the forced stop on Linux.
  A `ggml-rpc-server` you started yourself on that machine is stopped too.
- **Laptops**: keep it plugged in (on battery the GPU slows down a lot), set sleep to Never when plugged in,
  and set closing the lid to Do nothing if you run it closed. If the laptop sleeps, its worker disappears
  mid-chat. Use Ethernet if you can: Wi-Fi makes the first load slower and adds a little delay to every token.

## Opening the chat to other devices

By default the chat is only reachable from the portal PC. To use it from a phone or laptop on your network,
set `llamaServer.bindAddress` to `0.0.0.0` (or the PC's LAN address) and allow the port (default 8080) in the
Windows Firewall for private networks only. llama-server has no login: add `--api-key YOUR_KEY` to
`llamaServer.extraArgs` if the network is not entirely yours. Over the internet, put it behind a VPN such as WireGuard rather than forwarding the port.
