using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace DualLlmPortal;

/// <summary>
/// Starts, probes and stops llama.cpp RPC workers on the configured hosts over ssh.
///
/// A worker is just an ssh session whose remote command is ggml-rpc-server. The session is held open
/// for the worker's whole life: setsid and nohup do not help, because the server dies the moment the
/// session closes, so the running ssh IS the worker and killing it is how a worker is stopped.
/// </summary>
internal sealed class RemoteWorkers
{
    // Parses "…/ggml-rpc-server, 7078 MiB" out of nvidia-smi --query-compute-apps output.
    private static readonly Regex MibRegex = new(@"(\d+)\s*MiB", RegexOptions.Compiled);

    private readonly PortalConfig _config;
    private readonly object _gate = new();
    private readonly Dictionary<string, Process> _sessions = new();   // device id -> ssh session
    private readonly HashSet<string> _touchedHosts = new();           // hosts we ever started a worker on

    // Starts are serialised: a context estimate and a launch can both ask for the same worker, and the
    // second must see the first one's port answering rather than race it to bind.
    private readonly SemaphoreSlim _startLock = new(1, 1);

    public RemoteWorkers(PortalConfig config) => _config = config;

    /// <summary>True once any worker has been started, including one started only to take a measurement.</summary>
    public bool AnyStarted
    {
        get { lock (_gate) { return _touchedHosts.Count > 0; } }
    }

    /// <summary>The command a host runs to serve one device. Public so it can be tested.</summary>
    public static string BuildWorkerCommand(HostConfig host, DeviceConfig device)
        => host.IsWindows
            ? Shell.PowerShellCommand(BuildWindowsWorkerScript(host, device))
            : BuildPosixWorkerCommand(host, device);

    private static string BuildPosixWorkerCommand(HostConfig host, DeviceConfig device)
    {
        // `env` rather than a VAR=value prefix so it works whatever the remote login shell is.
        var parts = new List<string>();
        if (device.IsGpu)
        {
            parts.Add("env");
            parts.Add($"CUDA_VISIBLE_DEVICES={device.CudaIndex}");
        }
        parts.Add(Shell.PosixQuote(host.RpcServerPath));
        if (device.IsRam)
        {
            // The CPU backend publishes the host's system memory as an ordinary llama.cpp device.
            parts.Add($"-d CPU -t {device.Threads}");
        }
        if (!string.IsNullOrWhiteSpace(device.ExtraArgs))
        {
            parts.Add(device.ExtraArgs.Trim());
        }
        parts.Add($"-H {host.RpcBindAddress} -p {device.Port}");
        return string.Join(" ", parts);
    }

    /// <summary>
    /// The PowerShell a Windows host runs to serve one device, before it is encoded for ssh. Public so
    /// it can be tested. The CUDA toolkit's folders go on PATH first because an ssh session does not
    /// always see them, and ggml-rpc-server.exe cannot start without the CUDA runtime DLLs.
    /// </summary>
    public static string BuildWindowsWorkerScript(HostConfig host, DeviceConfig device)
    {
        var lines = new List<string>
        {
            "if ($env:CUDA_PATH) { $env:PATH = \"$env:CUDA_PATH\\bin;$env:CUDA_PATH\\bin\\x64;$env:PATH\" }",
        };
        if (device.IsGpu)
        {
            lines.Add($"$env:CUDA_VISIBLE_DEVICES = '{device.CudaIndex}'");
        }

        var args = new List<string>();
        if (device.IsRam)
        {
            args.Add($"-d CPU -t {device.Threads}");
        }
        if (!string.IsNullOrWhiteSpace(device.ExtraArgs))
        {
            args.Add(device.ExtraArgs.Trim());
        }
        args.Add($"-H {host.RpcBindAddress} -p {device.Port}");

        lines.Add($"& {Shell.PowerShellQuote(host.RpcServerPath)} {string.Join(" ", args)}");
        lines.Add("exit $LASTEXITCODE");
        return string.Join("; ", lines);
    }

    /// <summary>The command that stops every worker on a host. Public so it can be tested.</summary>
    public static string BuildKillCommand(HostConfig host, bool force)
    {
        var path = host.RpcServerPath.Replace('\\', '/');

        if (host.IsWindows)
        {
            // Windows has no gentle signal for a console program started over ssh, so both kinds of
            // stop end the process outright. Get-Process wants the name without ".exe".
            var exe = Path.GetFileNameWithoutExtension(path);
            if (string.IsNullOrEmpty(exe)) return "";
            return Shell.PowerShellCommand(
                $"Get-Process -Name {Shell.PowerShellQuote(exe)} -ErrorAction SilentlyContinue | " +
                "Stop-Process -Force -ErrorAction SilentlyContinue; exit 0");
        }

        var name = Path.GetFileName(path);
        if (string.IsNullOrEmpty(name)) return "";

        // killall (psmisc) where present, pkill (procps) otherwise; pkill -x only sees the first 15
        // characters of a process name, hence the truncation. Exit status is irrelevant: "nothing to
        // kill" is the good outcome.
        var q = Shell.PosixQuote(name);
        var q15 = Shell.PosixQuote(name.Length > 15 ? name[..15] : name);
        var signal = force ? "-9 " : "";
        return $"killall {signal}{q} 2>/dev/null || pkill {signal}-x {q15} 2>/dev/null; true";
    }

    /// <summary>
    /// An nvidia-smi call for <paramref name="host"/>. On Windows each argument is quoted, because
    /// PowerShell can read an unquoted "index,memory.free" as a list rather than one word.
    /// </summary>
    public static string BuildNvidiaSmiCommand(HostConfig host, params string[] args)
        => host.IsWindows
            ? Shell.PowerShellCommand(
                "& nvidia-smi " + string.Join(" ", args.Select(Shell.PowerShellQuote)) + "; exit $LASTEXITCODE")
            : "nvidia-smi " + string.Join(" ", args);

    /// <summary>
    /// Makes sure the worker for <paramref name="device"/> is accepting connections, starting it if
    /// not. Returns immediately when it already is (from a context estimate a moment ago, or an
    /// earlier launch). Throws with a readable reason when it cannot be brought up.
    /// </summary>
    public async Task EnsureRunningAsync(DeviceConfig device)
    {
        var host = _config.HostOf(device)
            ?? throw new InvalidOperationException($"Device \"{device.Label}\" has no remote host.");

        await _startLock.WaitAsync();
        try
        {
            if (await Shell.IsReachableAsync(host.Address, device.Port, 1500))
            {
                return;
            }

            lock (_gate) { _touchedHosts.Add(host.Id); }

            // Clear workers left behind by a crashed earlier session, which can hold a port or a GPU's
            // memory. Only when none of this host's workers answers: otherwise the sweep would take down
            // the workers the caller has already got running.
            if (!await AnyWorkerReachableAsync(host))
            {
                await Task.Run(() => KillRemoteWorkers(host, force: false, timeoutMs: 15000));
            }

            await StartSessionAsync(host, device);
        }
        finally
        {
            _startLock.Release();
        }
    }

    private async Task<bool> AnyWorkerReachableAsync(HostConfig host)
    {
        var probes = _config.Devices
            .Where(d => !d.IsLocal && d.Host == host.Id)
            .Select(d => Shell.IsReachableAsync(host.Address, d.Port, 1500));
        return (await Task.WhenAll(probes)).Any(reachable => reachable);
    }

    private async Task StartSessionAsync(HostConfig host, DeviceConfig device)
    {
        var psi = Shell.Ssh(host, BuildWorkerCommand(host, device));
        var stderrTail = new StringBuilder();

        var process = new Process { StartInfo = psi };
        process.OutputDataReceived += (_, __) => { };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data == null) return;
            lock (stderrTail)
            {
                stderrTail.AppendLine(e.Data);
                if (stderrTail.Length > 4096) stderrTail.Remove(0, stderrTail.Length - 4096);
            }
        };

        process.Start();
        lock (_gate) { _sessions[device.Id] = process; }

        // The server runs in the foreground, so its log streams back through these pipes for the life
        // of the session. Left undrained they fill and stall the remote worker.
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        // Wait for the worker to actually accept connections rather than sleeping a fixed guess.
        // Loading a model into an unreachable worker is what produces a "cluster" run that quietly
        // uses only the local GPU.
        var timeout = TimeSpan.FromSeconds(Math.Max(5, host.StartTimeoutSeconds));
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (await Shell.IsReachableAsync(host.Address, device.Port, 2000))
            {
                return;
            }

            if (process.HasExited)
            {
                string tail;
                lock (stderrTail) { tail = stderrTail.ToString().Trim(); }
                throw new Exception(
                    $"The ssh session that starts the worker for {device.Label} on {host.DisplayName} " +
                    $"exited early (code {process.ExitCode}). Check that 'ssh {host.Ssh}' works without a " +
                    $"password and that {host.RpcServerPath} exists." +
                    (tail.Length > 0 ? $"\n\n{tail}" : ""));
            }

            await Task.Delay(1000);
        }

        throw new Exception(
            $"Started the worker for {device.Label} on {host.DisplayName} but {host.Address}:{device.Port} " +
            $"never began accepting connections within {timeout.TotalSeconds:0}s. That device would not be used.");
    }

    /// <summary>
    /// Stops every worker this portal started: closes the ssh sessions, then sweeps each touched host
    /// for any ggml-rpc-server that survived. Keyed off "did we ever start one" rather than the
    /// current ticks, because a worker may have been started just to size the context and never
    /// launched against. Blocks on process kills and ssh round trips, so call it off the UI thread.
    /// </summary>
    public void StopAll()
    {
        List<Process> sessions;
        List<string> hostIds;
        lock (_gate)
        {
            sessions = _sessions.Values.ToList();
            _sessions.Clear();
            hostIds = _touchedHosts.ToList();
        }

        foreach (var session in sessions)
        {
            Shell.KillTree(session);
            try { session.Dispose(); } catch { }
        }

        foreach (var hostId in hostIds)
        {
            var host = _config.Hosts.FirstOrDefault(h => h.Id == hostId);
            if (host == null) continue;

            try
            {
                KillRemoteWorkers(host, force: true, timeoutMs: 5000);
                lock (_gate) { _touchedHosts.Remove(hostId); }
            }
            catch
            {
                // Leave the host marked so the next stop tries again.
            }
        }
    }

    // Synchronous on purpose: StopAll is called from the window's Closing handler on the UI thread,
    // where blocking on an async method that resumes on that thread would deadlock.
    private static void KillRemoteWorkers(HostConfig host, bool force, int timeoutMs)
    {
        var command = BuildKillCommand(host, force);
        if (command.Length == 0) return;

        var psi = Shell.Ssh(host, command);
        using var process = new Process { StartInfo = psi };
        process.OutputDataReceived += (_, __) => { };
        process.ErrorDataReceived += (_, __) => { };
        process.Start();
        // Drain before waiting: WaitForExit on a process with redirected streams nobody reads can
        // block forever if the child fills a pipe buffer.
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        process.WaitForExit(timeoutMs);
    }

    // ------------------------------------------------------------------ memory readings

    /// <summary>Free VRAM in MiB per CUDA index on this PC, or null if nvidia-smi gave no answer.</summary>
    public static async Task<Dictionary<int, double>?> QueryLocalFreeMibAsync()
        => ParseIndexedMib(await Shell.RunCaptureAsync(
            "nvidia-smi", "--query-gpu=index,memory.free --format=csv,noheader,nounits"));

    /// <summary>Free VRAM in MiB per CUDA index on a host, or null if it could not be read.</summary>
    public static async Task<Dictionary<int, double>?> QueryFreeMibAsync(HostConfig host)
        => ParseIndexedMib(await Shell.RunSshCaptureAsync(
            host, BuildNvidiaSmiCommand(host, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits")));

    private static Dictionary<int, double>? ParseIndexedMib(string? text)
    {
        if (text == null) return null;

        var result = new Dictionary<int, double>();
        foreach (var line in text.Split('\n'))
        {
            var parts = line.Split(',');
            if (parts.Length >= 2 &&
                int.TryParse(parts[0].Trim(), out var index) &&
                double.TryParse(parts[1].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var mib))
            {
                result[index] = mib;
            }
        }
        return result;
    }

    /// <summary>
    /// Total VRAM the RPC workers hold on <paramref name="hosts"/>, straight from nvidia-smi.
    ///
    /// This deliberately does not parse llama-server's log. llama.cpp only prints its
    /// "model buffer size" lines above the default verbosity, so at normal settings there is simply no
    /// text stating where the weights went. The workers' own GPU allocation is the ground truth and
    /// does not depend on llama.cpp's log format. Returns -1 when a reading could not be taken, which is
    /// distinct from a real 0.
    ///
    /// Windows is the exception. Its graphics driver model (WDDM, which consumer GPUs on Windows use)
    /// hides per-process memory, so nvidia-smi answers "[N/A]" there. For a Windows host this sums each
    /// card's total memory in use instead. The caller compares a reading taken before the model loads
    /// with one taken after, so whatever else the card was already holding cancels out.
    /// </summary>
    public static async Task<double> GetWorkerVramMibAsync(IEnumerable<HostConfig> hosts)
    {
        double total = 0;
        foreach (var host in hosts)
        {
            if (host.IsWindows)
            {
                var used = ParseIndexedMib(await Shell.RunSshCaptureAsync(
                    host, BuildNvidiaSmiCommand(host, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits")));
                if (used == null)
                {
                    return -1;
                }
                total += used.Values.Sum();
                continue;
            }

            var output = await Shell.RunSshCaptureAsync(
                host, BuildNvidiaSmiCommand(host, "--query-compute-apps=process_name,used_memory", "--format=csv,noheader"));
            if (output == null)
            {
                return -1;
            }

            foreach (var line in output.Split('\n'))
            {
                if (line.IndexOf("rpc-server", StringComparison.OrdinalIgnoreCase) < 0) continue;

                var m = MibRegex.Match(line);
                if (m.Success && double.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var mib))
                {
                    total += mib;
                }
            }
        }

        // No worker process on the GPU at all is a legitimate zero, not a failed reading.
        return total;
    }

    // ------------------------------------------------------------------ tensor cache

    /// <summary>
    /// One ssh round trip: bytes used by the host's tensor cache and free bytes on the volume that
    /// holds it. Null if either query fails; the setup page treats that as "measuring…".
    /// </summary>
    public static async Task<(long UsedBytes, long FreeBytes)?> GetCacheStatsAsync(HostConfig host)
    {
        if (!host.HasCache) return null;

        var output = await Shell.RunSshCaptureAsync(host, BuildCacheStatsCommand(host), 8000);
        if (output == null) return null;

        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        if (lines.Length < 2) return null;
        if (!long.TryParse(lines[0].Trim(), out var used)) used = 0;
        if (!long.TryParse(lines[1].Trim(), out var free)) return null;
        return (used, free);
    }

    /// <summary>
    /// Removes every tensor blob under the host's cache directory but keeps the directory itself, so
    /// the next worker start does not need to recreate it. The path comes from the config, not from the
    /// page, so a message from the UI cannot aim this at anything else; find -delete is used over a glob
    /// rm so an empty directory does not fail.
    /// </summary>
    public static async Task ClearCacheAsync(HostConfig host)
    {
        if (!host.HasCache) return;

        await Shell.RunSshCaptureAsync(host, BuildClearCacheCommand(host), 30000);
    }

    /// <summary>Prints the cache's size in bytes, then the free bytes on its volume, one per line.</summary>
    public static string BuildCacheStatsCommand(HostConfig host)
    {
        if (host.IsWindows)
        {
            var dir = Shell.PowerShellQuote(host.CacheDir!);
            var volume = Shell.PowerShellQuote(host.CacheVolume!);
            return Shell.PowerShellCommand(
                $"$d = {dir}; $v = {volume}; $used = 0; " +
                "if (Test-Path -LiteralPath $d) { $s = (Get-ChildItem -LiteralPath $d -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum; if ($s) { $used = [long]$s } }; " +
                "$free = (New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($v))).AvailableFreeSpace; " +
                "Write-Output $used; Write-Output $free");
        }

        var posixDir = Shell.PosixQuote(host.CacheDir!);
        var posixVolume = Shell.PosixQuote(host.CacheVolume!);
        return $"du -sb {posixDir} 2>/dev/null | cut -f1; df -B1 --output=avail {posixVolume} | tail -1";
    }

    /// <summary>Empties the cache directory of files and keeps the directory itself.</summary>
    public static string BuildClearCacheCommand(HostConfig host)
    {
        if (host.IsWindows)
        {
            var dir = Shell.PowerShellQuote(host.CacheDir!);
            return Shell.PowerShellCommand(
                $"$d = {dir}; if (Test-Path -LiteralPath $d) {{ Get-ChildItem -LiteralPath $d -Recurse -File -Force | Remove-Item -Force }}");
        }

        var posixDir = Shell.PosixQuote(host.CacheDir!);
        return $"if [ -d {posixDir} ]; then find {posixDir} -type f -delete; fi";
    }
}
