using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DualLlmPortal;

/// <summary>
/// Everything that differs between one person's machines and another's lives in this file's model:
/// where llama.cpp is, which GPUs exist, which other computers hold workers, and how to reach them.
/// The rest of the portal is written against these types and never names a device or host itself.
///
/// The file is JSON with comments and trailing commas allowed (duallm.config.json next to the exe).
/// Unknown properties are rejected rather than ignored, so a misspelt option fails loudly at load
/// instead of silently doing nothing.
/// </summary>
public sealed class PortalConfig
{
    /// <summary>Folder holding GGUF models. Relative paths resolve against the portal's home folder.</summary>
    public string ModelsDir { get; set; } = "llm_models";

    public LlamaServerConfig LlamaServer { get; set; } = new();
    public ControlApiConfig ControlApi { get; set; } = new();

    /// <summary>Other computers that run llama.cpp RPC workers, reached over ssh.</summary>
    public List<HostConfig> Hosts { get; set; } = new();

    /// <summary>
    /// Every place the model may be put, in the order llama.cpp should fill them. A device on the
    /// portal's own machine is local; one that names a host is served by an RPC worker on that host.
    /// </summary>
    public List<DeviceConfig> Devices { get; set; } = new();

    /// <summary>Optional helper servers (MCP bridges) the setup page can switch on for a session.</summary>
    public List<IntegrationConfig> Integrations { get; set; } = new();

    // ------------------------------------------------------------------------------------------

    public const string FileName = "duallm.config.json";

    private static readonly Regex IdPattern = new(@"^[A-Za-z0-9_-]+$", RegexOptions.Compiled);

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        UnmappedMemberHandling = System.Text.Json.Serialization.JsonUnmappedMemberHandling.Disallow,
        WriteIndented = true,
    };

    public HostConfig? HostOf(DeviceConfig device)
        => device.IsLocal ? null : Hosts.FirstOrDefault(h => h.Id == device.Host);

    public DeviceConfig? FindDevice(string id)
        => Devices.FirstOrDefault(d => string.Equals(d.Id, id, StringComparison.Ordinal));

    /// <summary>Resolves a config path: absolute paths are kept, relative ones hang off the home folder.</summary>
    public static string ResolvePath(string home, string path)
        => Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(home, path));

    /// <summary>
    /// Loads and validates the config at <paramref name="path"/>. Returns a usable config in every
    /// case: when the file is missing a sensible single-machine one is written and used, and when it
    /// is broken the built-in default is used and <paramref name="problems"/> says why, so the setup
    /// page can show the reason instead of the app refusing to start.
    /// </summary>
    public static PortalConfig Load(string path, out List<string> problems)
    {
        problems = new List<string>();

        if (!File.Exists(path))
        {
            var created = CreateDefault();
            try
            {
                File.WriteAllText(path, JsonSerializer.Serialize(created, JsonOptions));
                problems.Add($"No {FileName} was found, so a starting one was created at {path}. " +
                             "Edit it to add other machines, then press Reload config.");
            }
            catch (Exception ex)
            {
                problems.Add($"No {FileName} found and one could not be written ({ex.Message}); using built-in defaults.");
            }
            return created;
        }

        PortalConfig? config;
        try
        {
            config = JsonSerializer.Deserialize<PortalConfig>(File.ReadAllText(path), JsonOptions);
        }
        catch (Exception ex)
        {
            problems.Add($"{FileName} could not be read: {ex.Message} Using built-in defaults instead.");
            return CreateDefault();
        }

        if (config == null)
        {
            problems.Add($"{FileName} is empty. Using built-in defaults instead.");
            return CreateDefault();
        }

        var errors = config.Validate();
        if (errors.Count > 0)
        {
            problems.AddRange(errors.Select(e => $"{FileName}: {e}"));
            problems.Add("Using built-in defaults instead until the file is fixed.");
            return CreateDefault();
        }

        return config;
    }

    /// <summary>
    /// A working single-machine config: one local device per NVIDIA GPU nvidia-smi reports (or a
    /// placeholder when it reports none) plus this PC's RAM as the overflow device.
    /// </summary>
    public static PortalConfig CreateDefault()
    {
        var config = new PortalConfig();

        foreach (var (index, name, vramGb) in ProbeLocalGpus())
        {
            config.Devices.Add(new DeviceConfig
            {
                Id = $"gpu{index}",
                Type = "gpu",
                Label = name,
                CudaIndex = index,
                VramGb = vramGb,
                Default = true,
            });
        }

        if (config.Devices.Count == 0)
        {
            config.Devices.Add(new DeviceConfig
            {
                Id = "gpu0", Type = "gpu", Label = "GPU 0", CudaIndex = 0, VramGb = 8, Default = true,
            });
        }

        config.Devices.Add(new DeviceConfig
        {
            Id = "ram", Type = "ram", Label = "this PC's RAM", Chip = "RAM",
        });

        return config;
    }

    /// <summary>Asks nvidia-smi which GPUs this PC has. Empty when it is missing or fails.</summary>
    private static List<(int Index, string Name, double VramGb)> ProbeLocalGpus()
    {
        var gpus = new List<(int, string, double)>();
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=index,name,memory.total --format=csv,noheader,nounits")
            {
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            using var process = Process.Start(psi);
            if (process == null) return gpus;
            var output = process.StandardOutput.ReadToEnd();
            if (!process.WaitForExit(5000)) { try { process.Kill(); } catch { } return gpus; }
            if (process.ExitCode != 0) return gpus;

            foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                var parts = line.Split(',', 3);
                if (parts.Length == 3 &&
                    int.TryParse(parts[0].Trim(), out var index) &&
                    double.TryParse(parts[2].Trim(), System.Globalization.NumberStyles.Float,
                                    System.Globalization.CultureInfo.InvariantCulture, out var mib))
                {
                    gpus.Add((index, parts[1].Trim(), Math.Round(mib / 1024.0)));
                }
            }
        }
        catch
        {
            // No NVIDIA driver on this PC: the caller falls back to a placeholder.
        }
        return gpus;
    }

    /// <summary>Returns every reason this config cannot be used; empty when it is fine.</summary>
    public List<string> Validate()
    {
        var errors = new List<string>();

        if (string.IsNullOrWhiteSpace(ModelsDir)) errors.Add("modelsDir must not be empty.");
        if (string.IsNullOrWhiteSpace(LlamaServer.Path)) errors.Add("llamaServer.path must not be empty.");
        if (string.IsNullOrWhiteSpace(LlamaServer.FitParamsPath)) errors.Add("llamaServer.fitParamsPath must not be empty.");
        if (string.IsNullOrWhiteSpace(LlamaServer.KvCacheType)) errors.Add("llamaServer.kvCacheType must not be empty.");
        if (string.IsNullOrWhiteSpace(LlamaServer.BindAddress)) errors.Add("llamaServer.bindAddress must not be empty.");
        CheckPort(errors, "llamaServer.port", LlamaServer.Port);
        CheckPort(errors, "controlApi.port", ControlApi.Port);

        var hostIds = new HashSet<string>();
        foreach (var host in Hosts)
        {
            var where = $"hosts[{host.Id}]";
            if (!IdPattern.IsMatch(host.Id ?? "") || host.Id == "local")
                errors.Add($"{where}: id must be letters, digits, '_' or '-', and not \"local\".");
            else if (!hostIds.Add(host.Id!))
                errors.Add($"{where}: duplicate host id.");
            if (string.IsNullOrWhiteSpace(host.Ssh)) errors.Add($"{where}: ssh (the ssh destination) is required.");
            if (string.IsNullOrWhiteSpace(host.Address)) errors.Add($"{where}: address (how this PC reaches the host) is required.");
            if (string.IsNullOrWhiteSpace(host.RpcServerPath)) errors.Add($"{where}: rpcServerPath is required.");
            if (!host.IsWindows && !host.IsLinux)
                errors.Add($"{where}: os must be \"linux\" or \"windows\" (got \"{host.Os}\").");
        }

        var deviceIds = new HashSet<string>();
        var endpoints = new HashSet<string>();
        var gpuSlots = new HashSet<string>();
        var localRam = 0;
        foreach (var device in Devices)
        {
            var where = $"devices[{device.Id}]";
            if (!IdPattern.IsMatch(device.Id ?? ""))
                errors.Add($"{where}: id must be letters, digits, '_' or '-'.");
            else if (!deviceIds.Add(device.Id!))
                errors.Add($"{where}: duplicate device id.");

            if (device.Type != "gpu" && device.Type != "ram")
                errors.Add($"{where}: type must be \"gpu\" or \"ram\".");
            if (string.IsNullOrWhiteSpace(device.Label))
                errors.Add($"{where}: label is required.");
            if (device.IsGpu && device.CudaIndex < 0)
                errors.Add($"{where}: cudaIndex must be 0 or more.");
            if (device.IsGpu && !gpuSlots.Add($"{(device.IsLocal ? "local" : device.Host)}#{device.CudaIndex}"))
                errors.Add($"{where}: another gpu device already uses CUDA index {device.CudaIndex} on the same machine.");

            if (device.IsLocal)
            {
                if (device.IsRam) localRam++;
            }
            else
            {
                var host = Hosts.FirstOrDefault(h => h.Id == device.Host);
                if (host == null)
                {
                    errors.Add($"{where}: host \"{device.Host}\" is not defined under hosts.");
                    continue;
                }
                CheckPort(errors, $"{where}.port", device.Port);
                if (!endpoints.Add($"{host.Address}:{device.Port}"))
                    errors.Add($"{where}: {host.Address}:{device.Port} is already used by another device.");
            }
        }

        if (Devices.Count == 0) errors.Add("devices must list at least one device.");
        if (localRam > 1) errors.Add("only one local \"ram\" device is supported.");

        var integrationIds = new HashSet<string>();
        foreach (var integration in Integrations)
        {
            var where = $"integrations[{integration.Id}]";
            if (!IdPattern.IsMatch(integration.Id ?? ""))
                errors.Add($"{where}: id must be letters, digits, '_' or '-'.");
            else if (!integrationIds.Add(integration.Id!))
                errors.Add($"{where}: duplicate integration id.");
            if (string.IsNullOrWhiteSpace(integration.Command)) errors.Add($"{where}: command is required.");
        }

        return errors;
    }

    private static void CheckPort(List<string> errors, string name, int port)
    {
        if (port < 1 || port > 65535) errors.Add($"{name} must be between 1 and 65535 (got {port}).");
    }
}

public sealed class LlamaServerConfig
{
    /// <summary>llama-server executable on this PC.</summary>
    public string Path { get; set; } = @"llama.cpp\build\bin\Release\llama-server.exe";

    /// <summary>
    /// llama-fit-params executable, built from the same llama.cpp tree as llama-server. It is how the
    /// setup page works out which context sizes fit; without it the context list is entered by hand.
    /// </summary>
    public string FitParamsPath { get; set; } = @"llama.cpp\build\bin\Release\llama-fit-params.exe";

    public int Port { get; set; } = 8080;

    /// <summary>
    /// Interface llama-server listens on. 127.0.0.1 keeps the chat on this PC; 0.0.0.0 makes it
    /// reachable from the network (put a firewall rule in front of it if you do that).
    /// </summary>
    public string BindAddress { get; set; } = "127.0.0.1";

    /// <summary>Applied to both the K and V cache. q8_0 roughly halves them versus f16.</summary>
    public string KvCacheType { get; set; } = "q8_0";

    /// <summary>
    /// CUDA toolkit folder to expose to llama-server (CUDA_PATH and PATH). Leave null to inherit
    /// this PC's own CUDA_PATH, which is right when the toolkit is installed normally.
    /// </summary>
    public string? CudaPath { get; set; }

    /// <summary>Extra command-line text appended to every llama-server launch.</summary>
    public string ExtraArgs { get; set; } = "";
}

public sealed class ControlApiConfig
{
    public bool Enabled { get; set; } = true;

    /// <summary>Loopback port for the HTTP control API (see docs/CONTROL_API.md).</summary>
    public int Port { get; set; } = 8091;
}

public sealed class HostConfig
{
    public string Id { get; set; } = "";

    /// <summary>Name shown in the UI. Defaults to the id.</summary>
    public string? Label { get; set; }

    /// <summary>ssh destination: an alias from ~/.ssh/config or user@hostname. Key-based login is required.</summary>
    public string Ssh { get; set; } = "";

    /// <summary>
    /// The host's operating system: "linux" (the default) or "windows". It decides which commands the
    /// portal sends over ssh to start, stop and measure workers. A Windows host needs the OpenSSH Server
    /// feature; the portal talks to it through Windows PowerShell, whichever shell sshd is set to use.
    /// </summary>
    public string Os { get; set; } = "linux";

    /// <summary>Extra arguments passed to every ssh invocation, e.g. ["-i", "C:\\keys\\worker", "-p", "2222"].</summary>
    public string[] SshOptions { get; set; } = Array.Empty<string>();

    /// <summary>Address this PC uses to reach the host's RPC ports (IP or DNS name).</summary>
    public string Address { get; set; } = "";

    /// <summary>
    /// Absolute path of llama.cpp's ggml-rpc-server on the host. It must come from the same llama.cpp
    /// commit, and the same GGML_MAX_NAME, as this PC's llama-server.
    /// </summary>
    public string RpcServerPath { get; set; } = "";

    /// <summary>Interface the host's workers listen on. They are unauthenticated: keep this on a trusted network.</summary>
    public string RpcBindAddress { get; set; } = "0.0.0.0";

    /// <summary>
    /// Folder where the host's workers keep their tensor cache (created by rpc-server -c). Setting it,
    /// together with <see cref="CacheVolume"/>, adds a cache size / clear control to the setup page.
    /// </summary>
    public string? CacheDir { get; set; }

    /// <summary>A path on the volume that holds the cache, used for the "free space" reading.</summary>
    public string? CacheVolume { get; set; }

    /// <summary>How long to wait for a freshly started worker to accept connections.</summary>
    public int StartTimeoutSeconds { get; set; } = 30;

    public string DisplayName => string.IsNullOrWhiteSpace(Label) ? Id : Label!;
    public bool HasCache => !string.IsNullOrWhiteSpace(CacheDir) && !string.IsNullOrWhiteSpace(CacheVolume);

    [System.Text.Json.Serialization.JsonIgnore]
    public bool IsWindows => string.Equals(Os?.Trim(), "windows", StringComparison.OrdinalIgnoreCase);

    [System.Text.Json.Serialization.JsonIgnore]
    public bool IsLinux => string.IsNullOrWhiteSpace(Os) || string.Equals(Os.Trim(), "linux", StringComparison.OrdinalIgnoreCase);
}

public sealed class DeviceConfig
{
    public string Id { get; set; } = "";

    /// <summary>"gpu" (a CUDA card) or "ram" (system memory used as an overflow device).</summary>
    public string Type { get; set; } = "gpu";

    /// <summary>Id of the host that serves this device. Omit (or "local") for a device on this PC.</summary>
    public string? Host { get; set; }

    /// <summary>Full name, used in sentences ("RTX 3070", "the server's RAM").</summary>
    public string Label { get; set; } = "";

    /// <summary>Short text for the toggle chip. Defaults to the label.</summary>
    public string? Chip { get; set; }

    /// <summary>
    /// Nominal VRAM in GB. Only the setup page's quick "will it fit" bar uses it; the real context
    /// limit is measured from the card's free memory at the time.
    /// </summary>
    public double VramGb { get; set; } = 8;

    /// <summary>CUDA device index on its machine (gpu devices).</summary>
    public int CudaIndex { get; set; }

    /// <summary>Port of the RPC worker for this device (remote devices).</summary>
    public int Port { get; set; }

    /// <summary>Headroom to leave on a gpu on top of what llama.cpp projects, in MiB. Default depends on locality.</summary>
    public double? MarginMib { get; set; }

    /// <summary>Threads for a remote "ram" worker's CPU.</summary>
    public int Threads { get; set; } = 4;

    /// <summary>Extra rpc-server arguments for a remote worker, e.g. "-c" to cache tensors on disk.</summary>
    public string ExtraArgs { get; set; } = "";

    /// <summary>Whether the toggle starts ticked.</summary>
    public bool Default { get; set; }

    [System.Text.Json.Serialization.JsonIgnore] public bool IsLocal => string.IsNullOrWhiteSpace(Host) || Host == "local";
    [System.Text.Json.Serialization.JsonIgnore] public bool IsGpu => Type == "gpu";
    [System.Text.Json.Serialization.JsonIgnore] public bool IsRam => Type == "ram";
    [System.Text.Json.Serialization.JsonIgnore] public string ChipText => string.IsNullOrWhiteSpace(Chip) ? Label : Chip!;

    /// <summary>
    /// The cushion on this device. A local card usually also drives the desktop, so its free memory
    /// moves between the estimate and the load; a remote card only runs a worker.
    /// </summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public double EffectiveMarginMib => MarginMib ?? (IsLocal ? 768 : 512);
}

public sealed class IntegrationConfig
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";

    /// <summary>Program to start. Must be on PATH or an absolute path. "{home}" expands to the portal's folder.</summary>
    public string Command { get; set; } = "";

    public string[] Args { get; set; } = Array.Empty<string>();

    /// <summary>
    /// MCP endpoint URL registered in llama.cpp's web UI while this integration is on, e.g.
    /// "http://127.0.0.1:9191/". Omit to start the process without registering it.
    /// </summary>
    public string? Url { get; set; }

    /// <summary>Process name to also kill on stop, for launchers that spawn a differently named child.</summary>
    public string? ProcessName { get; set; }

    [System.Text.Json.Serialization.JsonIgnore]
    public string DisplayName => string.IsNullOrWhiteSpace(Name) ? Id : Name;
}
