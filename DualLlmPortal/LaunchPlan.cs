using System.Text;

namespace DualLlmPortal;

/// <summary>What the user ticked: which devices, and how many MoE expert layers go to each RAM device.</summary>
public sealed class LaunchSelection
{
    public HashSet<string> DeviceIds { get; init; } = new();

    /// <summary>RAM device id -> number of expert layers to place there. Missing means 0.</summary>
    public Dictionary<string, int> ExpertLayers { get; init; } = new();

    public int LayersFor(string deviceId) => ExpertLayers.TryGetValue(deviceId, out var n) ? Math.Max(0, n) : 0;
}

/// <summary>A device as llama.cpp will see it: the name --device takes, and the RPC endpoint behind it.</summary>
public sealed record PlannedDevice(DeviceConfig Config, string Name, string? Endpoint);

/// <summary>
/// Turns a selection of devices into the pieces of a llama-server command line, and nothing else:
/// no processes, no UI. The ordering rules below are load-bearing, so they live in one place where
/// they can be tested rather than being re-derived by every caller.
/// </summary>
public sealed class LaunchPlan
{
    // llama.cpp's own expert-tensor pattern, copied verbatim from common/common.h
    // (LLM_FFN_EXPS_REGEX) so that an offload here selects exactly the tensors -ncmoe would.
    private const string FfnExpsRegex = @"\.ffn_(up|down|gate|gate_up)_(ch|)exps";

    private readonly PortalConfig _config;
    private readonly LaunchSelection _selection;

    /// <summary>Everything ticked, in config order, including this PC's RAM.</summary>
    public IReadOnlyList<DeviceConfig> Selected { get; }

    /// <summary>The devices passed to --device, in the order llama.cpp should fill them.</summary>
    public IReadOnlyList<PlannedDevice> Devices { get; }

    /// <summary>Endpoints for --rpc. ggml numbers RPC devices RPC0, RPC1… in this order.</summary>
    public IReadOnlyList<string> RpcEndpoints { get; }

    /// <summary>
    /// Value for CUDA_VISIBLE_DEVICES. "-1" is not an index: the CUDA runtime stops enumerating at
    /// the first invalid entry, so the process sees no local CUDA device at all. That is what
    /// unticking every local GPU needs, because --device alone would still let the CUDA backend open
    /// a context on the card and hold a few hundred MiB for nothing.
    /// </summary>
    public string CudaVisibleDevices { get; }

    /// <summary>True when this PC's RAM is ticked: layers may spill onto the local CPU (-ngl auto).</summary>
    public bool UsesLocalRam { get; }

    public bool AnyLocalGpu { get; }
    public bool AnyRemote => RemoteHosts.Count > 0;
    public bool AnyRemoteGpu { get; }
    public IReadOnlyList<HostConfig> RemoteHosts { get; }

    private LaunchPlan(PortalConfig config, LaunchSelection selection)
    {
        _config = config;
        _selection = selection;

        Selected = config.Devices.Where(d => selection.DeviceIds.Contains(d.Id)).ToList();

        // Local cards are renumbered by CUDA_VISIBLE_DEVICES in ascending index order, whatever order
        // they are listed in, so the "CUDAn" a card ends up with is its rank among the ticked ones.
        var localGpus = Selected.Where(d => d.IsLocal && d.IsGpu).OrderBy(d => d.CudaIndex).ToList();
        CudaVisibleDevices = localGpus.Count == 0
            ? "-1"
            : string.Join(",", localGpus.Select(d => d.CudaIndex));
        AnyLocalGpu = localGpus.Count > 0;

        var devices = new List<PlannedDevice>();
        var endpoints = new List<string>();
        foreach (var device in Selected)
        {
            if (device.IsLocal)
            {
                if (device.IsGpu)
                {
                    devices.Add(new PlannedDevice(device, $"CUDA{localGpus.IndexOf(device)}", null));
                }
                continue;   // local RAM is not a llama.cpp device; it is -ngl auto / -ncmoe
            }

            var host = config.HostOf(device)!;
            var endpoint = $"{host.Address}:{device.Port}";
            devices.Add(new PlannedDevice(device, $"RPC{endpoints.Count}", endpoint));
            endpoints.Add(endpoint);
        }

        Devices = devices;
        RpcEndpoints = endpoints;
        UsesLocalRam = Selected.Any(d => d.IsLocal && d.IsRam);
        AnyRemoteGpu = Selected.Any(d => !d.IsLocal && d.IsGpu);
        RemoteHosts = Selected.Where(d => !d.IsLocal).Select(d => config.HostOf(d)!).Distinct().ToList();
    }

    public static LaunchPlan Build(PortalConfig config, LaunchSelection selection) => new(config, selection);

    /// <summary>
    /// Checks the selection for the mistakes that would otherwise surface only after the setup page is
    /// gone, as an opaque llama-server failure. Returns a readable message, or null when it is fine.
    /// </summary>
    public static string? Validate(PortalConfig config, LaunchSelection selection)
    {
        if (!config.Devices.Any(d => selection.DeviceIds.Contains(d.Id)))
        {
            return "No devices are ticked, so there is nowhere to put the model. Tick at least one.";
        }

        foreach (var (id, layers) in selection.ExpertLayers)
        {
            if (layers < 0)
            {
                return "Expert layers must be a whole number, or 0 to keep them on the GPUs.";
            }

            if (layers == 0) continue;

            var device = config.FindDevice(id);
            if (device == null || !device.IsRam)
            {
                return $"Expert layers can only go to a RAM device (\"{id}\" is not one).";
            }

            // Offloading to a device that was never started fails at argument parsing (remote) or is
            // silently obeyed against the user's own tick (local), so both are refused here.
            if (!selection.DeviceIds.Contains(id))
            {
                return $"Expert layers are set to go to \"{device.Label}\", but it is not ticked. " +
                       "Tick it, or set that layer count back to 0.";
            }
        }

        return null;
    }

    /// <summary>The llama-server command line, in the order llama.cpp needs it.</summary>
    public string BuildArguments(string modelPath, int contextSize, bool jinja)
    {
        var llama = _config.LlamaServer;

        // -ngl is the "this PC's RAM" box. "auto" lets llama.cpp measure free memory on each ticked
        // device, offload as many layers as fit and leave the remainder to the local CPU. 999 assigns
        // every layer to a ticked non-CPU device instead; where they cannot hold it a remote device
        // fails the load outright, while a local Windows card may quietly page the excess through
        // system RAM (WDDM oversubscription) - the same memory "auto" would have used, reached the slow
        // way. -c is explicit because llama-server otherwise falls back to a 4096-token context and
        // refuses any longer prompt.
        //
        // -ctk and -ctv are always set together. A quantised K cache with an f16 V cache is an
        // unsupported combination in the CUDA flash-attention kernels for ordinary attention models,
        // and aborts a worker with an error that only reaches this side as "Remote RPC server crashed".
        // A quantised cache needs flash attention, so -fa on is not optional either.
        var args = new StringBuilder();
        args.Append($"-m \"{modelPath}\" -ngl {(UsesLocalRam ? "auto" : "999")} -c {contextSize} -fa on ");
        args.Append($"-ctk {llama.KvCacheType} -ctv {llama.KvCacheType} ");
        args.Append($"--host {llama.BindAddress} --port {llama.Port}");

        // This block MUST stay ahead of --device and -ot. llama.cpp resolves a device name and an -ot
        // buffer type against the devices it knows at the moment it parses that argument, and the
        // remote devices only exist once --rpc has been handled. Put either first and it dies with
        // "invalid device: RPC0" listing only CPU and CUDA0 - which reads like the name is wrong when
        // the order is.
        if (RpcEndpoints.Count > 0)
        {
            args.Append($" --rpc {string.Join(",", RpcEndpoints)}");
        }

        // --device names exactly the ticked devices, so the tick boxes are the whole truth about where
        // the model can go. "none" is llama.cpp's own spelling for "offload nowhere", the honest
        // translation of ticking only this PC's RAM.
        args.Append(Devices.Count > 0
            ? $" --device {string.Join(",", Devices.Select(d => d.Name))}"
            : " --device none");

        // -ncmoe moves just the expert tensors of the first N layers to this PC's RAM, rather than
        // whole layers the way a lower -ngl would. An offloaded layer reads all of its experts per
        // token; an offloaded expert set is read only as far as the router activates, so for MoE
        // models this trades a lot of VRAM for far less bandwidth.
        var localLayers = Selected.Where(d => d.IsLocal && d.IsRam).Sum(d => _selection.LayersFor(d.Id));
        if (localLayers > 0)
        {
            args.Append($" -ncmoe {localLayers}");
        }

        // The same idea aimed at a remote RAM device. -ot has no range syntax, so one rule per layer,
        // mirroring how llama.cpp expands -ncmoe. Each block starts where the previous one stopped
        // (-ncmoe always claims layers 0..N-1), so the settings compose instead of fighting over the
        // same tensors. Batched prefill suffers more than generation here, because a large batch
        // activates most experts and loses the sparsity this relies on.
        var rules = new List<string>();
        var nextLayer = localLayers;
        foreach (var planned in Devices.Where(d => d.Config.IsRam))
        {
            var count = _selection.LayersFor(planned.Config.Id);
            // The buffer type name is "RPC0[endpoint]": the 0 is the device index within that one
            // worker, which is 0 for every worker, so only the endpoint tells them apart. (The *device*
            // name, RPCn, counts across all workers. They are not interchangeable.)
            for (var i = 0; i < count; i++)
            {
                rules.Add($"blk\\.{nextLayer + i}{FfnExpsRegex}=RPC0[{planned.Endpoint}]");
            }
            nextLayer += count;
        }
        if (rules.Count > 0)
        {
            args.Append($" -ot \"{string.Join(",", rules)}\"");
        }

        // --jinja turns on Jinja chat templating, which tool / function calling needs.
        if (jinja)
        {
            args.Append(" --jinja");
        }

        if (!string.IsNullOrWhiteSpace(llama.ExtraArgs))
        {
            args.Append(' ').Append(llama.ExtraArgs.Trim());
        }

        return args.ToString();
    }
}
