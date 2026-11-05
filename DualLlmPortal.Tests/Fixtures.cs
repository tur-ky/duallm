using DualLlmPortal;

namespace DualLlmPortal.Tests;

/// <summary>
/// A two-machine setup shaped like the one the portal grew out of: a local card and local RAM, plus a
/// Linux worker host with two GPUs and a RAM device. Addresses are from the documentation range.
/// </summary>
internal static class Fixtures
{
    public static PortalConfig TwoMachines()
    {
        var config = new PortalConfig
        {
            LlamaServer = { BindAddress = "0.0.0.0" },
            Hosts =
            {
                new HostConfig
                {
                    Id = "worker", Label = "worker", Ssh = "worker", Address = "192.0.2.72",
                    RpcServerPath = "/opt/llama.cpp/build/bin/ggml-rpc-server",
                },
            },
            Devices =
            {
                new DeviceConfig { Id = "localGpu", Type = "gpu", Label = "Local GPU", CudaIndex = 0, VramGb = 12, Default = true },
                new DeviceConfig { Id = "localRam", Type = "ram", Label = "this PC's RAM", Chip = "RAM (PC)" },
                new DeviceConfig { Id = "gpuA", Type = "gpu", Host = "worker", Label = "Worker GPU A", CudaIndex = 0, Port = 50052, ExtraArgs = "-c", Default = true },
                new DeviceConfig { Id = "gpuB", Type = "gpu", Host = "worker", Label = "Worker GPU B", CudaIndex = 1, Port = 50054, ExtraArgs = "-c" },
                new DeviceConfig { Id = "workerRam", Type = "ram", Host = "worker", Label = "worker's RAM", Port = 50053, Threads = 12 },
            },
        };
        Assert.Empty(config.Validate());
        return config;
    }

    public static LaunchSelection Select(params string[] ids) => new() { DeviceIds = new HashSet<string>(ids) };

    public static LaunchSelection Select(Dictionary<string, int> layers, params string[] ids)
        => new() { DeviceIds = new HashSet<string>(ids), ExpertLayers = layers };
}
