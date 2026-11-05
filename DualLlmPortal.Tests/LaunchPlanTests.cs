using DualLlmPortal;
using static DualLlmPortal.Tests.Fixtures;

namespace DualLlmPortal.Tests;

/// <summary>
/// Pins the llama-server command line the portal builds. The expected strings are the ones the
/// original hard-coded implementation produced for the same device combinations, so these tests are
/// what shows the generalisation changed no behaviour.
/// </summary>
public class LaunchPlanTests
{
    private const string Model = @"C:\models\m.gguf";
    private const string Common = "-fa on -ctk q8_0 -ctv q8_0 --host 0.0.0.0 --port 8080";

    private static string Args(LaunchSelection selection, bool jinja = false, int context = 32768)
        => LaunchPlan.Build(TwoMachines(), selection).BuildArguments(Model, context, jinja);

    [Fact]
    public void Local_and_one_remote_gpu_uses_999_layers_and_orders_cuda_before_rpc()
    {
        var args = Args(Select("localGpu", "gpuA"));

        Assert.Equal(
            $"-m \"{Model}\" -ngl 999 -c 32768 {Common} --rpc 192.0.2.72:50052 --device CUDA0,RPC0",
            args);
    }

    [Fact]
    public void Every_device_with_expert_offload_matches_the_original_ordering()
    {
        var layers = new Dictionary<string, int> { ["localRam"] = 4, ["workerRam"] = 2 };
        var args = Args(Select(layers, "localGpu", "localRam", "gpuA", "gpuB", "workerRam"), jinja: true);

        // -ngl auto because this PC's RAM is ticked; --rpc before --device and -ot; the worker's RAM
        // block starts at layer 4, where -ncmoe 4 stopped.
        Assert.Equal(
            $"-m \"{Model}\" -ngl auto -c 32768 {Common} " +
            "--rpc 192.0.2.72:50052,192.0.2.72:50054,192.0.2.72:50053 --device CUDA0,RPC0,RPC1,RPC2 " +
            "-ncmoe 4 " +
            "-ot \"blk\\.4\\.ffn_(up|down|gate|gate_up)_(ch|)exps=RPC0[192.0.2.72:50053]," +
            "blk\\.5\\.ffn_(up|down|gate|gate_up)_(ch|)exps=RPC0[192.0.2.72:50053]\" " +
            "--jinja",
            args);
    }

    [Fact]
    public void Only_local_ram_ticked_offloads_nowhere_and_hides_the_local_gpu()
    {
        var plan = LaunchPlan.Build(TwoMachines(), Select("localRam"));

        Assert.Equal(
            $"-m \"{Model}\" -ngl auto -c 8192 {Common} --device none",
            plan.BuildArguments(Model, 8192, jinja: false));
        Assert.Equal("-1", plan.CudaVisibleDevices);
        Assert.True(plan.UsesLocalRam);
    }

    [Fact]
    public void A_remote_gpu_alone_hides_the_local_gpu_and_is_still_RPC0()
    {
        var plan = LaunchPlan.Build(TwoMachines(), Select("gpuB"));

        Assert.Equal("-1", plan.CudaVisibleDevices);
        Assert.Contains("--rpc 192.0.2.72:50054 --device RPC0", plan.BuildArguments(Model, 32768, false));
        Assert.True(plan.AnyRemoteGpu);
        Assert.False(plan.AnyLocalGpu);
    }

    [Fact]
    public void Rpc_numbering_follows_config_order_not_tick_order()
    {
        var plan = LaunchPlan.Build(TwoMachines(), Select("workerRam", "gpuB", "gpuA"));

        Assert.Equal(new[] { "RPC0", "RPC1", "RPC2" }, plan.Devices.Select(d => d.Name));
        Assert.Equal(new[] { "gpuA", "gpuB", "workerRam" }, plan.Devices.Select(d => d.Config.Id));
    }

    [Fact]
    public void Several_local_gpus_are_renumbered_by_ascending_cuda_index()
    {
        var config = new PortalConfig
        {
            Devices =
            {
                // Listed high index first: the fill order is the list order, but CUDA numbering is not.
                new DeviceConfig { Id = "second", Type = "gpu", Label = "GPU 1", CudaIndex = 1 },
                new DeviceConfig { Id = "first", Type = "gpu", Label = "GPU 0", CudaIndex = 0 },
            },
        };

        var both = LaunchPlan.Build(config, Select("first", "second"));
        Assert.Equal("0,1", both.CudaVisibleDevices);
        Assert.Equal(new[] { "CUDA1", "CUDA0" }, both.Devices.Select(d => d.Name));

        var onlySecond = LaunchPlan.Build(config, Select("second"));
        Assert.Equal("1", onlySecond.CudaVisibleDevices);
        Assert.Equal(new[] { "CUDA0" }, onlySecond.Devices.Select(d => d.Name));
    }

    [Fact]
    public void Kv_cache_type_bind_address_port_and_extra_args_come_from_config()
    {
        var config = TwoMachines();
        config.LlamaServer.KvCacheType = "f16";
        config.LlamaServer.BindAddress = "127.0.0.1";
        config.LlamaServer.Port = 9000;
        config.LlamaServer.ExtraArgs = "  --no-webui  ";

        var args = LaunchPlan.Build(config, Select("localGpu")).BuildArguments(Model, 4096, false);

        Assert.Equal(
            $"-m \"{Model}\" -ngl 999 -c 4096 -fa on -ctk f16 -ctv f16 --host 127.0.0.1 --port 9000 --device CUDA0 --no-webui",
            args);
    }

    [Fact]
    public void Validate_refuses_an_empty_selection()
    {
        Assert.NotNull(LaunchPlan.Validate(TwoMachines(), Select()));
    }

    [Fact]
    public void Validate_refuses_expert_layers_for_a_device_that_is_not_ticked()
    {
        var layers = new Dictionary<string, int> { ["workerRam"] = 3 };

        var error = LaunchPlan.Validate(TwoMachines(), Select(layers, "localGpu"));

        Assert.NotNull(error);
        Assert.Contains("worker's RAM", error);
    }

    [Fact]
    public void Validate_refuses_expert_layers_on_a_gpu_and_negative_counts()
    {
        Assert.NotNull(LaunchPlan.Validate(TwoMachines(),
            Select(new Dictionary<string, int> { ["gpuA"] = 1 }, "gpuA")));
        Assert.NotNull(LaunchPlan.Validate(TwoMachines(),
            Select(new Dictionary<string, int> { ["localRam"] = -1 }, "localRam")));
    }

    [Fact]
    public void Validate_accepts_expert_layers_when_the_ram_device_is_ticked()
    {
        var layers = new Dictionary<string, int> { ["workerRam"] = 3, ["localRam"] = 0 };

        Assert.Null(LaunchPlan.Validate(TwoMachines(), Select(layers, "localGpu", "workerRam")));
    }
}
