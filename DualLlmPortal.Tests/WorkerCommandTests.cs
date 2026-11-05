using DualLlmPortal;
using static DualLlmPortal.Tests.Fixtures;

namespace DualLlmPortal.Tests;

public class WorkerCommandTests
{
    private static string Command(string deviceId)
    {
        var config = TwoMachines();
        var device = config.FindDevice(deviceId)!;
        return RemoteWorkers.BuildWorkerCommand(config.HostOf(device)!, device);
    }

    [Fact]
    public void A_gpu_worker_pins_its_cuda_device_and_binds_its_port()
    {
        Assert.Equal(
            "env CUDA_VISIBLE_DEVICES=0 /opt/llama.cpp/build/bin/ggml-rpc-server -c -H 0.0.0.0 -p 50052",
            Command("gpuA"));
        Assert.Equal(
            "env CUDA_VISIBLE_DEVICES=1 /opt/llama.cpp/build/bin/ggml-rpc-server -c -H 0.0.0.0 -p 50054",
            Command("gpuB"));
    }

    [Fact]
    public void A_ram_worker_serves_the_cpu_backend_with_its_thread_count()
    {
        Assert.Equal(
            "/opt/llama.cpp/build/bin/ggml-rpc-server -d CPU -t 12 -H 0.0.0.0 -p 50053",
            Command("workerRam"));
    }

    [Fact]
    public void The_bind_address_is_configurable_per_host()
    {
        var config = TwoMachines();
        config.Hosts[0].RpcBindAddress = "10.0.0.5";
        var device = config.FindDevice("gpuA")!;

        Assert.Contains("-H 10.0.0.5 -p 50052", RemoteWorkers.BuildWorkerCommand(config.Hosts[0], device));
    }

    [Theory]
    [InlineData("/usr/bin/rpc", "/usr/bin/rpc")]
    [InlineData("~/llama.cpp/bin/rpc-server", "~/llama.cpp/bin/rpc-server")]
    [InlineData("/opt/my tools/rpc", "'/opt/my tools/rpc'")]
    [InlineData("it's", "'it'\\''s'")]
    [InlineData("a;b", "'a;b'")]
    public void Posix_quoting_leaves_plain_paths_alone_and_protects_everything_else(string input, string expected)
    {
        Assert.Equal(expected, Shell.PosixQuote(input));
    }

    [Fact]
    public void Ssh_always_runs_non_interactively_and_passes_the_command_as_one_argument()
    {
        var host = new HostConfig { Id = "h", Ssh = "me@box", SshOptions = new[] { "-p", "2222" } };

        var psi = Shell.Ssh(host, "echo hello world");

        Assert.Equal("ssh", psi.FileName);
        Assert.Equal(
            new[] { "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", "2222", "me@box", "echo hello world" },
            psi.ArgumentList);
    }
}
