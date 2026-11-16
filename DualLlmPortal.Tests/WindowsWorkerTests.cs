using System.Text;
using DualLlmPortal;
using static DualLlmPortal.Tests.Fixtures;

namespace DualLlmPortal.Tests;

/// <summary>Workers on a Windows host: commands go to PowerShell, encoded so no shell can mangle them.</summary>
public class WindowsWorkerTests
{
    private const string Prefix = "powershell -NoLogo -NoProfile -NonInteractive -EncodedCommand ";

    /// <summary>The PowerShell script inside an encoded command, as the host will run it.</summary>
    private static string Decode(string command)
    {
        Assert.StartsWith(Prefix, command);
        return Encoding.Unicode.GetString(Convert.FromBase64String(command[Prefix.Length..]));
    }

    private static (PortalConfig Config, HostConfig Laptop) WithLaptop()
    {
        var config = TwoMachines();
        var laptop = new HostConfig
        {
            Id = "laptop", Ssh = "me@laptop", Address = "192.0.2.80", Os = "windows",
            RpcServerPath = @"C:\Users\me\duallm\llama.cpp\build\bin\Release\ggml-rpc-server.exe",
        };
        config.Hosts.Add(laptop);
        config.Devices.Add(new DeviceConfig { Id = "laptopGpu", Type = "gpu", Host = "laptop", Label = "Laptop RTX 3050", CudaIndex = 0, Port = 50052, ExtraArgs = "-c" });
        config.Devices.Add(new DeviceConfig { Id = "laptopRam", Type = "ram", Host = "laptop", Label = "laptop RAM", Port = 50053, Threads = 6 });
        Assert.Empty(config.Validate());
        return (config, laptop);
    }

    [Fact]
    public void A_gpu_worker_pins_its_card_puts_cuda_on_path_and_runs_the_exe()
    {
        var (config, laptop) = WithLaptop();

        var script = Decode(RemoteWorkers.BuildWorkerCommand(laptop, config.FindDevice("laptopGpu")!));

        Assert.Equal(
            "$ProgressPreference = 'SilentlyContinue'; " +
            "if ($env:CUDA_PATH) { $env:PATH = \"$env:CUDA_PATH\\bin;$env:CUDA_PATH\\bin\\x64;$env:PATH\" }; " +
            "$env:CUDA_VISIBLE_DEVICES = '0'; " +
            "& 'C:\\Users\\me\\duallm\\llama.cpp\\build\\bin\\Release\\ggml-rpc-server.exe' -c -H 0.0.0.0 -p 50052; " +
            "exit $LASTEXITCODE",
            script);
    }

    [Fact]
    public void A_ram_worker_serves_the_cpu_backend_without_pinning_a_card()
    {
        var (config, laptop) = WithLaptop();

        var script = Decode(RemoteWorkers.BuildWorkerCommand(laptop, config.FindDevice("laptopRam")!));

        Assert.DoesNotContain("CUDA_VISIBLE_DEVICES", script);
        Assert.Contains("ggml-rpc-server.exe' -d CPU -t 6 -H 0.0.0.0 -p 50053", script);
    }

    [Fact]
    public void The_encoded_command_has_nothing_cmd_or_powershell_would_reinterpret()
    {
        var (config, laptop) = WithLaptop();
        laptop.RpcServerPath = @"C:\Program Files\it's & mine\ggml-rpc-server.exe";

        var command = RemoteWorkers.BuildWorkerCommand(laptop, config.FindDevice("laptopGpu")!);

        Assert.Matches(@"^[A-Za-z -]+[A-Za-z0-9+/=]+$", command);
        Assert.Contains(@"& 'C:\Program Files\it''s & mine\ggml-rpc-server.exe'", Decode(command));
    }

    [Fact]
    public void Stopping_ends_processes_by_name_without_the_exe_suffix()
    {
        var (_, laptop) = WithLaptop();

        var script = Decode(RemoteWorkers.BuildKillCommand(laptop, force: false));

        Assert.Contains("Get-Process -Name 'ggml-rpc-server' -ErrorAction SilentlyContinue | Stop-Process -Force", script);
        Assert.EndsWith("exit 0", script);
        Assert.Equal(script, Decode(RemoteWorkers.BuildKillCommand(laptop, force: true)));
    }

    [Fact]
    public void Linux_hosts_keep_their_posix_commands()
    {
        var config = TwoMachines();
        var worker = config.Hosts[0];

        Assert.Equal(
            "killall -9 ggml-rpc-server 2>/dev/null || pkill -9 -x ggml-rpc-server 2>/dev/null; true",
            RemoteWorkers.BuildKillCommand(worker, force: true));
        Assert.Equal(
            "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
            RemoteWorkers.BuildNvidiaSmiCommand(worker, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"));
    }

    [Fact]
    public void Nvidia_smi_arguments_are_quoted_so_commas_stay_inside_one_argument()
    {
        var (_, laptop) = WithLaptop();

        var script = Decode(RemoteWorkers.BuildNvidiaSmiCommand(laptop, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"));

        Assert.Contains("& nvidia-smi '--query-gpu=index,memory.free' '--format=csv,noheader,nounits'; exit $LASTEXITCODE", script);
    }

    [Fact]
    public void Cache_commands_use_literal_paths()
    {
        var (_, laptop) = WithLaptop();
        laptop.CacheDir = @"C:\Users\me\AppData\Local\llama.cpp\rpc";
        laptop.CacheVolume = @"C:\";

        var stats = Decode(RemoteWorkers.BuildCacheStatsCommand(laptop));
        var clear = Decode(RemoteWorkers.BuildClearCacheCommand(laptop));

        Assert.Contains(@"$d = 'C:\Users\me\AppData\Local\llama.cpp\rpc'", stats);
        Assert.Contains("Test-Path -LiteralPath $d", stats);
        Assert.Contains("AvailableFreeSpace", stats);
        Assert.Contains("Get-ChildItem -LiteralPath $d -Recurse -File -Force", clear);
    }

    [Theory]
    [InlineData("windows", true)]
    [InlineData("Windows", true)]
    [InlineData("linux", false)]
    [InlineData("", false)]
    public void Os_is_case_insensitive_and_defaults_to_linux(string os, bool windows)
    {
        var host = new HostConfig { Os = os };

        Assert.Equal(windows, host.IsWindows);
        Assert.Equal(!windows, host.IsLinux);
    }

    [Fact]
    public void Validate_rejects_an_unknown_os()
    {
        var config = TwoMachines();
        config.Hosts[0].Os = "macos";

        Assert.Contains(config.Validate(), e => e.Contains("os must be"));
    }
}
