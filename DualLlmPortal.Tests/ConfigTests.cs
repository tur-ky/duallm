using DualLlmPortal;
using static DualLlmPortal.Tests.Fixtures;

namespace DualLlmPortal.Tests;

public class ConfigTests : IDisposable
{
    private readonly string _dir = Path.Combine(AppContext.BaseDirectory, "scratch-" + Guid.NewGuid().ToString("N"));

    public ConfigTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Write(string json)
    {
        var path = Path.Combine(_dir, PortalConfig.FileName);
        File.WriteAllText(path, json);
        return path;
    }

    [Fact]
    public void Comments_and_trailing_commas_are_allowed()
    {
        var path = Write("""
            {
              // a comment
              "devices": [
                { "id": "gpu0", "type": "gpu", "label": "GPU 0", "default": true, },
              ],
            }
            """);

        var config = PortalConfig.Load(path, out var problems);

        Assert.Empty(problems);
        Assert.Equal("gpu0", Assert.Single(config.Devices).Id);
    }

    [Fact]
    public void A_misspelt_option_is_reported_rather_than_ignored()
    {
        var path = Write("""{ "devices": [ { "id": "gpu0", "type": "gpu", "label": "GPU 0", "vram": 12 } ] }""");

        var config = PortalConfig.Load(path, out var problems);

        Assert.Contains(problems, p => p.Contains("vram", StringComparison.OrdinalIgnoreCase));
        Assert.NotEmpty(config.Devices);   // fell back to a usable default
    }

    [Fact]
    public void A_missing_file_is_created_with_a_usable_default()
    {
        var path = Path.Combine(_dir, PortalConfig.FileName);

        var config = PortalConfig.Load(path, out var problems);

        Assert.True(File.Exists(path));
        Assert.NotEmpty(problems);          // tells the user the file was created
        Assert.Empty(config.Validate());
        Assert.Contains(config.Devices, d => d.IsRam && d.IsLocal);
        Assert.Contains(config.Devices, d => d.IsGpu && d.IsLocal);

        // And what was written reads back cleanly.
        var again = PortalConfig.Load(path, out var secondProblems);
        Assert.Empty(secondProblems);
        Assert.Equal(config.Devices.Select(d => d.Id), again.Devices.Select(d => d.Id));
    }

    [Fact]
    public void Broken_json_falls_back_to_the_default_with_an_explanation()
    {
        var path = Write("{ not json");

        var config = PortalConfig.Load(path, out var problems);

        Assert.NotEmpty(problems);
        Assert.NotEmpty(config.Devices);
    }

    [Fact]
    public void The_example_config_in_the_repo_is_valid()
    {
        var example = FindUp("duallm.example.json");

        var config = PortalConfig.Load(example, out var problems);

        Assert.Empty(problems);
        Assert.Empty(config.Validate());
        Assert.NotEmpty(config.Hosts);
    }

    [Fact]
    public void Relative_paths_resolve_against_the_home_folder_and_absolute_ones_are_kept()
    {
        Assert.Equal(Path.GetFullPath(Path.Combine(_dir, "llm_models")), PortalConfig.ResolvePath(_dir, "llm_models"));
        Assert.Equal(@"D:\models", PortalConfig.ResolvePath(_dir, @"D:\models"));
    }

    // ------------------------------------------------------------------ validation

    [Fact]
    public void Validate_flags_duplicate_ids_unknown_hosts_and_clashing_endpoints()
    {
        var config = TwoMachines();
        config.Devices.Add(new DeviceConfig { Id = "gpuA", Type = "gpu", Label = "dup" });
        config.Devices.Add(new DeviceConfig { Id = "ghost", Type = "gpu", Host = "nowhere", Label = "x", Port = 1 });
        config.Devices.Add(new DeviceConfig { Id = "clash", Type = "gpu", Host = "worker", Label = "c", CudaIndex = 7, Port = 50052 });

        var errors = config.Validate();

        Assert.Contains(errors, e => e.Contains("duplicate device id"));
        Assert.Contains(errors, e => e.Contains("nowhere"));
        Assert.Contains(errors, e => e.Contains("already used"));
    }

    [Fact]
    public void Validate_flags_a_second_gpu_on_the_same_cuda_index_and_a_second_local_ram()
    {
        var config = TwoMachines();
        config.Devices.Add(new DeviceConfig { Id = "same", Type = "gpu", Label = "again", CudaIndex = 0 });
        config.Devices.Add(new DeviceConfig { Id = "ram2", Type = "ram", Label = "more ram" });

        var errors = config.Validate();

        Assert.Contains(errors, e => e.Contains("CUDA index"));
        Assert.Contains(errors, e => e.Contains("only one local"));
    }

    [Fact]
    public void Validate_requires_the_fields_a_host_cannot_work_without()
    {
        var config = TwoMachines();
        config.Hosts.Add(new HostConfig { Id = "half" });

        var errors = config.Validate();

        Assert.Contains(errors, e => e.Contains("ssh"));
        Assert.Contains(errors, e => e.Contains("address"));
        Assert.Contains(errors, e => e.Contains("rpcServerPath"));
    }

    [Fact]
    public void Validate_rejects_bad_ports_types_and_ids()
    {
        var config = TwoMachines();
        config.LlamaServer.Port = 0;
        config.Devices.Add(new DeviceConfig { Id = "bad id!", Type = "tpu", Label = "x" });

        var errors = config.Validate();

        Assert.Contains(errors, e => e.Contains("llamaServer.port"));
        Assert.Contains(errors, e => e.Contains("letters, digits"));
        Assert.Contains(errors, e => e.Contains("\"gpu\" or \"ram\""));
    }

    private static string FindUp(string fileName)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, fileName);
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException($"{fileName} not found above {AppContext.BaseDirectory}");
    }
}
