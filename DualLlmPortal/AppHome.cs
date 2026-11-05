using System.IO;

namespace DualLlmPortal;

/// <summary>
/// The portal's home folder: where its config, logs and (by default) llama.cpp and the models live.
///
/// It is the folder the running exe sits in, so moving the whole folder just works - nothing is
/// compiled in as an absolute path. Environment.ProcessPath is the real .exe on disk even for a
/// single-file publish (AppContext.BaseDirectory can be a temp-extract directory). DUALLM_HOME
/// overrides it, which is mainly useful when running from a build output folder.
/// </summary>
internal static class AppHome
{
    public static string Directory { get; } = Resolve();

    /// <summary>The config file. DUALLM_CONFIG overrides the default location inside the home folder.</summary>
    public static string ConfigPath { get; } = ResolveConfigPath();

    private static string Resolve()
    {
        var overridePath = Environment.GetEnvironmentVariable("DUALLM_HOME");
        if (!string.IsNullOrWhiteSpace(overridePath) && System.IO.Directory.Exists(overridePath))
        {
            return Path.GetFullPath(overridePath);
        }

        return Path.GetDirectoryName(Environment.ProcessPath) ?? AppContext.BaseDirectory;
    }

    private static string ResolveConfigPath()
    {
        var overridePath = Environment.GetEnvironmentVariable("DUALLM_CONFIG");
        return !string.IsNullOrWhiteSpace(overridePath)
            ? Path.GetFullPath(overridePath)
            : Path.Combine(Directory, PortalConfig.FileName);
    }
}
