using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;

namespace DualLlmPortal;

/// <summary>Process and network helpers shared by the local and remote halves of the portal.</summary>
internal static class Shell
{
    /// <summary>
    /// An ssh invocation for <paramref name="remoteCommand"/> on <paramref name="host"/>. BatchMode is
    /// always on: the portal has no console, so a password or host-key prompt could only hang it.
    /// That is why key-based login, and one manual ssh to trust the host key, are prerequisites.
    /// </summary>
    public static ProcessStartInfo Ssh(HostConfig host, string remoteCommand)
    {
        var psi = new ProcessStartInfo("ssh")
        {
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        psi.ArgumentList.Add("-o");
        psi.ArgumentList.Add("BatchMode=yes");
        psi.ArgumentList.Add("-o");
        psi.ArgumentList.Add("ConnectTimeout=8");
        foreach (var option in host.SshOptions)
        {
            psi.ArgumentList.Add(option);
        }
        psi.ArgumentList.Add(host.Ssh);
        psi.ArgumentList.Add(remoteCommand);
        return psi;
    }

    private static readonly Regex SafeShellWord = new(@"^[A-Za-z0-9_./~@%+=:,-]+$", RegexOptions.Compiled);

    /// <summary>Quotes a word for a POSIX shell, leaving plain paths (including a leading ~) untouched.</summary>
    public static string PosixQuote(string word)
        => SafeShellWord.IsMatch(word) ? word : "'" + word.Replace("'", "'\\''") + "'";

    /// <summary>Quotes a string as a PowerShell literal: single quotes, with any ' doubled.</summary>
    public static string PowerShellQuote(string word) => "'" + word.Replace("'", "''") + "'";

    /// <summary>
    /// Wraps a PowerShell script as one command line for a Windows ssh host. The script travels as
    /// -EncodedCommand (base64 of UTF-16), which contains no quotes or shell metacharacters, so it reaches
    /// PowerShell intact whether the host's sshd hands commands to cmd.exe (the default) or to PowerShell.
    /// Progress output is switched off because, with redirected output, PowerShell would otherwise write
    /// it to stderr as CLIXML.
    /// </summary>
    public static string PowerShellCommand(string script)
    {
        var full = "$ProgressPreference = 'SilentlyContinue'; " + script;
        return "powershell -NoLogo -NoProfile -NonInteractive -EncodedCommand " +
               Convert.ToBase64String(Encoding.Unicode.GetBytes(full));
    }

    /// <summary>
    /// Runs a program and returns its stdout, or null if it could not run, timed out or exited
    /// non-zero. Never throws: every caller treats "no answer" as a normal outcome.
    /// </summary>
    public static async Task<string?> RunCaptureAsync(ProcessStartInfo psi, int timeoutMs = 15000)
    {
        try
        {
            psi.CreateNoWindow = true;
            psi.UseShellExecute = false;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;

            using var proc = new Process { StartInfo = psi };
            var stdout = new StringBuilder();

            proc.OutputDataReceived += (s, ev) =>
            {
                if (ev.Data != null)
                {
                    lock (stdout) { stdout.AppendLine(ev.Data); }
                }
            };
            proc.ErrorDataReceived += (s, ev) => { };

            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();

            var exited = await Task.Run(() => proc.WaitForExit(timeoutMs));
            if (!exited)
            {
                try { proc.Kill(true); } catch { }
                return null;
            }

            if (proc.ExitCode != 0)
            {
                return null;
            }

            lock (stdout) { return stdout.ToString(); }
        }
        catch
        {
            return null;
        }
    }

    public static Task<string?> RunCaptureAsync(string fileName, string arguments, int timeoutMs = 15000)
        => RunCaptureAsync(new ProcessStartInfo(fileName, arguments), timeoutMs);

    /// <summary>Runs a command on a host over ssh and returns stdout, or null if it could not run.</summary>
    public static Task<string?> RunSshCaptureAsync(HostConfig host, string remoteCommand, int timeoutMs = 15000)
        => RunCaptureAsync(Ssh(host, remoteCommand), timeoutMs);

    /// <summary>
    /// TCP-connects to an endpoint. llama-server does not necessarily fail hard when an --rpc
    /// endpoint is unreachable, so an explicit probe is the difference between "told you the worker is
    /// down" and "silently ran local-only while the UI claimed otherwise".
    /// </summary>
    public static async Task<bool> IsReachableAsync(string address, int port, int timeoutMs = 4000)
    {
        try
        {
            using var tcp = new TcpClient();
            var connect = tcp.ConnectAsync(address, port);
            var finished = await Task.WhenAny(connect, Task.Delay(timeoutMs));
            if (finished != connect)
            {
                return false;
            }

            await connect; // observe any connect exception
            return tcp.Connected;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Kills a process and its whole child tree. Never throws.</summary>
    public static void KillTree(Process? proc)
    {
        try
        {
            if (proc != null && !proc.HasExited)
            {
                proc.Kill(true);
                proc.WaitForExit(3000);
            }
        }
        catch { }
    }

    /// <summary>
    /// Environment for a process that loads llama.cpp's CUDA backend. CUDA_VISIBLE_DEVICES is how the
    /// local GPU ticks are honoured (see <see cref="LaunchPlan.CudaVisibleDevices"/>). CUDA_PATH and PATH
    /// are only touched when a toolkit folder is configured or already known, so that the backend's DLLs
    /// resolve even when the portal was started from a shell with a trimmed PATH.
    ///
    /// With every local GPU unticked llama-server logs one alarming line on startup:
    ///     E ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
    /// That is the expected result, not a failure: the CUDA backend registers with no devices, the RPC
    /// devices stay, and the run carries on.
    /// </summary>
    public static void ApplyCudaEnvironment(
        System.Collections.Specialized.StringDictionary env, LlamaServerConfig llama, string cudaVisibleDevices)
    {
        env["CUDA_VISIBLE_DEVICES"] = cudaVisibleDevices;

        var cudaPath = string.IsNullOrWhiteSpace(llama.CudaPath)
            ? Environment.GetEnvironmentVariable("CUDA_PATH")
            : llama.CudaPath;
        if (!string.IsNullOrWhiteSpace(cudaPath) && Directory.Exists(cudaPath))
        {
            env["CUDA_PATH"] = cudaPath;
            env["PATH"] = $"{env["PATH"]};{cudaPath}\\bin;{cudaPath}\\bin\\x64";
        }
    }
}
