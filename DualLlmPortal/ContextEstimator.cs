using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace DualLlmPortal
{
    /// <summary>One device the model may be placed on, as llama.cpp names it.</summary>
    public sealed class EstimatorDevice
    {
        /// <summary>llama.cpp's device name: "CUDA0", "RPC0", "RPC1"…</summary>
        public string Name { get; init; } = "";

        /// <summary>How the device is described to the user, e.g. "RTX 3070".</summary>
        public string Label { get; init; } = "";

        /// <summary>
        /// True for a system-RAM device. It stays in --device so the estimate places weights as the run
        /// will, but it is not a card that can run out of memory in the same way, so it is never the
        /// limiting device.
        /// </summary>
        public bool IsRam { get; init; }

        /// <summary>Free VRAM in MiB, measured just before the estimate. -1 when it could not be read.</summary>
        public double FreeMib { get; set; } = -1;

        /// <summary>
        /// Headroom to leave on this device on top of what llama.cpp projects, in MiB.
        ///
        /// This is not slack for a bad projection -- the projection is good. Measured on a 15.46 GB
        /// model at 76,288 tokens: llama.cpp was told 9,986 MiB and took 9,936. The margin is for the
        /// *other* things on the card. Over the same run, desktop applications grew by 284 MiB between
        /// the estimate and the load, which is why the card driving the display gets a bigger cushion
        /// than a remote one that only ever runs a worker.
        /// </summary>
        public double MarginMib { get; init; } = 512;
    }

    public sealed class ContextEstimate
    {
        /// <summary>Largest context that fits in VRAM on every selected device, 0 when unknown.</summary>
        public int MaxContext { get; init; }

        /// <summary>The device that ran out first, for the explanation line.</summary>
        public string LimitingDevice { get; init; } = "";

        /// <summary>Cost of one token of context, summed over the devices, in KiB. For the caption.</summary>
        public double KibPerToken { get; init; }

        public string? Error { get; init; }
    }

    /// <summary>
    /// Works out how much context actually fits, by asking llama.cpp rather than reimplementing its
    /// memory model.
    ///
    /// llama-fit-params -fitp on prints, per device, the MiB it estimates for (model, context,
    /// compute) at a given -c. Everything but the weights is linear in the context length -- measured:
    /// a 9B at -c 4096/8192/16384/65536 reports 118/186/322/1138 MiB of context,
    /// exactly 0.0166 MiB per token with a fixed 50 MiB base. So two probes at two context sizes pin
    /// down a straight line per device, and the largest context that fits solves in closed form. That
    /// beats reimplementing the sizing in C#, which would have to special-case MLA (no V cache at all)
    /// and sliding-window attention, and would silently drift from llama.cpp on the next update.
    ///
    /// The compute buffer is folded into the same line rather than treated as fixed: it grows with
    /// context too on some devices (RPC0 went 92 -> 376 MiB over the same sweep).
    /// </summary>
    public static class ContextEstimator
    {
        // The two probe points. Far enough apart that the per-MiB rounding in the tool's output does
        // not distort the slope, and both cheap -- neither allocates anything, they only estimate.
        private const int ProbeLow = 4096;
        private const int ProbeHigh = 65536;

        // Contexts offered when they fit. Values people actually use, rather than a smooth sweep.
        private static readonly int[] Ladder =
        {
            2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304,
            131072, 196608, 262144, 393216, 524288, 786432, 1048576,
        };

        private static readonly Regex FitLineRegex =
            new(@"^\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", RegexOptions.Compiled);

        /// <summary>
        /// Builds the list to show, given a ceiling and the model's own training context.
        /// Always ends with the ceiling itself, so the largest usable value is one click away even
        /// when it falls between ladder steps.
        /// </summary>
        public static List<int> BuildOptions(int ceiling, uint trainingContext)
        {
            if (trainingContext > 0 && trainingContext < ceiling)
            {
                // Past the training context llama.cpp still runs but the model degrades, so there is
                // no reason to offer it -- the VRAM is better spent on nothing at all.
                ceiling = (int)trainingContext;
            }

            var options = Ladder.Where(v => v <= ceiling).ToList();

            // Round down to a multiple of 256: llama.cpp pads the KV cache and a ragged number here
            // reads like false precision.
            var rounded = ceiling / 256 * 256;
            if (rounded > 0 && !options.Contains(rounded))
            {
                options.Add(rounded);
            }

            options.Sort();
            return options;
        }

        /// <summary>
        /// Runs the two probes and solves for the largest context that fits every device.
        /// <paramref name="devices"/> must carry FreeMib already measured.
        /// </summary>
        public static async Task<ContextEstimate> EstimateAsync(
            string fitParamsExe,
            string modelPath,
            IReadOnlyList<EstimatorDevice> devices,
            IReadOnlyList<string> rpcEndpoints,
            LlamaServerConfig llama,
            string cudaVisibleDevices,
            CancellationToken cancel)
        {
            if (!File.Exists(fitParamsExe))
            {
                return new ContextEstimate { Error = $"llama-fit-params.exe not found at {fitParamsExe}" };
            }

            // No GPU-like device at all: nothing to run out of, so the model's own limit is the limit.
            if (devices.Count == 0)
            {
                return new ContextEstimate { MaxContext = int.MaxValue };
            }

            var low = await RunProbeAsync(fitParamsExe, modelPath, devices, rpcEndpoints, llama,
                                          cudaVisibleDevices, ProbeLow, cancel);
            if (low == null)
            {
                return new ContextEstimate { Error = "llama-fit-params did not report any device memory" };
            }

            var high = await RunProbeAsync(fitParamsExe, modelPath, devices, rpcEndpoints, llama,
                                           cudaVisibleDevices, ProbeHigh, cancel);
            if (high == null)
            {
                return new ContextEstimate { Error = "llama-fit-params did not report any device memory" };
            }

            double maxContext = double.MaxValue;
            var limiting = "";
            double totalSlope = 0;

            foreach (var device in devices)
            {
                if (!low.TryGetValue(device.Name, out var atLow) ||
                    !high.TryGetValue(device.Name, out var atHigh))
                {
                    // The tool did not mention this device, which means llama.cpp put nothing on it.
                    // It cannot be the constraint, so it drops out of the minimum.
                    continue;
                }

                // MiB per token, from the two probes. Folding compute in with the context buffer is
                // deliberate: both grow with -c.
                var slope = (atHigh - atLow) / (double)(ProbeHigh - ProbeLow);
                totalSlope += Math.Max(0, slope);

                if (device.FreeMib < 0)
                {
                    continue;   // could not read this card; do not invent a limit for it
                }

                var budget = device.FreeMib - device.MarginMib;
                var fixedPart = atLow - slope * ProbeLow;
                var limit = slope <= 0
                    ? (budget >= fixedPart ? double.MaxValue : 0)
                    : (budget - fixedPart) / slope;

                if (limit < maxContext)
                {
                    maxContext = limit;
                    limiting = device.Label;
                }
            }

            if (maxContext == double.MaxValue)
            {
                return new ContextEstimate { MaxContext = int.MaxValue, KibPerToken = totalSlope * 1024 };
            }

            return new ContextEstimate
            {
                MaxContext = (int)Math.Max(0, Math.Min(int.MaxValue, Math.Floor(maxContext))),
                LimitingDevice = limiting,
                KibPerToken = totalSlope * 1024,
            };
        }

        /// <summary>
        /// One -fitp run. Returns MiB per device summed over (model, context, compute), or null if
        /// the tool produced nothing usable.
        /// </summary>
        private static async Task<Dictionary<string, double>?> RunProbeAsync(
            string fitParamsExe,
            string modelPath,
            IReadOnlyList<EstimatorDevice> devices,
            IReadOnlyList<string> rpcEndpoints,
            LlamaServerConfig llama,
            string cudaVisibleDevices,
            int contextSize,
            CancellationToken cancel)
        {
            // Mirrors the launch command line so the estimate describes the run that will follow:
            // same KV quantisation, same flash attention, same device set, same ordering. -ngl 999 is
            // used regardless of the RAM-overflow box, because the question being answered here is
            // what fits in VRAM.
            var args = new StringBuilder();
            args.Append($"-m \"{modelPath}\" -c {contextSize} -ngl 999 -fa on ");
            args.Append($"-ctk {llama.KvCacheType} -ctv {llama.KvCacheType} -fitp on");
            if (rpcEndpoints.Count > 0)
            {
                args.Append($" --rpc {string.Join(",", rpcEndpoints)}");
            }
            args.Append($" --device {string.Join(",", devices.Select(d => d.Name))}");

            var startInfo = new ProcessStartInfo
            {
                FileName = fitParamsExe,
                Arguments = args.ToString(),
                WorkingDirectory = Path.GetDirectoryName(fitParamsExe) ?? "",
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            // Same reasoning as the launch path: a local card that is unticked has to be invisible here
            // too, or the estimate would be taken against a device the run will not have.
            Shell.ApplyCudaEnvironment(startInfo.EnvironmentVariables, llama, cudaVisibleDevices);

            var stdout = new StringBuilder();
            using var process = new Process { StartInfo = startInfo };
            process.OutputDataReceived += (_, e) =>
            {
                if (e.Data != null)
                {
                    lock (stdout) { stdout.AppendLine(e.Data); }
                }
            };
            process.ErrorDataReceived += (_, __) => { };

            try
            {
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                await process.WaitForExitAsync(cancel);
            }
            catch (OperationCanceledException)
            {
                try { process.Kill(true); } catch { }
                throw;
            }
            catch
            {
                return null;
            }

            string text;
            lock (stdout) { text = stdout.ToString(); }

            // "CUDA0 4812 186 501" -- device, model, context, compute. The trailing "Host" line is the
            // CPU device and is not a VRAM constraint.
            var totals = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
            foreach (var line in text.Split('\n'))
            {
                var m = FitLineRegex.Match(line.TrimEnd('\r'));
                if (!m.Success || m.Groups[1].Value.Equals("Host", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                var sum = 0.0;
                for (var g = 2; g <= 4; g++)
                {
                    sum += double.Parse(m.Groups[g].Value, CultureInfo.InvariantCulture);
                }
                totals[m.Groups[1].Value] = sum;
            }

            return totals.Count > 0 ? totals : null;
        }
    }
}
