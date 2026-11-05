using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Media;
using Microsoft.Web.WebView2.Core;

namespace DualLlmPortal
{
    public partial class MainWindow : Window
    {
        // ---- Configuration -------------------------------------------------------------------
        // Nothing about a particular set of machines is compiled in. Which devices exist, where
        // llama.cpp lives and which hosts run workers all come from duallm.config.json (see
        // PortalConfig.cs); paths are relative to the folder the exe sits in (see AppHome.cs).
        private PortalConfig _config = new();
        private List<string> _configProblems = new();
        private RemoteWorkers _workers = null!;

        private static string Home => AppHome.Directory;
        private string ModelsDir => PortalConfig.ResolvePath(Home, _config.ModelsDir);
        private string LocalServerExecutable => PortalConfig.ResolvePath(Home, _config.LlamaServer.Path);
        private string FitParamsExecutable => PortalConfig.ResolvePath(Home, _config.LlamaServer.FitParamsPath);
        private static string ServerLogPath => Path.Combine(Home, "portal-llama-server.log");

        /// <summary>Port the local llama-server listens on (also read by the control API's health probe).</summary>
        internal int ServerPort => _config.LlamaServer.Port;

        // ---- Live setup state ----------------------------------------------------------------
        // The setup page updates these the instant a control is touched (via web messages), so the
        // estimate and the launch both read straight from here rather than from any UI control.
        // Launch arguments name exactly the ticked set: there is no implicit "and also whatever else
        // it finds". Unticking every local GPU is what makes a remote-only run possible.
        private readonly HashSet<string> _selectedDevices = new();
        private readonly Dictionary<string, int> _expertLayers = new();
        private readonly HashSet<string> _enabledIntegrations = new();

        private string? _selectedModelPath;
        private int _setupContextSize = 32768;

        // The scanned models, kept so a selectModel/launch message can look one up by path and so a
        // re-scan (after Stop) can be re-pushed to the page. _scanStatus is the one-line summary the
        // page shows under the header.
        private List<LlmModel> _models = new();
        private string _scanStatus = "";

        // ---- Processes -----------------------------------------------------------------------
        private Process? _localServerProcess;
        private readonly Dictionary<string, Process> _integrationProcesses = new();
        private string? _chatMcpInitScriptId;
        private LaunchPlan? _activePlan;

        // Cancels an in-flight context estimate when the selection changes again underneath it.
        private CancellationTokenSource? _estimateCts;

        // ---- Control API state (listener in ControlApi.cs; contract: docs/CONTROL_API.md) ----
        // The loopback API has no UI of its own, so its launch progress is tracked here rather than
        // read off the setup page's LoadingPanel. Only written on the UI thread and serialised by
        // BuildApiStateJson for GET /control/state.
        private ControlApi? _controlApi;
        // True when spawned with --headless: no window, no WebView2, control API only.
        private readonly bool _headless;
        private string _apiPhase = "idle";              // idle | loading | ready | error
        private string _apiMessage = "Ready to launch.";
        private string? _apiError;                       // readable launch error when _apiPhase == "error"
        private string? _apiRunningModelPath;
        private string? _apiRunningModelName;
        private int _apiRunningContextSize;
        private Dictionary<string, bool>? _apiRunningDevices;      // device flags this launch actually ran with
        private Dictionary<string, int>? _apiRunningExpertLayers;
        private bool _apiReady;                          // llama-server answered /health for this launch
        private bool _apiLaunchBusy;                     // serialises launches; 409 while true
        // Identity/ownership of the launch task: only the newest launch may clear _apiLaunchBusy.
        private int _apiLaunchId;
        // Bumped by /control/stop. A launch task from an older epoch must not write ready/error
        // state over the idle state the stop installed.
        private int _apiStateEpoch;
        // Cancels the in-flight launch's checkpoints when /control/stop is called.
        private CancellationTokenSource? _apiLaunchCts;

        private Microsoft.Web.WebView2.Wpf.WebView2? ChatWebView;
        // Hosts setup.html - the configuration page.
        private Microsoft.Web.WebView2.Wpf.WebView2? SetupWebView;

        // Rolling tail of the local llama-server's stdout+stderr, used for error reporting.
        private readonly StringBuilder _localServerLog = new StringBuilder();
        private const int ServerLogMaxChars = 64 * 1024;
        private readonly object _logFileLock = new object();

        // Accumulated as the log streams, rather than scraped from _localServerLog afterwards -
        // the load_tensors lines appear early and would be trimmed out of the rolling tail.
        // VRAM the workers held before the model was loaded (their bare CUDA contexts), and after.
        // The difference is what this model actually placed on the remote GPUs. -1 = not measured.
        private double _workerVramBaselineMib = -1;
        private double _workerVramLoadedMib = -1;

        // Anything smaller than this is the workers' own CUDA contexts or measurement noise rather
        // than model weights.
        private const double WorkerWeightsThresholdMib = 256;

        // llama-server accepts far smaller values, but a context this short cannot hold a system
        // prompt plus one useful exchange, and the overflow surfaces as an opaque HTTP 500 mid-chat
        // rather than as a launch-time error. Reject it here, while there is still a dialog to show.
        private const int MinContextSize = 512;

        public MainWindow(bool headless = false)
        {
            _headless = headless;
            InitializeComponent();

            LoadConfig();

            Closing += MainWindow_Closing;

            // Safety net: ensure cleanup even if the app crashes or is killed
            AppDomain.CurrentDomain.ProcessExit += (s, ev) => StopAllProcesses();

            if (headless)
            {
                // Windowless mode for programmatic clients. No WebView2 and no window are created: the
                // setup page is not needed because the loopback control API is the only client. Scan
                // models and bind the API now; the process stays alive until POST /control/shutdown (or
                // the process is killed).
                ShowInTaskbar = false;
                Visibility = Visibility.Hidden;
                ScanModels();
                StartControlApi();
                return;
            }

            // WebView2 controls are created in code rather than XAML so the host stays a thin shell.
            ChatWebView = new Microsoft.Web.WebView2.Wpf.WebView2();
            ChatWebViewGrid.Children.Add(ChatWebView);

            // The setup page is a second WebView2. Its content (setup.html) and the JS<->C# bridge are
            // wired in MainWindow_Loaded once the CoreWebView2 environment is ready.
            SetupWebView = new Microsoft.Web.WebView2.Wpf.WebView2();
            SetupWebView.DefaultBackgroundColor = System.Drawing.Color.FromArgb(0x12, 0x12, 0x12);
            SetupWebViewGrid.Children.Add(SetupWebView);

            Loaded += MainWindow_Loaded;
        }

        /// <summary>
        /// (Re)reads the config and resets everything derived from it: the worker supervisor and the
        /// default ticks. Callers stop any running processes first, because they belong to the old config.
        /// </summary>
        private void LoadConfig()
        {
            _config = PortalConfig.Load(AppHome.ConfigPath, out _configProblems);
            _workers = new RemoteWorkers(_config);

            _selectedDevices.Clear();
            _expertLayers.Clear();
            _enabledIntegrations.Clear();

            foreach (var device in _config.Devices.Where(d => d.Default))
            {
                _selectedDevices.Add(device.Id);
            }
            if (_selectedDevices.Count == 0 && _config.Devices.Count > 0)
            {
                _selectedDevices.Add(_config.Devices[0].Id);
            }
        }

        private LaunchSelection Snapshot() => new()
        {
            DeviceIds = new HashSet<string>(_selectedDevices),
            ExpertLayers = new Dictionary<string, int>(_expertLayers),
        };

        private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                await SetupWebView!.EnsureCoreWebView2Async();

                var core = SetupWebView.CoreWebView2;
                // Nothing on this page loads remote content, so lock the host down: no dev-tools, no
                // right-click menu, no browser-managed autofill. The bridge is the only channel.
                core.Settings.AreDevToolsEnabled = false;
                core.Settings.AreDefaultContextMenusEnabled = false;
                core.Settings.IsPasswordAutosaveEnabled = false;
                core.Settings.IsGeneralAutofillEnabled = false;

                core.WebMessageReceived += SetupWebView_WebMessageReceived;

                // setup.html is embedded in the assembly and rendered with NavigateToString, so nothing
                // extra has to travel next to the exe. The page posts a "ready" message once it has
                // rendered, which is when the first model scan is pushed to it.
                core.NavigateToString(ReadEmbeddedSetupHtml());
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not initialise the setup page: {ex.Message}",
                    "Setup UI Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            // Loopback control API (ControlApi.cs). Started after the WebView2 environment is up; it has
            // no UI and must never take the portal down, so StartControlApi swallows bind failures.
            StartControlApi();
        }

        /// <summary>
        /// Loads setup.html out of the assembly's embedded resources. The resource name is
        /// "DualLlmPortal.SetupUi.setup.html" (root namespace + folder), but it is matched by suffix
        /// so a project rename does not silently break it.
        /// </summary>
        private static string ReadEmbeddedSetupHtml()
        {
            var asm = Assembly.GetExecutingAssembly();
            var name = asm.GetManifestResourceNames()
                .FirstOrDefault(n => n.EndsWith("setup.html", StringComparison.OrdinalIgnoreCase))
                ?? throw new FileNotFoundException(
                    "setup.html is not embedded in the assembly. Check <EmbeddedResource Include=\"SetupUi\\setup.html\"/> in the csproj.");

            using var stream = asm.GetManifestResourceStream(name)!;
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        // =====================================================================================
        // Models
        // =====================================================================================

        /// <summary>
        /// True for the CLIP vision projectors that sit next to multimodal models in the same folder.
        /// Covers both naming conventions in use: "mmproj-&lt;model&gt;.gguf" and "&lt;model&gt;-mmproj-&lt;type&gt;.gguf".
        /// </summary>
        private static bool IsVisionProjector(string path)
            => Path.GetFileName(path).Contains("mmproj", StringComparison.OrdinalIgnoreCase);

        private void ScanModels()
        {
            try
            {
                var modelsDir = ModelsDir;
                if (!Directory.Exists(modelsDir))
                {
                    Directory.CreateDirectory(modelsDir);
                }

                var files = Directory.GetFiles(modelsDir, "*.gguf");
                var models = new List<LlmModel>();
                var projectors = 0;

                foreach (var file in files)
                {
                    // mmproj files are the CLIP vision encoders that pair with a multimodal model via
                    // --mmproj. There is no language model inside them, so selecting one only ever
                    // produces "unsupported model architecture: 'clip'" once llama-server is already
                    // starting. Listing them invites a launch that cannot succeed.
                    //
                    // Matched on filename rather than by reading general.architecture out of the GGUF
                    // header: every projector ships with "mmproj" in its name, and a misnamed one just
                    // lands back on the same error as before, which is no worse than listing them all.
                    if (IsVisionProjector(file))
                    {
                        projectors++;
                        continue;
                    }

                    var fileInfo = new FileInfo(file);
                    double sizeGb = (double)fileInfo.Length / (1024 * 1024 * 1024);
                    var model = new LlmModel
                    {
                        Name = Path.GetFileName(file),
                        Path = file,
                        SizeGB = $"{sizeGb:F2} GB",
                        SizeGiB = sizeGb
                    };
                    // The setup page shows family / params / quant / tag badges. None of that is in the
                    // sparse GGUF metadata this app reads, so it is parsed out of the filename.
                    ParseModelCard(model);
                    models.Add(model);
                }

                // Sort by name so the list is stable between scans and easy to scan by eye.
                _models = models.OrderBy(m => m.Name, StringComparer.OrdinalIgnoreCase).ToList();

                var hidden = projectors == 0 ? "" : $" ({projectors} mmproj vision projector(s) hidden)";

                if (_models.Count == 0)
                {
                    _selectedModelPath = null;
                    _scanStatus = $"No models found in {modelsDir}. Download some GGUFs there.{hidden}";
                }
                else
                {
                    // Keep the current selection across a re-scan if it survived; otherwise fall to the
                    // first model.
                    if (_selectedModelPath == null || _models.All(m => m.Path != _selectedModelPath))
                    {
                        _selectedModelPath = _models[0].Path;
                    }
                    _scanStatus = $"Found {_models.Count} models. Ready to launch.{hidden}";
                }
            }
            catch (Exception ex)
            {
                if (_headless)
                {
                    _scanStatus = $"Error scanning models folder: {ex.Message}";
                }
                else
                {
                    MessageBox.Show($"Error scanning models folder: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        // Quant tokens like Q4_K_M, IQ3_XXS, Q8_0, BF16, F16. First match wins.
        private static readonly Regex QuantRegex = new(
            @"(?i)\b(IQ\d+(?:_[A-Z0-9]+)*|Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32)\b", RegexOptions.Compiled);
        // MoE "mixture" form 8x7B.
        private static readonly Regex MoeParamsRegex = new(
            @"(?i)\b(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)B\b", RegexOptions.Compiled);
        // Active-params form A3B / A4B for MoE models that state it.
        private static readonly Regex ActiveParamsRegex = new(
            @"(?i)\bA(\d+(?:\.\d+)?)B\b", RegexOptions.Compiled);
        // Plain total-params form 8B, 26B, 3.2B.
        private static readonly Regex TotalParamsRegex = new(
            @"(?i)\b(\d+(?:\.\d+)?)B\b", RegexOptions.Compiled);
        // Descriptor keywords surfaced as tag chips, in display order.
        private static readonly string[] TagKeywords =
        {
            "Uncensored", "Abliterated", "Heretic", "Instruct", "Reasoning", "Reasoner", "Thinking",
            "Distill", "Coder", "Code", "Vision", "Roleplay", "Mythos", "Hermes", "MoE", "imat"
        };

        /// <summary>
        /// Fills in the display-only badge fields on a model from its filename. Best-effort and purely
        /// cosmetic - a filename that yields nothing just shows fewer badges.
        /// </summary>
        private static void ParseModelCard(LlmModel model)
        {
            var stem = Path.GetFileNameWithoutExtension(model.Name);

            var quant = QuantRegex.Match(stem);
            model.Quant = quant.Success ? quant.Value.ToUpperInvariant() : "GGUF";

            // Params: prefer the MoE "8x7B" form, then a total with an optional active count.
            var moe = MoeParamsRegex.Match(stem);
            if (moe.Success)
            {
                model.Params = moe.Value.ToUpperInvariant();
            }
            else
            {
                var total = TotalParamsRegex.Match(stem);
                var active = ActiveParamsRegex.Match(stem);
                if (total.Success)
                {
                    model.Params = active.Success
                        ? $"{total.Groups[1].Value}B·A{active.Groups[1].Value}B"
                        : $"{total.Groups[1].Value}B";
                }
                else
                {
                    model.Params = "";
                }
            }

            // Family: the run of tokens before the first size/quant token. Gives "Llama 3.2", "GLM 4.7
            // Flash", "Qwen3" etc. from typical filenames without needing real metadata.
            var tokens = stem.Split(new[] { '-', '_', '.', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var familyParts = new List<string>();
            foreach (var tok in tokens)
            {
                if (Regex.IsMatch(tok, @"(?i)^(IQ\d|Q\d|BF16|F16|F32)") ||
                    Regex.IsMatch(tok, @"(?i)^\d+(\.\d+)?x?\d*B$") ||
                    Regex.IsMatch(tok, @"(?i)^A\d"))
                {
                    break;
                }
                familyParts.Add(tok);
                if (familyParts.Count >= 3) break;
            }
            model.Family = familyParts.Count > 0 ? string.Join(" ", familyParts) : stem;

            var tags = new List<string>();
            foreach (var kw in TagKeywords)
            {
                if (stem.IndexOf(kw, StringComparison.OrdinalIgnoreCase) >= 0 &&
                    !tags.Any(t => string.Equals(t, kw, StringComparison.OrdinalIgnoreCase)))
                {
                    tags.Add(kw);
                }
            }
            model.Tags = tags;
        }

        // =====================================================================================
        // Setup page bridge
        // =====================================================================================

        /// <summary>
        /// The page finished rendering and asked for data. Scan, push the model list and device state,
        /// then kick off the first context estimate for the pre-selected model.
        /// </summary>
        private void OnSetupReady()
        {
            ScanModels();
            PushSetupInit();
            _ = PushCacheStatsAsync();
            if (_selectedModelPath != null)
            {
                ScheduleContextEstimate();
            }
        }

        private void SetupWebView_WebMessageReceived(object? sender, CoreWebView2WebMessageReceivedEventArgs e)
        {
            // Runs on the UI thread. Messages are the page's own JSON objects (WebMessageAsJson), which
            // is why every value below is read defensively - a malformed message must never crash the
            // host.
            string type;
            JsonElement root;
            try
            {
                using var doc = JsonDocument.Parse(e.WebMessageAsJson);
                root = doc.RootElement.Clone();
                type = root.TryGetProperty("type", out var t) ? (t.GetString() ?? "") : "";
            }
            catch
            {
                return;
            }

            switch (type)
            {
                case "ready":
                    OnSetupReady();
                    break;

                case "selectModel":
                    if (root.TryGetProperty("path", out var p))
                    {
                        _selectedModelPath = p.GetString();
                        ScheduleContextEstimate();
                    }
                    break;

                case "setDevices":
                    if (root.TryGetProperty("devices", out var devices) && devices.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var device in _config.Devices)
                        {
                            if (!devices.TryGetProperty(device.Id, out var flag)) continue;
                            if (flag.ValueKind == JsonValueKind.True) _selectedDevices.Add(device.Id);
                            else _selectedDevices.Remove(device.Id);
                        }
                        ScheduleContextEstimate();
                    }
                    break;

                case "setContext":
                    if (root.TryGetProperty("value", out var v) && v.TryGetInt32(out var cv))
                    {
                        _setupContextSize = cv;
                    }
                    break;

                case "setExpert":
                    if (root.TryGetProperty("layers", out var layers) && layers.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var device in _config.Devices.Where(d => d.IsRam))
                        {
                            _expertLayers[device.Id] = Math.Max(0, MsgInt(layers, device.Id));
                        }
                    }
                    break;

                case "setIntegration":
                    if (root.TryGetProperty("id", out var integrationId) &&
                        _config.Integrations.Any(i => i.Id == integrationId.GetString()))
                    {
                        if (MsgBool(root, "enabled")) _enabledIntegrations.Add(integrationId.GetString()!);
                        else _enabledIntegrations.Remove(integrationId.GetString()!);
                    }
                    break;

                case "launch":
                    LaunchSelectedAsync();
                    break;

                case "refreshCache":
                    _ = PushCacheStatsAsync();
                    break;

                case "clearCache":
                    if (root.TryGetProperty("hostId", out var hostId))
                    {
                        _ = HandleClearCacheAsync(hostId.GetString());
                    }
                    break;

                case "releaseGpus":
                    _ = HandleReleaseGpusAsync();
                    break;

                case "openConfig":
                    OpenConfigFile();
                    break;

                case "reloadConfig":
                    _ = ReloadConfigAsync();
                    break;
            }
        }

        private static bool MsgBool(JsonElement root, string name)
            => root.TryGetProperty(name, out var el) &&
               (el.ValueKind == JsonValueKind.True ||
                (el.ValueKind == JsonValueKind.String && el.GetString() == "true"));

        private static int MsgInt(JsonElement root, string name)
        {
            if (!root.TryGetProperty(name, out var el)) return 0;
            if (el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out var n)) return n;
            if (el.ValueKind == JsonValueKind.String && int.TryParse(el.GetString(), out var s)) return s;
            return 0;
        }

        /// <summary>Serialises an object to JSON and calls window.&lt;func&gt;(payload) in the page.</summary>
        private async void PostToSetup(string func, object payload)
        {
            var core = SetupWebView?.CoreWebView2;
            if (core == null) return;
            try
            {
                var json = JsonSerializer.Serialize(payload);
                await core.ExecuteScriptAsync($"window.{func}({json})");
            }
            catch
            {
                // A page reload mid-push, or the host going away, must not take the app down.
            }
        }

        /// <summary>Pushes the model list, the configured devices and their current state to the setup page.</summary>
        private void PushSetupInit()
        {
            var models = _models.Select(m => new
            {
                path = m.Path,
                name = m.Name,
                family = m.Family,
                @params = m.Params,
                quant = m.Quant,
                tags = m.Tags,
                sizeGB = m.SizeGiB
            }).ToList();

            var devices = _config.Devices.Select(d => new
            {
                id = d.Id,
                type = d.Type,
                label = d.Label,
                chip = d.ChipText,
                vramGb = d.VramGb,
                hostLabel = d.IsLocal ? null : _config.HostOf(d)?.DisplayName,
                on = _selectedDevices.Contains(d.Id),
                layers = _expertLayers.TryGetValue(d.Id, out var n) ? n : 0
            }).ToList();

            PostToSetup("applyInit", new
            {
                folder = ModelsDir,
                status = _scanStatus,
                selectedPath = _selectedModelPath,
                configPath = AppHome.ConfigPath,
                problems = _configProblems,
                localAddresses = LocalAddresses(),
                hosts = _config.Hosts.Select(h => new { id = h.Id, label = h.DisplayName, address = h.Address, hasCache = h.HasCache }),
                integrations = _config.Integrations.Select(i => new
                {
                    id = i.Id,
                    name = i.DisplayName,
                    description = i.Description,
                    on = _enabledIntegrations.Contains(i.Id)
                }),
                devices,
                models
            });
        }

        private static List<string> LocalAddresses()
        {
            try
            {
                return NetworkInterface.GetAllNetworkInterfaces()
                    .Where(n => n.OperationalStatus == OperationalStatus.Up &&
                                n.NetworkInterfaceType != NetworkInterfaceType.Loopback)
                    .SelectMany(n => n.GetIPProperties().UnicastAddresses)
                    .Where(a => a.Address.AddressFamily == AddressFamily.InterNetwork)
                    .Select(a => a.Address.ToString())
                    .Distinct()
                    .ToList();
            }
            catch
            {
                return new List<string>();
            }
        }

        /// <summary>Updates only the context-card caption on the page (used for the "measuring" and
        /// error states, where there is no fitted list to send).</summary>
        private void PushCaption(string caption) => PostToSetup("applyEstimate", new { caption });

        private void OpenConfigFile()
        {
            var path = AppHome.ConfigPath;
            try
            {
                Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
            }
            catch
            {
                // No program is associated with .json: fall back to Notepad rather than doing nothing.
                try { Process.Start("notepad.exe", $"\"{path}\""); } catch { }
            }
        }

        /// <summary>
        /// Re-reads the config. Anything running belongs to the old config, so it is stopped first;
        /// this is only offered on the setup page, where nothing is serving a chat.
        /// </summary>
        private async Task ReloadConfigAsync()
        {
            if (_apiLaunchBusy)
            {
                PostToSetup("applyReleased", new { message = "A launch is in progress." });
                return;
            }

            _estimateCts?.Cancel();
            await Task.Run(() => StopAllProcesses());

            LoadConfig();
            StopControlApi();
            StartControlApi();

            ScanModels();
            PushSetupInit();
            _ = PushCacheStatsAsync();
            if (_selectedModelPath != null)
            {
                ScheduleContextEstimate();
            }
        }

        // =====================================================================================
        // Remote worker tensor cache
        // =====================================================================================

        private async Task PushCacheStatsAsync()
        {
            var hosts = _config.Hosts.Where(h => h.HasCache).ToList();

            var results = await Task.WhenAll(hosts.Select(async h =>
            {
                var stats = await RemoteWorkers.GetCacheStatsAsync(h);
                return new
                {
                    id = h.Id,
                    label = h.DisplayName,
                    available = stats != null,
                    usedBytes = stats?.UsedBytes ?? 0,
                    freeBytes = stats?.FreeBytes ?? 0
                };
            }));

            PostToSetup("applyCache", new { hosts = results });
        }

        private async Task HandleClearCacheAsync(string? hostId)
        {
            var host = _config.Hosts.FirstOrDefault(h => h.Id == hostId);
            if (host == null) return;

            await RemoteWorkers.ClearCacheAsync(host);
            await PushCacheStatsAsync();
        }

        /// <summary>
        /// Setup-screen equivalent of the chat screen's "Stop &amp; Change Model" button. Covers the
        /// gap where a context estimate has already started the remote GPU workers (to measure free
        /// VRAM) but the model was never launched - until now, closing the whole app was the only
        /// way to free them back up. Runs the kill on a background thread since stopping a remote
        /// worker blocks on an ssh round trip; the WebView2 message handler itself must stay on the UI thread.
        /// </summary>
        private async Task HandleReleaseGpusAsync()
        {
            await Task.Run(() => StopAllProcesses());
            PostToSetup("applyReleased", new { message = "GPUs released." });
        }

        // =====================================================================================
        // Context estimate
        // =====================================================================================

        /// <summary>
        /// Recomputes the context list for the current model and device selection.
        ///
        /// Debounced, because ticking three boxes fires three times and each estimate spawns two
        /// llama-fit-params runs against every selected device. The previous estimate is cancelled
        /// rather than left to finish, so a fast series of clicks costs one measurement, not five.
        /// </summary>
        private async void ScheduleContextEstimate()
        {
            // The setup page may not have finished initialising yet (a device message cannot arrive
            // before then, but guard anyway).
            if (SetupWebView?.CoreWebView2 == null)
            {
                return;
            }

            _estimateCts?.Cancel();
            var cts = new CancellationTokenSource();
            _estimateCts = cts;

            try
            {
                await Task.Delay(400, cts.Token);
                await RunContextEstimateAsync(cts.Token);
            }
            catch (OperationCanceledException)
            {
                // Superseded by a newer selection.
            }
            catch (Exception ex)
            {
                PushCaption($"Could not size the context: {ex.Message}");
            }
        }

        private async Task RunContextEstimateAsync(CancellationToken cancel)
        {
            var model = _models.FirstOrDefault(m => m.Path == _selectedModelPath);
            if (model == null)
            {
                PushCaption("Select a model.");
                return;
            }

            var selection = Snapshot();
            if (!_config.Devices.Any(d => selection.DeviceIds.Contains(d.Id)))
            {
                PushCaption("No devices ticked.");
                return;
            }

            var plan = LaunchPlan.Build(_config, selection);
            var meta = GgufMetadata.TryRead(model.Path);
            var trainingContext = meta?.TrainingContext ?? 0;

            // The same device and endpoint ordering the launch will use, so the names line up with
            // what llama.cpp will call them. See LaunchPlan for why the order is load-bearing.
            var byName = plan.Devices.ToDictionary(p => p.Name, p => new EstimatorDevice
            {
                Name = p.Name,
                Label = p.Config.Label,
                MarginMib = p.Config.EffectiveMarginMib,
                IsRam = p.Config.IsRam
            });
            var devices = plan.Devices.Select(p => byName[p.Name]).ToList();

            PushCaption(plan.RpcEndpoints.Count > 0
                ? "Measuring free memory on the selected devices…"
                : "Measuring free memory…");

            // A remote device can only report its free memory through a running worker, so the ones
            // being measured have to be up. They are started exactly as a launch would start them and
            // are then reused by it, so this is not extra work - only earlier work.
            //
            // If a host cannot be reached, give up on the estimate rather than measuring the devices
            // that did come up: dropping one would renumber the RPC devices and quietly produce a
            // figure for a different arrangement than the one on screen.
            try
            {
                foreach (var planned in plan.Devices.Where(d => !d.Config.IsLocal))
                {
                    await _workers.EnsureRunningAsync(planned.Config);
                }
            }
            catch (Exception ex)
            {
                PushCaption(
                    $"Could not start a remote worker to measure its devices ({ex.Message.Split('.')[0]}). " +
                    "Pick a context size by hand.");
                return;
            }
            cancel.ThrowIfCancellationRequested();

            await PopulateFreeVramAsync(plan, byName);
            cancel.ThrowIfCancellationRequested();

            // A RAM device reports the machine's system memory, not VRAM, and it is not a card that can
            // run out in the same way. It stays in --device for the estimate so the placement matches
            // the run, but it never gets to be the limiting device.
            var vramDeviceCount = devices.Count(d => !d.IsRam);

            var estimate = await ContextEstimator.EstimateAsync(
                FitParamsExecutable, model.Path, devices, plan.RpcEndpoints,
                _config.LlamaServer, plan.CudaVisibleDevices, cancel);
            cancel.ThrowIfCancellationRequested();

            if (estimate.Error != null)
            {
                PushCaption(estimate.Error + " - pick a context size by hand.");
                return;
            }

            ApplyContextOptions(estimate, trainingContext, plan.UsesLocalRam, vramDeviceCount);
        }

        /// <summary>
        /// Puts the fitted list into the page, keeping the current choice when it still fits and
        /// otherwise dropping to the largest value that does.
        /// </summary>
        private void ApplyContextOptions(ContextEstimate estimate, uint trainingContext, bool useLocalRam,
                                         int vramDeviceCount)
        {
            var previous = _setupContextSize;

            // With RAM overflow on, running out of VRAM is not the end of the road: -ngl auto moves
            // layers onto this PC's CPU until the thing fits, so contexts past the VRAM figure still
            // load. The model's own training context becomes the only real ceiling. Without a readable
            // training context there is nothing to bound it with, so the VRAM figure stands.
            var ceiling = useLocalRam && trainingContext > 0 ? (int)trainingContext : estimate.MaxContext;
            var options = ContextEstimator.BuildOptions(ceiling, trainingContext);

            if (options.Count == 0)
            {
                var ramChip = _config.Devices.FirstOrDefault(d => d.IsLocal && d.IsRam)?.ChipText;
                // Empty option list + a zero ceiling tells the page to grey every ladder chip.
                PostToSetup("applyEstimate", new
                {
                    caption =
                        "The model does not fit these devices with any usable context" +
                        (estimate.LimitingDevice.Length > 0 ? $" - {estimate.LimitingDevice} fills up first." : ".") +
                        (ramChip != null
                            ? $" Tick another device, or \"{ramChip}\" to spill the rest onto the CPU."
                            : " Tick another device."),
                    options = Array.Empty<int>(),
                    maxContext = 0,
                    contextSize = _setupContextSize
                });
                return;
            }

            // Keep what was selected if it still fits; otherwise take the biggest that does, which is
            // the value someone picking from this list almost always wants.
            var chosen = previous > 0 && previous <= options[^1] ? previous : options[^1];
            _setupContextSize = chosen;

            var caption = new StringBuilder();
            if (estimate.MaxContext == int.MaxValue)
            {
                caption.Append(trainingContext > 0
                    ? $"No VRAM limit with these devices; capped at the model's {trainingContext:N0} tokens."
                    : "No VRAM limit with these devices.");
            }
            else
            {
                caption.Append($"{estimate.MaxContext:N0} tokens fit entirely in VRAM");
                if (estimate.LimitingDevice.Length > 0 && vramDeviceCount > 1)
                {
                    caption.Append($" ({estimate.LimitingDevice} fills first)");
                }
                if (estimate.KibPerToken > 0)
                {
                    caption.Append($", {estimate.KibPerToken:F1} KiB per token");
                }
                caption.Append('.');

                if (trainingContext > 0 && trainingContext < estimate.MaxContext)
                {
                    caption.Append($" Capped at the model's own {trainingContext:N0}.");
                }
            }

            if (useLocalRam)
            {
                caption.Append(" RAM overflow is on, so the list goes past that - llama.cpp moves layers onto this PC's CPU to make room, which is slower.");
            }

            PostToSetup("applyEstimate", new
            {
                caption = caption.ToString(),
                options,
                // The largest fitted value is the ceiling the page uses to grey any base ladder chip
                // that overshoots it.
                maxContext = options[^1],
                contextSize = chosen
            });
        }

        /// <summary>
        /// Fills in each GPU's free VRAM. nvidia-smi is the source on every machine, because it reports
        /// what is actually free right now - with a desktop, a game or another model running, that is
        /// far from the card's nominal size.
        /// </summary>
        private static async Task PopulateFreeVramAsync(LaunchPlan plan, Dictionary<string, EstimatorDevice> byName)
        {
            var reads = new List<Task>();

            var localGpus = plan.Devices.Where(d => d.Config.IsLocal && d.Config.IsGpu).ToList();
            if (localGpus.Count > 0)
            {
                reads.Add(Fill(RemoteWorkers.QueryLocalFreeMibAsync(), localGpus));
            }

            // One ssh round trip per host covers all of its cards.
            foreach (var host in plan.RemoteHosts)
            {
                var gpus = plan.Devices.Where(d => !d.Config.IsLocal && d.Config.IsGpu && d.Config.Host == host.Id).ToList();
                if (gpus.Count > 0)
                {
                    reads.Add(Fill(RemoteWorkers.QueryFreeMibAsync(host), gpus));
                }
            }

            await Task.WhenAll(reads);

            async Task Fill(Task<Dictionary<int, double>?> query, List<PlannedDevice> gpus)
            {
                var free = await query;
                if (free == null) return;
                foreach (var gpu in gpus)
                {
                    if (free.TryGetValue(gpu.Config.CudaIndex, out var mib))
                    {
                        byName[gpu.Name].FreeMib = mib;
                    }
                }
            }
        }

        // =====================================================================================
        // Launch
        // =====================================================================================

        private async void LaunchSelectedAsync()
        {
            // A context estimate in flight would be racing this for the same worker start calls.
            _estimateCts?.Cancel();

            var selectedModel = _models.FirstOrDefault(m => m.Path == _selectedModelPath);
            if (selectedModel == null)
            {
                MessageBox.Show("Please select a GGUF model first.", "No Model Selected", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // The device ticks, integrations, context size and expert-layer counts are all live setup
            // state, kept current by the page's messages. Only the context size still needs validating.
            if (!ValidateContextSize(out int contextSize))
            {
                return;
            }

            // Every one of these checks exists because the corresponding failure would otherwise happen
            // after the setup panel has been torn down, with only an opaque llama-server message to show
            // for it. Catch them while the boxes are still on screen.
            var selection = Snapshot();
            var problem = LaunchPlan.Validate(_config, selection);
            if (problem != null)
            {
                MessageBox.Show(problem, "Cannot Launch", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var plan = LaunchPlan.Build(_config, selection);
            var integrations = _config.Integrations.Where(i => _enabledIntegrations.Contains(i.Id)).ToList();

            SetupPanel.Visibility = Visibility.Collapsed;
            LoadingPanel.Visibility = Visibility.Visible;
            LoadingStatus.Text = "Initializing local and remote servers...";

            try
            {
                // --jinja is needed for tool calling, which is what an MCP integration is for.
                await BringUpAsync(plan, selectedModel.Path, contextSize, jinja: integrations.Count > 0,
                                   status: text => LoadingStatus.Text = text);

                if (plan.AnyRemoteGpu)
                {
                    await Task.Delay(1000);   // long enough to read the split verdict
                }

                // Start enabled MCP bridges.
                foreach (var integration in integrations)
                {
                    LoadingStatus.Text = $"Starting {integration.DisplayName}...";
                    StartIntegration(integration);
                }

                // Initialize WebView2 and show the chat panel.
                LoadingStatus.Text = "Initializing chat interface...";
                await ChatWebView!.EnsureCoreWebView2Async();
                await SyncChatWebViewMcpSettingsAsync();
                ChatWebView.CoreWebView2.Navigate($"http://localhost:{ServerPort}");

                _activePlan = plan;
                RunningModelTitle.Text = $"Model: {selectedModel.Name}";
                ApplyClusterStatus(plan);

                // Show/hide the MCP status indicator.
                if (integrations.Count > 0)
                {
                    McpStatusLabel.Text = string.Join(" + ", integrations.Select(DescribeIntegration));
                    McpStatusDot.Visibility = Visibility.Visible;
                    McpStatusLabel.Visibility = Visibility.Visible;
                }
                else
                {
                    McpStatusDot.Visibility = Visibility.Collapsed;
                    McpStatusLabel.Visibility = Visibility.Collapsed;
                }

                LoadingPanel.Visibility = Visibility.Collapsed;
                ChatPanel.Visibility = Visibility.Visible;
            }
            catch (Exception ex)
            {
                StopAllProcesses();
                LoadingPanel.Visibility = Visibility.Collapsed;
                SetupPanel.Visibility = Visibility.Visible;
                MessageBox.Show($"Failed to launch LLM: {ex.Message}", "Launch Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private static string DescribeIntegration(IntegrationConfig integration)
        {
            if (Uri.TryCreate(integration.Url, UriKind.Absolute, out var uri))
            {
                return $"{integration.DisplayName} (port {uri.Port})";
            }
            return integration.DisplayName;
        }

        /// <summary>
        /// The launch pipeline shared by the setup page and the control API: remote workers, then the
        /// local llama-server, then a wait for the model to load. It reports progress through
        /// <paramref name="status"/> and throws a readable error if anything fails; the caller owns
        /// cleanup.
        /// </summary>
        private async Task BringUpAsync(LaunchPlan plan, string modelPath, int contextSize, bool jinja,
                                        Action<string> status, CancellationToken cancel = default)
        {
            // Refuse to launch on top of a server we did not start.
            EnsurePortFree(ServerPort);

            // Remote GPU workers first, then the baseline, then the RAM workers. Each EnsureRunning only
            // returns once its port answers, so reaching the baseline means every requested GPU worker
            // is genuinely up. Take it before any model is loaded: nvidia-smi sums every rpc-server on
            // the box, so a worker started later would otherwise have its bare CUDA context counted as
            // model weights.
            foreach (var planned in plan.Devices.Where(d => !d.Config.IsLocal && d.Config.IsGpu))
            {
                status($"Starting {planned.Config.Label} worker on {HostName(planned.Config)} ({planned.Endpoint})...");
                await _workers.EnsureRunningAsync(planned.Config);
                cancel.ThrowIfCancellationRequested();
            }

            if (plan.AnyRemoteGpu)
            {
                _workerVramBaselineMib = await RemoteWorkers.GetWorkerVramMibAsync(GpuHosts(plan));
                cancel.ThrowIfCancellationRequested();
                status("Remote GPU workers are up. Starting local server...");
            }

            foreach (var planned in plan.Devices.Where(d => !d.Config.IsLocal && d.Config.IsRam))
            {
                status($"Starting {planned.Config.Label} worker on {HostName(planned.Config)} ({planned.Endpoint})...");
                await _workers.EnsureRunningAsync(planned.Config);
                cancel.ThrowIfCancellationRequested();
            }

            status("Starting local llama-server...");
            cancel.ThrowIfCancellationRequested();
            StartLocalServer(plan, modelPath, contextSize, jinja);

            status("Waiting for the model to load...");
            // Any remote device means the weights cross the network, so the long budget applies.
            var serverReady = await PollServerAsync(plan.AnyRemote, cancel);
            if (!serverReady)
            {
                throw new Exception(
                    $"llama-server failed to start or load the model within the timeout.\n\n{GetServerLogTail()}");
            }

            // Measure the workers' GPUs now that the model is loaded. The port only binds after "model
            // loaded", so reaching here means placement has already happened.
            if (plan.AnyRemoteGpu)
            {
                status("Checking what actually landed on the remote GPUs...");
                _workerVramLoadedMib = await RemoteWorkers.GetWorkerVramMibAsync(GpuHosts(plan));

                var weights = _workerVramLoadedMib - Math.Max(0, _workerVramBaselineMib);
                if (_workerVramLoadedMib < 0)
                {
                    status("Could not read the remote GPUs to confirm the split.");
                }
                else if (weights >= WorkerWeightsThresholdMib)
                {
                    status($"Split confirmed: {weights:N0} MiB of weights on {HostNames(plan)}.");
                }
                else
                {
                    status("Warning: no weights landed on the remote GPUs.");
                }
            }
        }

        private string HostName(DeviceConfig device) => _config.HostOf(device)?.DisplayName ?? "this PC";

        private static string HostNames(LaunchPlan plan)
            => string.Join(" + ", plan.RemoteHosts.Select(h => h.DisplayName));

        /// <summary>The hosts whose GPU workers' memory should be summed for the split check.</summary>
        private IEnumerable<HostConfig> GpuHosts(LaunchPlan plan)
            => plan.Selected.Where(d => !d.IsLocal && d.IsGpu)
                            .Select(d => _config.HostOf(d)!)
                            .Distinct();

        private static readonly Brush StatusOkBrush = CreateFrozenBrush(0x10, 0xB9, 0x81);   // green
        private static readonly Brush StatusWarnBrush = CreateFrozenBrush(0xF5, 0x9E, 0x0B); // amber
        private static readonly Brush StatusMutedBrush = CreateFrozenBrush(0x9C, 0xA3, 0xAF); // grey

        private static Brush CreateFrozenBrush(byte r, byte g, byte b)
        {
            var brush = new SolidColorBrush(Color.FromRgb(r, g, b));
            brush.Freeze();
            return brush;
        }

        /// <summary>
        /// Reports where the weights actually went, based on the workers' real GPU allocation. An
        /// earlier version derived this from the checkbox, so it read "Split-GPU Active" even when
        /// nothing had been offloaded to the remote machine.
        /// </summary>
        private void ApplyClusterStatus(LaunchPlan plan)
        {
            // The ticked devices, in the order llama.cpp was given them. Printed whatever happens, so
            // the bar always says what the run was asked to use even when the check below cannot run.
            var deviceList = string.Join(" + ", plan.Selected.Select(d => d.Label));

            if (!plan.AnyRemoteGpu)
            {
                ClusterStatusLabel.Foreground = StatusMutedBrush;
                ClusterStatusLabel.Text = $"  |  {deviceList}";
                return;
            }

            var remoteGpus = string.Join(" + ", plan.Selected.Where(d => !d.IsLocal && d.IsGpu).Select(d => d.Label));
            var hosts = HostNames(plan);

            // Could not read the workers' GPUs: say so rather than pick a verdict at random.
            if (_workerVramLoadedMib < 0)
            {
                ClusterStatusLabel.Foreground = StatusMutedBrush;
                ClusterStatusLabel.Text = $"  |  {deviceList} - could not read the remote GPUs to confirm";
                return;
            }

            var baseline = _workerVramBaselineMib < 0 ? 0 : _workerVramBaselineMib;
            var weights = _workerVramLoadedMib - baseline;

            if (weights >= WorkerWeightsThresholdMib)
            {
                ClusterStatusLabel.Foreground = StatusOkBrush;
                ClusterStatusLabel.Text = plan.AnyLocalGpu
                    ? $"  |  {deviceList} - {weights:N0} MiB of weights on {hosts} ({remoteGpus})"
                    : $"  |  {deviceList} - {weights:N0} MiB on {hosts}, nothing on this PC's GPUs";
            }
            else
            {
                ClusterStatusLabel.Foreground = StatusWarnBrush;
                ClusterStatusLabel.Text =
                    $"  |  ⚠ {deviceList} requested, but the remote GPUs hold no weights ({_workerVramLoadedMib:N0} MiB)";
            }
        }

        /// <summary>
        /// llama-server binds its port, and PollServerAsync only probes that port - it cannot tell our
        /// server apart from someone else's. If a server from an earlier run is still listening (for
        /// example one orphaned by a crash), the poll would succeed immediately and the chat panel
        /// would quietly serve that process's model instead of the one just selected. Fail loudly.
        /// </summary>
        private static void EnsurePortFree(int port)
        {
            var listeners = IPGlobalProperties
                .GetIPGlobalProperties()
                .GetActiveTcpListeners();

            if (listeners.Any(ep => ep.Port == port))
            {
                throw new Exception(
                    $"Port {port} is already in use, most likely by a llama-server left over from an " +
                    "earlier session. Stop that process first - otherwise this window would silently " +
                    "talk to it, and its model, instead of the one you selected.");
            }
        }

        /// <summary>
        /// Validates the context size chosen on the setup page. llama-server takes -c as a plain token
        /// count; a too-short value would otherwise be accepted and only bite mid-chat as an opaque
        /// HTTP 500 (see MinContextSize), so it is rejected here while the setup page is still up.
        /// </summary>
        private bool ValidateContextSize(out int contextSize)
        {
            contextSize = _setupContextSize;
            if (contextSize < MinContextSize)
            {
                MessageBox.Show(
                    $"Context size must be a whole number of at least {MinContextSize} tokens.",
                    "Invalid Context Size", MessageBoxButton.OK, MessageBoxImage.Warning);
                contextSize = 0;
                return false;
            }

            return true;
        }

        // =====================================================================================
        // Local llama-server
        // =====================================================================================

        private void StartLocalServer(LaunchPlan plan, string modelPath, int contextSize, bool jinja)
        {
            var executable = LocalServerExecutable;
            if (!File.Exists(executable))
            {
                throw new FileNotFoundException(
                    $"llama-server was not found at {executable}. Build llama.cpp (see the README) or " +
                    $"point llamaServer.path in {PortalConfig.FileName} at your build.");
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = executable,
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };

            Shell.ApplyCudaEnvironment(startInfo.EnvironmentVariables, _config.LlamaServer, plan.CudaVisibleDevices);

            var args = plan.BuildArguments(modelPath, contextSize, jinja);
            startInfo.Arguments = args;

            lock (_localServerLog)
            {
                _localServerLog.Clear();
            }

            _workerVramLoadedMib = -1;

            try
            {
                lock (_logFileLock)
                {
                    File.WriteAllText(ServerLogPath,
                        $"# llama-server launch {DateTime.Now:yyyy-MM-dd HH:mm:ss}{Environment.NewLine}" +
                        $"# args: {args}{Environment.NewLine}");
                }
            }
            catch { }

            _localServerProcess = new Process { StartInfo = startInfo };
            _localServerProcess.OutputDataReceived += (s, ev) => AppendServerLog(ev.Data);
            _localServerProcess.ErrorDataReceived += (s, ev) => AppendServerLog(ev.Data);

            _localServerProcess.Start();

            // llama.cpp logs through a bounded 512-entry queue whose writer thread blocks on a full
            // stderr pipe; once the queue backs up, llama-server's main thread blocks forever. These
            // two calls keep the pipes empty and must stay for as long as the process is redirected.
            _localServerProcess.BeginOutputReadLine();
            _localServerProcess.BeginErrorReadLine();
        }

        private void AppendServerLog(string? data)
        {
            if (data == null)
            {
                return;
            }

            lock (_localServerLog)
            {
                _localServerLog.AppendLine(data);
                if (_localServerLog.Length > ServerLogMaxChars)
                {
                    _localServerLog.Remove(0, _localServerLog.Length - ServerLogMaxChars);
                }
            }

            WriteServerLogFile(data);
        }

        /// <summary>
        /// Mirror llama-server's output to a file. Without this the only copy lives in memory and
        /// dies with the process, which makes a wrong reading of the load_tensors output impossible
        /// to diagnose after the fact.
        /// </summary>
        private void WriteServerLogFile(string line)
        {
            try
            {
                lock (_logFileLock)
                {
                    File.AppendAllText(ServerLogPath, line + Environment.NewLine);
                }
            }
            catch
            {
                // Diagnostics must never take the app down.
            }
        }

        private string GetServerLogTail(int maxChars = 4000)
        {
            lock (_localServerLog)
            {
                if (_localServerLog.Length == 0)
                {
                    return "(no output captured from llama-server)";
                }

                return _localServerLog.Length <= maxChars
                    ? _localServerLog.ToString()
                    : _localServerLog.ToString(_localServerLog.Length - maxChars, maxChars);
            }
        }

        private async Task<bool> PollServerAsync(bool useRemoteDevices, CancellationToken cancel = default)
        {
            // A large quant loading over RPC has to push gigabytes to the remote worker across the
            // network before it will answer, which comfortably exceeds a 2-minute budget on a cold
            // page cache. A premature exit is caught by the HasExited check below, so waiting longer
            // costs nothing on the failure path.
            var budget = useRemoteDevices ? TimeSpan.FromMinutes(20) : TimeSpan.FromMinutes(5);
            var deadline = DateTime.UtcNow + budget;

            using (var client = new HttpClient())
            {
                client.Timeout = TimeSpan.FromSeconds(2);

                while (DateTime.UtcNow < deadline)
                {
                    // The control API's /control/stop can cancel an in-flight launch; the default
                    // token (the setup page's own launch path) is never cancellable.
                    cancel.ThrowIfCancellationRequested();

                    // Check if process exited prematurely. The streams are drained by
                    // BeginOutputReadLine/BeginErrorReadLine, so read the captured tail rather than
                    // the stream itself - ReadToEnd on a stream in async-read mode throws.
                    if (_localServerProcess != null && _localServerProcess.HasExited)
                    {
                        throw new Exception(
                            $"Local llama-server exited with code {_localServerProcess.ExitCode}.\n\n{GetServerLogTail()}");
                    }

                    try
                    {
                        // Must be /health, not "/". llama-server binds the socket and serves the web
                        // UI from a thread BEFORE it loads the model, so "/" answers 200 within
                        // seconds and says nothing about readiness - polling it reported the model
                        // ready ~70s early, while the weights were still being sent to the worker.
                        // /health sits behind the loading middleware: 503 until the model is live.
                        var response = await client.GetAsync($"http://localhost:{ServerPort}/health");
                        if (response.IsSuccessStatusCode)
                        {
                            return true;
                        }
                    }
                    catch
                    {
                        // Server not up yet.
                    }

                    await Task.Delay(2000, cancel);
                }
            }
            return false;
        }

        // =====================================================================================
        // Integrations (MCP bridges)
        // =====================================================================================

        private static string ExpandHome(string text) => text.Replace("{home}", Home, StringComparison.OrdinalIgnoreCase);

        private void StartIntegration(IntegrationConfig integration)
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = ExpandHome(integration.Command),
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            foreach (var arg in integration.Args)
            {
                startInfo.ArgumentList.Add(ExpandHome(arg));
            }

            var process = new Process { StartInfo = startInfo };
            try
            {
                process.Start();
            }
            catch (Exception ex)
            {
                throw new Exception($"Could not start {integration.DisplayName} ({integration.Command}): {ex.Message}");
            }
            lock (_integrationProcesses) { _integrationProcesses[integration.Id] = process; }

            // Long-lived child with redirected stdio: drain or it stalls once the pipes fill.
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
        }

        /// <summary>
        /// Registers the running integrations with llama.cpp's web UI by editing its saved settings
        /// before the page loads, so they appear (and are switched on or off) without manual setup.
        /// </summary>
        private async Task SyncChatWebViewMcpSettingsAsync()
        {
            try
            {
                var core = ChatWebView?.CoreWebView2;
                if (core == null) return;

                if (!string.IsNullOrEmpty(_chatMcpInitScriptId))
                {
                    core.RemoveScriptToExecuteOnDocumentCreated(_chatMcpInitScriptId);
                    _chatMcpInitScriptId = null;
                }

                var upserts = new StringBuilder();
                foreach (var integration in _config.Integrations.Where(i => !string.IsNullOrWhiteSpace(i.Url)))
                {
                    var enabled = _enabledIntegrations.Contains(integration.Id) ? "true" : "false";
                    upserts.AppendLine(
                        $"    upsert({JsonSerializer.Serialize(integration.Id)}, " +
                        $"{JsonSerializer.Serialize(integration.DisplayName)}, " +
                        $"{JsonSerializer.Serialize(integration.Url)}, {enabled});");
                }

                var script = @"
(function() {
  try {
    if (location.origin !== 'http://localhost:__PORT__' && location.origin !== 'http://127.0.0.1:__PORT__') return;
    var key = 'LlamaUi.config';
    var cfg = {};
    try { cfg = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) { cfg = {}; }
    var servers = [];
    try {
      servers = typeof cfg.mcpServers === 'string'
        ? JSON.parse(cfg.mcpServers || '[]')
        : (Array.isArray(cfg.mcpServers) ? cfg.mcpServers : []);
    } catch (e) { servers = []; }
    function upsert(id, name, url, enabled) {
      var idx = servers.findIndex(function(s) { return s && (s.id === id || s.url === url); });
      var entry = { id: id, name: name, displayName: name, url: url, enabled: enabled, useProxy: false };
      if (idx >= 0) { servers[idx] = Object.assign({}, servers[idx], entry); }
      else if (enabled) { servers.push(entry); }
    }
__UPSERTS__
    cfg.mcpServers = JSON.stringify(servers);
    localStorage.setItem(key, JSON.stringify(cfg));
  } catch (e) {}
})();"
                    .Replace("__PORT__", ServerPort.ToString(CultureInfo.InvariantCulture))
                    .Replace("__UPSERTS__", upserts.ToString());

                _chatMcpInitScriptId = await core.AddScriptToExecuteOnDocumentCreatedAsync(script);
            }
            catch
            {
                // Non-fatal: the user can still toggle MCP servers inside llama-server's web UI settings.
            }
        }

        // =====================================================================================
        // Stop
        // =====================================================================================

        private void StopBtn_Click(object sender, RoutedEventArgs e)
        {
            StopAllProcesses();
            ChatPanel.Visibility = Visibility.Collapsed;
            SetupPanel.Visibility = Visibility.Visible;

            // Re-scan (a model may have been added/removed while chatting) and re-push the list + state
            // to the setup page, then re-estimate for whatever ends up selected.
            ScanModels();
            PushSetupInit();
            _ = PushCacheStatsAsync();
            if (_selectedModelPath != null)
            {
                ScheduleContextEstimate();
            }
        }

        private void StopAllProcesses()
        {
            // --- Phase 1: Kill tracked process trees ---
            var server = _localServerProcess;
            _localServerProcess = null;
            Shell.KillTree(server);

            List<Process> integrationProcesses;
            lock (_integrationProcesses)
            {
                integrationProcesses = _integrationProcesses.Values.ToList();
                _integrationProcesses.Clear();
            }
            foreach (var process in integrationProcesses)
            {
                Shell.KillTree(process);
            }

            // --- Phase 2: Kill by name as fallback, in case tracked references were lost (crash, race) ---
            // Only instances whose executable is the configured one (for llama-server) or lives under
            // the portal's folder (for integrations) are touched, so unrelated processes of the same
            // name on this machine survive. A process whose path cannot be read is left alone.
            var serverExe = LocalServerExecutable;
            KillProcessesByName(Path.GetFileNameWithoutExtension(serverExe),
                path => string.Equals(path, serverExe, StringComparison.OrdinalIgnoreCase));
            foreach (var name in _config.Integrations
                         .Select(i => i.ProcessName)
                         .Where(n => !string.IsNullOrWhiteSpace(n))
                         .Distinct()!)
            {
                KillProcessesByName(name!, path => path.StartsWith(Home, StringComparison.OrdinalIgnoreCase));
            }

            // --- Phase 3: Stop the remote workers ---
            _workers.StopAll();
        }

        private static void KillProcessesByName(string processName, Func<string, bool> pathMatches)
        {
            Process[] candidates;
            try
            {
                candidates = Process.GetProcessesByName(processName);
            }
            catch
            {
                return;
            }

            foreach (var proc in candidates)
            {
                try
                {
                    string? path = null;
                    try
                    {
                        path = proc.MainModule?.FileName;
                    }
                    catch
                    {
                        // Access denied, or the process exited between enumeration and inspection.
                    }

                    if (path == null || !pathMatches(path))
                    {
                        continue;
                    }

                    proc.Kill(true);
                    proc.WaitForExit(2000);
                }
                catch { }
                finally { proc.Dispose(); }
            }
        }

        private void MainWindow_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
        {
            StopControlApi();
            StopAllProcesses();

            // Dispose WebView2 instances to release their browser processes
            try { ChatWebView?.Dispose(); } catch { }
            try { SetupWebView?.Dispose(); } catch { }
        }

        // =====================================================================================
        // Control API (loopback HTTP; the listener/routing lives in ControlApi.cs)
        //
        // It exists so another program can read this portal's state and ask it to load a chosen GGUF
        // across a chosen set of devices without driving the setup WebView. Everything below is
        // additive: the setup page's own bridge and the manual launch path above are unchanged.
        // =====================================================================================

        private void StartControlApi()
        {
            if (!_config.ControlApi.Enabled)
            {
                return;
            }

            try
            {
                _controlApi = new ControlApi(this, Dispatcher, _config.ControlApi.Port);
                _controlApi.Start();
            }
            catch (Exception ex)
            {
                // A control API that cannot bind must not take the portal down; the setup page
                // remains fully usable on its own.
                Debug.WriteLine($"control API failed to start: {ex.Message}");
            }
        }

        private void StopControlApi()
        {
            try { _controlApi?.Stop(); } catch { }
            _controlApi = null;
        }

        /// <summary>True while an API launch pipeline is running (used to answer 409s).</summary>
        internal bool ApiLaunchBusy => _apiLaunchBusy;

        /// <summary>Every configured device id mapped to whether it is ticked.</summary>
        private Dictionary<string, bool> SnapshotApiDevices()
            => _config.Devices.ToDictionary(d => d.Id, d => _selectedDevices.Contains(d.Id));

        private Dictionary<string, int> SnapshotApiExpertLayers()
            => _config.Devices.Where(d => d.IsRam).ToDictionary(d => d.Id, d => _expertLayers.TryGetValue(d.Id, out var n) ? n : 0);

        /// <summary>
        /// Serialises the object GET /control/state returns. Called on the UI thread by ControlApi;
        /// the field names and nesting are fixed by docs/CONTROL_API.md.
        /// </summary>
        internal string BuildApiStateJson()
        {
            var models = _models.Select(m => new
            {
                path = m.Path,
                name = m.Name,
                family = m.Family,
                @params = m.Params,
                quant = m.Quant,
                sizeGB = m.SizeGiB
            }).ToList();

            object? running = _apiRunningModelPath == null
                ? null
                : new
                {
                    modelPath = _apiRunningModelPath,
                    modelName = _apiRunningModelName,
                    contextSize = _apiRunningContextSize,
                    devices = _apiRunningDevices,
                    expertLayers = _apiRunningExpertLayers
                };

            var deviceCatalog = _config.Devices.Select(d => new
            {
                id = d.Id,
                type = d.Type,
                label = d.Label,
                host = d.IsLocal ? null : d.Host
            }).ToList();

            var state = new
            {
                ok = true,
                apiVersion = 1,
                phase = _apiPhase,
                message = _apiMessage,
                error = _apiError,
                models,
                selectedPath = _selectedModelPath,
                deviceCatalog,
                devices = SnapshotApiDevices(),
                expertLayers = SnapshotApiExpertLayers(),
                running,
                serverReady = _apiReady && _apiPhase == "ready",
                headless = _headless
            };

            return JsonSerializer.Serialize(state);
        }

        /// <summary>
        /// POST /control/devices: applies the flags exactly like the setup page's "setDevices"
        /// message, but without scheduling a context estimate - the API caller drives the launch
        /// itself, and an estimate would start the remote workers just to measure memory. Flags for
        /// devices that are left out keep their current value. Returns null on success, or a readable
        /// error for a 400; nothing is applied when any entry is invalid.
        /// </summary>
        internal string? ApplyApiDevices(JsonElement devices)
        {
            if (devices.ValueKind != JsonValueKind.Object)
            {
                return "devices must be an object";
            }

            var updates = new Dictionary<string, bool>();
            foreach (var property in devices.EnumerateObject())
            {
                if (_config.FindDevice(property.Name) == null)
                {
                    return $"unknown device id: {property.Name}";
                }

                switch (property.Value.ValueKind)
                {
                    case JsonValueKind.True:
                        updates[property.Name] = true;
                        break;
                    case JsonValueKind.False:
                        updates[property.Name] = false;
                        break;
                    default:
                        return $"{property.Name} must be a boolean";
                }
            }

            if (updates.Count == 0)
            {
                return "devices must contain at least one of: " + string.Join(", ", _config.Devices.Select(d => d.Id));
            }

            foreach (var (id, on) in updates)
            {
                if (on) _selectedDevices.Add(id);
                else _selectedDevices.Remove(id);
            }

            return null;
        }

        /// <summary>Applies {"<ram device id>": layers} from a launch request. Null on success.</summary>
        private string? ApplyApiExpertLayers(JsonElement layers)
        {
            if (layers.ValueKind != JsonValueKind.Object)
            {
                return "expertLayers must be an object";
            }

            var updates = new Dictionary<string, int>();
            foreach (var property in layers.EnumerateObject())
            {
                var device = _config.FindDevice(property.Name);
                if (device == null || !device.IsRam)
                {
                    return $"expertLayers: \"{property.Name}\" is not a RAM device id";
                }
                if (property.Value.ValueKind != JsonValueKind.Number || !property.Value.TryGetInt32(out var n) || n < 0)
                {
                    return $"expertLayers.{property.Name} must be 0 or a positive integer";
                }
                updates[property.Name] = n;
            }

            foreach (var (id, n) in updates)
            {
                _expertLayers[id] = n;
            }

            return null;
        }

        /// <summary>
        /// POST /control/select. Only accepts a path from the current scan, so a typo cannot make a
        /// later launch fail after the workers are already up.
        /// </summary>
        internal string? TrySelectApiModel(string path)
        {
            var model = _models.FirstOrDefault(m =>
                string.Equals(m.Path, path, StringComparison.OrdinalIgnoreCase));
            if (model == null)
            {
                return $"path is not one of the scanned models: {path}";
            }

            _selectedModelPath = model.Path;
            return null;
        }

        /// <summary>
        /// POST /control/launch validation + start. Mirrors every check LaunchSelectedAsync makes
        /// behind a MessageBox, but returns a readable error instead of a dialog, and never touches
        /// SetupPanel/ChatPanel/LoadingPanel or the MCP bridges.
        /// Returns 202 once the launch task is running, 400 for a bad request, 409 when busy.
        /// </summary>
        internal (int Status, string? Error) TryBeginApiLaunch(ApiLaunchOptions options)
        {
            if (_apiLaunchBusy)
            {
                return (409, "launch already in progress");
            }

            if (options.Devices.HasValue)
            {
                var devicesError = ApplyApiDevices(options.Devices.Value);
                if (devicesError != null)
                {
                    return (400, devicesError);
                }
            }

            if (options.ExpertLayers.HasValue)
            {
                var layersError = ApplyApiExpertLayers(options.ExpertLayers.Value);
                if (layersError != null)
                {
                    return (400, layersError);
                }
            }

            LlmModel? model;
            if (options.Path != null)
            {
                model = _models.FirstOrDefault(m =>
                    string.Equals(m.Path, options.Path, StringComparison.OrdinalIgnoreCase));
                if (model == null)
                {
                    return (400, $"path is not one of the scanned models: {options.Path}");
                }
                _selectedModelPath = model.Path;
            }
            else
            {
                model = _models.FirstOrDefault(m => m.Path == _selectedModelPath);
            }

            if (model == null)
            {
                return (400, "no GGUF model is selected; send a path or select a model first");
            }

            var contextSize = options.ContextSize ?? _setupContextSize;
            if (contextSize < MinContextSize)
            {
                return (400, $"contextSize must be at least {MinContextSize} tokens (got {contextSize})");
            }

            // Same device / expert-offload checks as LaunchSelectedAsync, without the dialogs.
            var launchSelection = Snapshot();
            var problem = LaunchPlan.Validate(_config, launchSelection);
            if (problem != null)
            {
                return (400, problem);
            }

            // The API-requested context becomes the portal's current context, so a later launch
            // that omits contextSize agrees with the one that is actually loaded.
            _setupContextSize = contextSize;

            _apiLaunchBusy = true;
            var launchId = ++_apiLaunchId;
            var stateEpoch = _apiStateEpoch;
            _apiLaunchCts = new CancellationTokenSource();

            _apiPhase = "loading";
            _apiMessage = $"Loading {model.Name}...";
            _apiError = null;
            _apiReady = false;
            _apiRunningModelPath = null;
            _apiRunningModelName = null;
            _apiRunningDevices = null;
            _apiRunningExpertLayers = null;

            // /control/devices answers 409 while a launch is busy, so nothing can change the selection
            // mid-startup; the snapshot is passed down as well to keep the launch independent of any
            // later field write.
            var launchToken = _apiLaunchCts.Token;
            _ = RunApiLaunchAsync(model, contextSize, launchSelection, launchId, stateEpoch, launchToken);
            return (202, null);
        }

        /// <summary>
        /// The API launch pipeline: the same worker / llama-server machinery LaunchSelectedAsync uses,
        /// minus the dialogs, the chat-panel switch, the WebView navigation and the MCP bridges, and
        /// with --jinja always on (OpenAI tool calling needs it).
        /// </summary>
        private async Task RunApiLaunchAsync(LlmModel model, int contextSize, LaunchSelection selection,
                                             int launchId, int stateEpoch, CancellationToken cancel)
        {
            // Pin the selection this launch was accepted with, so --rpc/--device always match the
            // workers that started.
            _selectedDevices.Clear();
            foreach (var id in selection.DeviceIds) _selectedDevices.Add(id);
            _expertLayers.Clear();
            foreach (var (id, n) in selection.ExpertLayers) _expertLayers[id] = n;

            var plan = LaunchPlan.Build(_config, selection);

            try
            {
                // A context estimate in flight would race this for the same worker start calls.
                _estimateCts?.Cancel();

                // A launch replaces whatever is running (contract: stop the existing processes
                // first). StopAllProcesses blocks on process kills and an ssh round trip, so it
                // runs off the UI thread; the delay lets the old listener actually go away.
                await Task.Run(() => StopAllProcesses());
                cancel.ThrowIfCancellationRequested();
                await Task.Delay(300, cancel);

                await BringUpAsync(plan, model.Path, contextSize, jinja: true,
                                   status: text => _apiMessage = text, cancel);

                // A /control/stop may have superseded this launch while the model was loading.
                // In that case leave the idle state the stop installed alone.
                if (stateEpoch != _apiStateEpoch)
                {
                    return;
                }

                _apiRunningModelPath = model.Path;
                _apiRunningModelName = model.Name;
                _apiRunningContextSize = contextSize;
                _apiRunningDevices = _config.Devices.ToDictionary(d => d.Id, d => selection.DeviceIds.Contains(d.Id));
                _apiRunningExpertLayers = _config.Devices.Where(d => d.IsRam).ToDictionary(d => d.Id, d => selection.LayersFor(d.Id));
                _apiReady = true;
                _apiPhase = "ready";
                _apiMessage = $"Ready: {model.Name}";
            }
            catch (OperationCanceledException)
            {
                // /control/stop cancelled this launch. Always free whatever was started; only
                // write state when no newer stop/launch has taken ownership in the meantime.
                try
                {
                    await Task.Run(() => StopAllProcesses());
                }
                catch { }

                if (stateEpoch == _apiStateEpoch)
                {
                    ResetApiIdleStateCore();
                }
            }
            catch (Exception ex)
            {
                // Never leave a half-started cluster behind.
                try
                {
                    await Task.Run(() => StopAllProcesses());
                }
                catch { }

                if (stateEpoch == _apiStateEpoch)
                {
                    _apiPhase = "error";
                    _apiMessage = "Launch failed.";
                    _apiError = ex.Message;
                    _apiReady = false;
                    _apiRunningModelPath = null;
                    _apiRunningModelName = null;
                    _apiRunningDevices = null;
                    _apiRunningExpertLayers = null;
                }
            }
            finally
            {
                // Only the launch that currently owns the busy flag may clear it. A stop does not
                // clear it: it stays true until this pipeline has finished freeing processes, so a
                // launch issued immediately after a stop gets a clean 409 instead of racing the
                // cancelled pipeline's cleanup.
                if (launchId == _apiLaunchId)
                {
                    _apiLaunchBusy = false;
                }
            }
        }

        /// <summary>POST /control/stop: stop every tracked process (called off the UI thread).</summary>
        internal void StopProcessesFromApi()
        {
            // Bump the epoch before touching processes: an in-flight launch that is about to
            // report ready/error then sees the epoch mismatch and leaves the idle state alone.
            _apiStateEpoch++;

            // Cancel the pipeline's checkpoints so it stops starting new workers/servers instead
            // of continuing after the stop.
            try { _apiLaunchCts?.Cancel(); } catch { }

            StopAllProcesses();
        }

        /// <summary>POST /control/stop: return the API state to idle.</summary>
        internal void ResetApiIdleState() => ResetApiIdleStateCore();

        private void ResetApiIdleStateCore()
        {
            _apiPhase = "idle";
            _apiMessage = "Ready to launch.";
            _apiError = null;
            _apiReady = false;
            _apiRunningModelPath = null;
            _apiRunningModelName = null;
            _apiRunningDevices = null;
            _apiRunningExpertLayers = null;
        }
    }

    public class LlmModel
    {
        public string Name { get; set; } = "";
        public string Path { get; set; } = "";
        public string SizeGB { get; set; } = "";

        // Display-only fields for the setup page's model cards. SizeGiB is the numeric size the page's
        // fit heuristic needs; the rest are parsed from the filename by ParseModelCard.
        public double SizeGiB { get; set; }
        public string Family { get; set; } = "";
        public string Params { get; set; } = "";
        public string Quant { get; set; } = "";
        public List<string> Tags { get; set; } = new();
    }
}
