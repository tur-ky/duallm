using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Threading;

namespace DualLlmPortal
{
    /// <summary>Body of POST /control/launch. Every field is optional.</summary>
    internal sealed class ApiLaunchOptions
    {
        public string? Path { get; set; }
        public JsonElement? Devices { get; set; }
        public int? ContextSize { get; set; }

        /// <summary>{"&lt;ram device id&gt;": layers}: MoE expert layers to place on each RAM device.</summary>
        public JsonElement? ExpertLayers { get; set; }
    }

    /// <summary>
    /// Loopback HTTP control API (default 127.0.0.1:8091), documented in docs/CONTROL_API.md. It exists
    /// so another program can read the portal's model/device state and ask it to load a GGUF across a
    /// chosen set of devices without driving the setup page.
    ///
    /// All state reads and mutations are marshalled onto the WPF Dispatcher, because the state
    /// lives in MainWindow's fields (the setup page's C# mirror) and is only safe to touch there.
    /// Everything is exception-safe: a failed bind or a malformed request must never take the
    /// portal down, and unknown paths answer JSON 404.
    /// </summary>
    internal sealed class ControlApi : IDisposable
    {
        private const int ApiVersion = 1;

        private readonly MainWindow _owner;
        private readonly Dispatcher _dispatcher;
        private readonly int _port;
        private readonly CancellationTokenSource _cts = new();

        private HttpListener? _listener;

        // Only ever used for the optional health probe behind /control/state when phase=ready.
        private static readonly HttpClient HealthClient = new() { Timeout = TimeSpan.FromSeconds(1) };

        internal ControlApi(MainWindow owner, Dispatcher dispatcher, int port)
        {
            _owner = owner;
            _dispatcher = dispatcher;
            _port = port;
        }

        /// <summary>Starts the listener. Never throws: a control API that cannot bind must not
        /// take down the portal, which is still usable from its own setup page.</summary>
        internal void Start()
        {
            try
            {
                var listener = new HttpListener();
                // The prefix must end with '/' for HttpListener to accept it.
                listener.Prefixes.Add($"http://127.0.0.1:{_port}/");
                listener.Start();
                _listener = listener;
                _ = Task.Run(() => AcceptLoopAsync(listener, _cts.Token));
                Log($"control API listening on http://127.0.0.1:{_port}/");
            }
            catch (Exception ex)
            {
                try { _listener?.Close(); } catch { }
                _listener = null;
                Log($"control API could not bind 127.0.0.1:{_port}: {ex.Message}");
            }
        }

        internal void Stop()
        {
            try { _cts.Cancel(); } catch { }
            try { _listener?.Stop(); } catch { }
            try { _listener?.Close(); } catch { }
            _listener = null;
        }

        public void Dispose() => Stop();

        // Diagnostics only: one line per listener start/failure, next to portal-llama-server.log.
        private static void Log(string message)
        {
            try
            {
                File.AppendAllText(
                    Path.Combine(AppHome.Directory, "portal-control-api.log"),
                    $"{DateTime.Now:yyyy-MM-dd HH:mm:ss} {message}{Environment.NewLine}");
            }
            catch { }
        }

        // ---------------------------------------------------------------- listener loop

        private async Task AcceptLoopAsync(HttpListener listener, CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                HttpListenerContext context;
                try
                {
                    context = await listener.GetContextAsync();
                }
                catch
                {
                    break; // listener stopped, or the process is shutting down
                }

                _ = Task.Run(() => HandleContextAsync(context));
            }
        }

        private async Task HandleContextAsync(HttpListenerContext context)
        {
            try
            {
                var method = (context.Request.HttpMethod ?? "GET").ToUpperInvariant();
                var path = context.Request.Url?.AbsolutePath ?? "/";

                string body = "";
                if (context.Request.HasEntityBody)
                {
                    using var reader = new StreamReader(context.Request.InputStream, Encoding.UTF8);
                    body = await reader.ReadToEndAsync();
                }

                var (status, json) = await DispatchAsync(method, path, body);
                var bytes = Encoding.UTF8.GetBytes(json);

                context.Response.StatusCode = status;
                if (status != 204)
                {
                    context.Response.ContentType = "application/json; charset=utf-8";
                    context.Response.ContentLength64 = bytes.Length;
                    await context.Response.OutputStream.WriteAsync(bytes, 0, bytes.Length);
                }
            }
            catch
            {
                try
                {
                    var bytes = Encoding.UTF8.GetBytes(ErrorJson("internal control API error"));
                    context.Response.StatusCode = 500;
                    context.Response.ContentType = "application/json; charset=utf-8";
                    context.Response.ContentLength64 = bytes.Length;
                    context.Response.OutputStream.Write(bytes, 0, bytes.Length);
                }
                catch { }
            }
            finally
            {
                try { context.Response.Close(); } catch { }
            }
        }

        // ---------------------------------------------------------------- routing

        private async Task<(int Status, string Json)> DispatchAsync(string method, string path, string body)
        {
            try
            {
                if (method == "OPTIONS")
                {
                    return (204, "");
                }

                switch (path)
                {
                    case "/control/health":
                        if (method != "GET")
                        {
                            return (405, ErrorJson($"method {method} is not allowed for {path}"));
                        }
                        return (200, JsonSerializer.Serialize(new { ok = true, apiVersion = ApiVersion }));

                    case "/control/state":
                        if (method != "GET")
                        {
                            return (405, ErrorJson($"method {method} is not allowed for {path}"));
                        }
                        return (200, await BuildStateJsonAsync());

                    case "/control/devices":
                        if (method != "POST")
                        {
                            return (405, ErrorJson($"method {method} is not allowed for {path}"));
                        }
                        return await HandleDevicesAsync(body);

                    case "/control/select":
                        if (method != "POST")
                        {
                            return (405, ErrorJson($"method {method} is not allowed for {path}"));
                        }
                        return await HandleSelectAsync(body);

                    case "/control/launch":
                        if (method != "POST")
                        {
                            return (405, ErrorJson($"method {method} is not allowed for {path}"));
                        }
                        return await HandleLaunchAsync(body);

                    case "/control/stop":
                        if (method != "POST")
                        {
                            return (405, ErrorJson($"method {method} is not allowed for {path}"));
                        }
                        return await HandleStopAsync();

                    case "/control/shutdown":
                        if (method != "POST")
                        {
                            return (405, ErrorJson($"method {method} is not allowed for {path}"));
                        }
                        return await HandleShutdownAsync();

                    default:
                        return (404, ErrorJson($"unknown path: {path}"));
                }
            }
            catch (Exception ex)
            {
                return (500, ErrorJson("control API error: " + ex.Message));
            }
        }

        /// <summary>Builds /control/state on the UI thread, then (only when the launch claims to
        /// be ready) confirms the server really answers /health, so serverReady is not stale.</summary>
        private async Task<string> BuildStateJsonAsync()
        {
            var json = await _dispatcher.InvokeAsync(() => _owner.BuildApiStateJson());

            bool ready;
            try
            {
                using var document = JsonDocument.Parse(json);
                ready = document.RootElement.TryGetProperty("phase", out var phase) &&
                        phase.GetString() == "ready";
            }
            catch
            {
                ready = false;
            }

            if (ready && !await ProbeServerHealthAsync(_owner.ServerPort))
            {
                try
                {
                    var node = JsonNode.Parse(json)!.AsObject();
                    node["serverReady"] = false;
                    json = node.ToJsonString();
                }
                catch { }
            }

            return json;
        }

        private static async Task<bool> ProbeServerHealthAsync(int port)
        {
            try
            {
                using var response = await HealthClient.GetAsync($"http://127.0.0.1:{port}/health");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        private async Task<(int Status, string Json)> HandleDevicesAsync(string body)
        {
            // Device flags are frozen for the duration of a launch: RunApiLaunchAsync pins its
            // --rpc/--device set from a snapshot taken when the launch was accepted, so accepting a
            // change mid-launch would make the reported devices disagree with llama-server's args.
            if (await _dispatcher.InvokeAsync(() => _owner.ApiLaunchBusy))
            {
                return (409, ErrorJson("launch already in progress"));
            }

            if (!TryParseJsonObject(body, out var root))
            {
                return (400, ErrorJson("body must be a JSON object like {\"devices\":{...}}"));
            }

            if (!root.TryGetProperty("devices", out var devices) || devices.ValueKind != JsonValueKind.Object)
            {
                return (400, ErrorJson("body must contain a \"devices\" object"));
            }

            var cloned = devices.Clone();
            var error = await _dispatcher.InvokeAsync(() => _owner.ApplyApiDevices(cloned));
            if (error != null)
            {
                return (400, ErrorJson(error));
            }

            return (200, await BuildStateJsonAsync());
        }

        private async Task<(int Status, string Json)> HandleSelectAsync(string body)
        {
            if (!TryParseJsonObject(body, out var root))
            {
                return (400, ErrorJson("body must be a JSON object like {\"path\":\"...\"}"));
            }

            if (!root.TryGetProperty("path", out var pathElement) ||
                pathElement.ValueKind != JsonValueKind.String ||
                string.IsNullOrWhiteSpace(pathElement.GetString()))
            {
                return (400, ErrorJson("body must contain a non-empty \"path\" string"));
            }

            var path = pathElement.GetString()!;
            var error = await _dispatcher.InvokeAsync(() => _owner.TrySelectApiModel(path));
            if (error != null)
            {
                return (400, ErrorJson(error));
            }

            return (200, await BuildStateJsonAsync());
        }

        private async Task<(int Status, string Json)> HandleLaunchAsync(string body)
        {
            var parseError = TryParseLaunchOptions(body, out var options);
            if (parseError != null)
            {
                return (400, ErrorJson(parseError));
            }

            // TryBeginApiLaunch validates, flips phase to loading and starts the launch task;
            // it returns immediately so the caller gets its 202 while loading continues.
            var (status, error) = await _dispatcher.InvokeAsync(() => _owner.TryBeginApiLaunch(options!));
            if (status == 202)
            {
                return (202, JsonSerializer.Serialize(new { ok = true, accepted = true }));
            }

            return (status, ErrorJson(error ?? "launch could not be started"));
        }

        private async Task<(int Status, string Json)> HandleStopAsync()
        {
            // Stop cancels any in-flight launch: StopProcessesFromApi bumps the launch state epoch
            // and cancels the pipeline's checkpoints before freeing processes, and the pipeline
            // only writes terminal state when its epoch is still current. The idle state below is
            // therefore not overwritten by a cancelled launch. While the cancelled pipeline is
            // still unwinding, /control/launch and /control/devices answer 409 (the busy flag is
            // owned by that pipeline until it has finished cleaning up).
            //
            // StopAllProcesses blocks on process kills and (when a remote worker was used) one ssh
            // round trip per host, so keep it off the UI thread like the setup page's "release GPUs" does.
            await Task.Run(() => _owner.StopProcessesFromApi());
            await _dispatcher.InvokeAsync(() => _owner.ResetApiIdleState());
            return (200, await BuildStateJsonAsync());
        }

        /// <summary>
        /// POST /control/shutdown: release every GPU/worker process, answer, then exit the app.
        /// Lets a client release the GPUs and close the portal in one call; works for both a headless
        /// instance the client spawned and a visible one the user opened.
        /// </summary>
        private async Task<(int Status, string Json)> HandleShutdownAsync()
        {
            await Task.Run(() => _owner.StopProcessesFromApi());

            // Answer first, then leave the message loop shortly after so the response is flushed.
            _ = Task.Run(async () =>
            {
                try
                {
                    await Task.Delay(200);
                    await _dispatcher.InvokeAsync(() =>
                    {
                        try { System.Windows.Application.Current?.Shutdown(); }
                        catch { }
                    });
                }
                catch { }
            });

            return (200, JsonSerializer.Serialize(new { ok = true, shuttingDown = true }));
        }

        // ---------------------------------------------------------------- parsing helpers

        private static string? TryParseLaunchOptions(string body, out ApiLaunchOptions? options)
        {
            options = null;

            // The whole body is optional: an empty body launches the current selection.
            JsonElement root = default;
            if (!string.IsNullOrWhiteSpace(body))
            {
                if (!TryParseJsonObject(body, out root))
                {
                    return "body must be a JSON object";
                }
            }

            var parsed = new ApiLaunchOptions();
            if (root.ValueKind == JsonValueKind.Object)
            {
                if (root.TryGetProperty("path", out var path) && path.ValueKind != JsonValueKind.Null)
                {
                    if (path.ValueKind != JsonValueKind.String)
                    {
                        return "path must be a string";
                    }
                    parsed.Path = path.GetString();
                }

                if (root.TryGetProperty("devices", out var devices) && devices.ValueKind != JsonValueKind.Null)
                {
                    if (devices.ValueKind != JsonValueKind.Object)
                    {
                        return "devices must be an object";
                    }
                    parsed.Devices = devices.Clone();
                }

                var intError = ReadOptionalInt(root, "contextSize", out var contextSize);
                if (intError != null) return intError;
                parsed.ContextSize = contextSize;

                if (root.TryGetProperty("expertLayers", out var expertLayers) && expertLayers.ValueKind != JsonValueKind.Null)
                {
                    if (expertLayers.ValueKind != JsonValueKind.Object)
                    {
                        return "expertLayers must be an object";
                    }
                    parsed.ExpertLayers = expertLayers.Clone();
                }
            }

            options = parsed;
            return null;
        }

        private static string? ReadOptionalInt(JsonElement root, string name, out int? value)
        {
            value = null;
            if (!root.TryGetProperty(name, out var element) || element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }

            if (element.ValueKind != JsonValueKind.Number || !element.TryGetInt32(out var number))
            {
                return $"{name} must be an integer";
            }

            value = number;
            return null;
        }

        private static bool TryParseJsonObject(string body, out JsonElement root)
        {
            root = default;
            if (string.IsNullOrWhiteSpace(body))
            {
                return false;
            }

            try
            {
                using var document = JsonDocument.Parse(body);
                if (document.RootElement.ValueKind != JsonValueKind.Object)
                {
                    return false;
                }

                // Clone: the JsonDocument is disposed when this method returns.
                root = document.RootElement.Clone();
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static string ErrorJson(string message)
            => JsonSerializer.Serialize(new { ok = false, error = message });
    }
}
