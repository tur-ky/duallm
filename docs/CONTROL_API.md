# Control API

A small HTTP API on the loopback interface lets another program read the portal's state and load or unload
models without driving the setup page. It works both with the portal's window open and with a headless
instance (`DualLlmPortal.exe --headless`, no window and no WebView2).

- Address: `http://127.0.0.1:<port>/control/...`, port from `controlApi.port` (default `8091`).
- Disable with `"controlApi": { "enabled": false }`.
- Bound to loopback only, with no authentication: any local program can call it.
- Bodies and replies are JSON. Every reply carries `apiVersion` (currently `1`) or is an `{ "ok": false, "error": "..." }` object.
- If the port cannot be bound, the portal still runs; the reason is written to `portal-control-api.log` in the home folder.

Device ids and expert layers are keyed by the ids in your `duallm.config.json`, so nothing here is specific to one setup.

## Endpoints

### `GET /control/health`

`200 {"ok": true, "apiVersion": 1}`. Use it to wait for a freshly started portal.

### `GET /control/state`

```jsonc
{
  "ok": true, "apiVersion": 1,
  "phase": "idle",                // idle | loading | ready | error
  "message": "Ready to launch.",  // human-readable status
  "error": null,                  // set when phase is "error"
  "headless": true,
  "models": [ { "path": "...", "name": "...", "family": "...", "params": "8B", "quant": "Q4_K_M", "sizeGB": 4.3 } ],
  "selectedPath": "S:\\models\\a.gguf",
  "deviceCatalog": [ { "id": "local-gpu", "type": "gpu", "label": "Local GPU", "host": null } ],
  "devices": { "local-gpu": true, "worker-ram": false },    // ticked state per device id
  "expertLayers": { "worker-ram": 0 },                      // MoE expert layers per RAM device id
  "running": { "modelPath": "...", "modelName": "...", "contextSize": 8192, "devices": { }, "expertLayers": { } },  // null unless loaded
  "serverReady": false            // true only while llama-server answers its own /health
}
```

### `POST /control/devices`

`{ "devices": { "<device id>": true|false, ... } }` ticks and unticks devices. Devices you leave out keep
their state. Replies with the new state. `400` for an unknown id or a value that is not a boolean.
`409` while a launch is in progress.

### `POST /control/select`

`{ "path": "<absolute path of a scanned .gguf>" }` selects the model. `400` if the path is not one of `models`.

### `POST /control/launch`

Starts loading and returns immediately with `202 {"ok": true, "accepted": true}`; poll `/control/state`
until `phase` is `ready` (and `serverReady` is true) or `error`. Every field is optional; an empty body
launches the current selection.

```json
{
  "path": "S:\\models\\a.gguf",
  "devices": { "local-gpu": true, "worker-gpu0": true, "worker-ram": false },
  "contextSize": 32768,
  "expertLayers": { "worker-ram": 12 }
}
```

`devices` sets the listed devices and leaves the others as they are. `contextSize` defaults to the setup
page's current choice, and becomes it. `expertLayers` sets how many MoE expert layers go to each RAM device.
Launches from the API always enable `--jinja` (tool calling) and do not start the optional `integrations`. Errors: `400` (no model, unknown path or device, context size
too small, a selection that cannot be launched), `409` (a launch is already in progress).

### `POST /control/stop`

Stops llama-server and every remote worker the portal started, frees the VRAM, and returns the idle state.
Also cancels a launch that is still loading.

### `POST /control/shutdown`

Same as stop, then closes the portal. Replies `{"ok": true, "shuttingDown": true}` before exiting. Use it for
a headless instance you started.

## Example

```powershell
$api = 'http://127.0.0.1:8091/control'
Invoke-RestMethod "$api/select" -Method Post -ContentType application/json `
  -Body (@{ path = 'S:\models\a.gguf' } | ConvertTo-Json)
Invoke-RestMethod "$api/launch" -Method Post -ContentType application/json `
  -Body (@{ devices = @{ 'local-gpu' = $true; 'worker-gpu0' = $true }; contextSize = 8192 } | ConvertTo-Json)
do { Start-Sleep 3; $s = Invoke-RestMethod "$api/state" } while ($s.phase -eq 'loading')
$s.phase, $s.message
# chat at http://127.0.0.1:8080/v1/chat/completions (OpenAI-compatible)
Invoke-RestMethod "$api/stop" -Method Post | Out-Null
```

## Running headless

```powershell
$env:DUALLM_HOME = 'D:\duallm'                 # optional: home folder (config, models, logs)
Start-Process .\DualLlmPortal.exe -ArgumentList '--headless' -WindowStyle Hidden
```

A headless portal logs to the same files in the home folder and exits on `POST /control/shutdown`.
