using System.Windows;

namespace DualLlmPortal;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    /// <summary>
    /// Creates the portal window, or runs it windowless when another program spawns it with
    /// <c>--headless</c>. Headless mode exists so model/device selection can be driven through the
    /// control API (docs/CONTROL_API.md) without the portal window ever appearing on the desktop.
    /// </summary>
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        var headless = e.Args.Any(argument =>
            string.Equals(argument, "--headless", StringComparison.OrdinalIgnoreCase));

        var window = new MainWindow(headless);
        MainWindow = window;

        if (headless)
        {
            // No window is ever shown; keep the message loop alive until the control API
            // asks for shutdown (or the process is killed).
            ShutdownMode = ShutdownMode.OnExplicitShutdown;
        }
        else
        {
            window.Show();
        }
    }
}
