import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';

const BUS_NAME = 'io.github.nsg.Singstone';
const RECORDER_PATH = '/io/github/nsg/Singstone/Recorder';
const APPLICATION_PATH = '/io/github/nsg/Singstone';

const RECORDER_XML = `
<node>
  <interface name="io.github.nsg.Singstone.Recorder">
    <method name="StartRecording"/>
    <method name="StopRecording"/>
    <method name="GetStatus">
      <arg type="a{sv}" name="status" direction="out"/>
    </method>
    <method name="Quit"/>
    <signal name="StatusChanged">
      <arg type="a{sv}" name="status"/>
    </signal>
  </interface>
</node>
`;

function idleStatus() {
    return {
        recording: false,
        stopping: false,
        mic: false,
        system: false,
        mic_level: 0,
        system_level: 0,
        elapsed: 0,
        screenshots: 0,
    };
}

function unpack(value) {
    return value && typeof value.deepUnpack === 'function'
        ? value.deepUnpack()
        : value;
}

export const RecorderClient = GObject.registerClass({
    GTypeName: 'SingstoneRecorderClient',
    Signals: {
        'available-changed': {param_types: [GObject.TYPE_BOOLEAN]},
        'status-changed': {},
    },
}, class RecorderClient extends GObject.Object {
    _init({onError = () => {}, launch = null} = {}) {
        super._init();

        this.status = idleStatus();
        this.available = false;
        this._onError = onError;
        this._launch = launch;
        this._proxyClass = Gio.DBusProxy.makeProxyWrapper(RECORDER_XML);
        this._proxy = null;
        this._proxySignalId = 0;
        this._proxyCancellable = null;
        this._connection = null;
        this._proxyGeneration = 0;
        this._startPending = false;
        this._startWatchdogId = 0;
        this._timeoutIds = new Set();
        this._unavailableWaiters = new Set();
        this._destroyed = false;

        this._watchId = Gio.bus_watch_name(
            Gio.BusType.SESSION,
            BUS_NAME,
            Gio.BusNameWatcherFlags.NONE,
            (connection, _name, _owner) => this._nameAppeared(connection),
            () => this._nameVanished()
        );
    }

    start() {
        if (this._destroyed)
            return;

        if (this._proxy) {
            this._startRecording(this._proxy, []);
            return;
        }

        if (this._startPending)
            return;

        this._startPending = true;
        this._startWatchdogId = this._addTimeout(20_000, () => {
            this._startWatchdogId = 0;
            if (!this._startPending)
                return;

            this._startPending = false;
            this._reportError('Singstone did not start');
        });
        if (this.available) {
            if (!this._proxyCancellable)
                this._createProxy(this._connection);
            return;
        }
        if (!this._launchApp()) {
            this._startPending = false;
            this._clearStartWatchdog();
        }
    }

    stop() {
        if (!this._proxy || this._destroyed)
            return;

        this._proxy.StopRecordingRemote((_result, error) => {
            if (error)
                this._reportError(error);
        });
    }

    quit() {
        if (this._destroyed || !this._proxy)
            return;

        this._proxy.QuitRemote((_result, error) => {
            if (error)
                this._reportError(error);
        });
    }

    waitUntilUnavailable(timeoutMs) {
        if (this._destroyed)
            return Promise.reject(new Error('Recorder client was destroyed'));
        if (!this.available)
            return Promise.resolve();

        return new Promise((resolve, reject) => {
            const waiter = {signalId: 0, timeoutId: 0, resolve, reject};
            waiter.signalId = this.connect(
                'available-changed', (_source, available) => {
                    if (!available)
                        this._settleUnavailableWaiter(waiter);
                }
            );
            waiter.timeoutId = this._addTimeout(timeoutMs, () => {
                waiter.timeoutId = 0;
                this._settleUnavailableWaiter(
                    waiter,
                    new Error('Singstone did not quit')
                );
            });
            this._unavailableWaiters.add(waiter);
        });
    }

    openApp() {
        if (this._destroyed)
            return;

        if (!this.available) {
            this._launchApp();
            return;
        }

        this._connection.call(
            BUS_NAME,
            APPLICATION_PATH,
            'org.gtk.Application',
            'Activate',
            new GLib.Variant('(a{sv})', [{}]),
            null,
            Gio.DBusCallFlags.NONE,
            -1,
            null,
            null
        );
    }

    destroy() {
        if (this._destroyed)
            return;

        this._destroyed = true;
        for (const waiter of [...this._unavailableWaiters]) {
            this._settleUnavailableWaiter(
                waiter,
                new Error('Recorder client was destroyed')
            );
        }
        this._startPending = false;
        this._clearStartWatchdog();
        this._proxyGeneration++;
        if (this._watchId) {
            Gio.bus_unwatch_name(this._watchId);
            this._watchId = 0;
        }
        this._clearTimeouts();
        this._cancelProxyConstruction();
        this._dropProxy();
        this._connection = null;
        this._onError = null;
        this._launch = null;
        this._proxyClass = null;
    }

    _nameAppeared(connection) {
        if (this._destroyed)
            return;

        this._connection = connection;
        this._setAvailable(true);
        this._createProxy(connection);
    }

    _createProxy(connection) {
        if (this._destroyed || !connection)
            return;

        this._cancelProxyConstruction();
        this._dropProxy();
        const generation = ++this._proxyGeneration;
        const cancellable = new Gio.Cancellable();
        this._proxyCancellable = cancellable;
        new this._proxyClass(connection, BUS_NAME, RECORDER_PATH, (proxy, error) => {
            if (this._destroyed || generation !== this._proxyGeneration)
                return;
            this._proxyCancellable = null;

            if (error) {
                this._startPending = false;
                this._clearStartWatchdog();
                this._reportError(error);
                return;
            }

            this._proxy = proxy;
            this._proxySignalId = proxy.connectSignal(
                'StatusChanged',
                (_source, _sender, [status]) => this._applyStatus(status)
            );
            this._fetchStatus(proxy);

            if (this._startPending) {
                this._startPending = false;
                this._clearStartWatchdog();
                this._startRecording(proxy, [250, 500, 1000, 2000, 4000]);
            }
        }, cancellable,
        Gio.DBusProxyFlags.DO_NOT_LOAD_PROPERTIES |
            Gio.DBusProxyFlags.DO_NOT_AUTO_START);
    }

    _nameVanished() {
        if (this._destroyed)
            return;

        this._startPending = false;
        this._clearStartWatchdog();
        this._proxyGeneration++;
        this._clearTimeouts();
        this._cancelProxyConstruction();
        this._dropProxy();
        this._connection = null;
        this._setAvailable(false);
        this.status = idleStatus();
        this.emit('status-changed');
    }

    _setAvailable(available) {
        if (this.available === available)
            return;

        this.available = available;
        this.emit('available-changed', available);
    }

    _dropProxy() {
        if (this._proxy && this._proxySignalId)
            this._proxy.disconnectSignal(this._proxySignalId);
        this._proxySignalId = 0;
        this._proxy = null;
    }

    _cancelProxyConstruction() {
        this._proxyCancellable?.cancel();
        this._proxyCancellable = null;
    }

    _fetchStatus(proxy) {
        proxy.GetStatusRemote((result, error) => {
            if (this._destroyed || this._proxy !== proxy)
                return;
            if (error)
                return;

            this._applyStatus(result[0]);
        });
    }

    _applyStatus(statusVariant) {
        if (this._destroyed)
            return;

        const raw = unpack(statusVariant) ?? {};
        const value = key => unpack(raw[key]);
        this.status = {
            recording: Boolean(value('recording')),
            stopping: Boolean(value('stopping')),
            mic: Boolean(value('mic')),
            system: Boolean(value('system')),
            mic_level: Number(value('mic_level')),
            system_level: Number(value('system_level')),
            elapsed: Number(value('elapsed')),
            screenshots: Number(value('screenshots')),
        };
        this.emit('status-changed');
    }

    _startRecording(proxy, retryDelays) {
        if (this._destroyed || this._proxy !== proxy)
            return;

        proxy.StartRecordingRemote((_result, error) => {
            if (this._destroyed || this._proxy !== proxy || !error)
                return;

            if (retryDelays.length > 0 && this._isStartupRace(error)) {
                const [delay, ...remainingDelays] = retryDelays;
                this._addTimeout(delay, () =>
                    this._startRecording(proxy, remainingDelays));
                return;
            }

            this._reportError(error);
        });
    }

    _isStartupRace(error) {
        const remoteName = Gio.DBusError.get_remote_error(error) ?? '';

        return remoteName === 'org.freedesktop.DBus.Error.UnknownObject' ||
            remoteName === 'org.freedesktop.DBus.Error.UnknownMethod' ||
            remoteName === 'org.freedesktop.DBus.Error.UnknownInterface';
    }

    _launchApp() {
        let launchError = null;
        const desktopIds = [
            'singstone_singstone.desktop',
            'io.github.nsg.Singstone.desktop',
        ];
        if (this._launch) {
            for (const desktopId of desktopIds) {
                try {
                    if (this._launch(desktopId))
                        return true;
                } catch (error) {
                    launchError = error;
                }
            }
        } else {
            for (const desktopId of desktopIds) {
                const appInfo = Gio.DesktopAppInfo.new(desktopId);
                if (!appInfo)
                    continue;

                try {
                    appInfo.launch([], null);
                    return true;
                } catch (error) {
                    launchError = error;
                }
            }
        }

        this._reportError(launchError ?? new Error('Singstone is not installed.'));
        return false;
    }

    _reportError(error) {
        if (!this._onError || this._destroyed)
            return;

        const message = `${error.message ?? error}`
            .replace(/^GDBus\.Error:[^:]+:\s*/, '');
        this._onError(message);
    }

    _addTimeout(milliseconds, callback) {
        let sourceId = 0;
        sourceId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, milliseconds, () => {
            this._timeoutIds.delete(sourceId);
            callback();
            return GLib.SOURCE_REMOVE;
        });
        this._timeoutIds.add(sourceId);
        return sourceId;
    }

    _clearStartWatchdog() {
        if (!this._startWatchdogId)
            return;

        GLib.Source.remove(this._startWatchdogId);
        this._timeoutIds.delete(this._startWatchdogId);
        this._startWatchdogId = 0;
    }

    _clearTimeouts() {
        for (const sourceId of this._timeoutIds)
            GLib.Source.remove(sourceId);
        this._timeoutIds.clear();
        for (const waiter of this._unavailableWaiters)
            waiter.timeoutId = 0;
    }

    _settleUnavailableWaiter(waiter, error = null) {
        if (!this._unavailableWaiters.delete(waiter))
            return;

        if (waiter.signalId)
            this.disconnect(waiter.signalId);
        if (waiter.timeoutId) {
            GLib.Source.remove(waiter.timeoutId);
            this._timeoutIds.delete(waiter.timeoutId);
        }
        if (error)
            waiter.reject(error);
        else
            waiter.resolve();
    }
});
