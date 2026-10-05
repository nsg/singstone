import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Soup from 'gi://Soup?version=3.0';

Gio._promisify(Soup.Session.prototype, 'send_async', 'send_finish');
Gio._promisify(Gio.InputStream.prototype, 'read_bytes_async',
    'read_bytes_finish');

const REPO = 'nsg/singstone';
const RELEASE_URL = `https://api.github.com/repos/${REPO}/releases/tags/latest`;
const SNAP_NAME = 'singstone';
const SNAP_MOUNT_ROOTS = ['/snap', '/var/lib/snapd/snap'];
const USER_AGENT = 'singstone-gnome-shell-extension';
const UPDATE_SCRIPT = 'update.sh';
// Tried in order; each is the argv prefix that runs a command in a new window.
const TERMINALS = [
    ['xdg-terminal-exec'],
    ['ptyxis', '--'],
    ['kgx', '-e'],
    ['gnome-terminal', '--'],
    ['x-terminal-emulator', '-e'],
    ['xterm', '-e'],
];
const FIRST_CHECK_DELAY = 60;
const CHECK_INTERVAL = 6 * 60 * 60;
const STALE_AFTER = 15 * 60 * GLib.USEC_PER_SEC;

export function commitsMatch(a, b) {
    if (typeof a !== 'string' || typeof b !== 'string' || !a || !b)
        return false;

    const left = a.toLowerCase();
    const right = b.toLowerCase();
    const [shorter, longer] = left.length <= right.length
        ? [left, right]
        : [right, left];

    return shorter.length >= 7 && longer.startsWith(shorter);
}

function shortCommit(commit) {
    return commit?.slice(0, 7) ?? 'unknown';
}

function bytesToString(bytes) {
    return new TextDecoder().decode(bytes);
}

export const UpdateManager = GObject.registerClass({
    GTypeName: 'SingstoneUpdateManager',
    Signals: {
        'changed': {},
    },
}, class UpdateManager extends GObject.Object {
    _init({
        extensionPath,
        metadata = {},
        notify = () => {},
        autoCheck = true,
        snapMountRoots = SNAP_MOUNT_ROOTS,
        userExtensionsDir = GLib.build_filenamev([
            GLib.get_user_data_dir(),
            'gnome-shell',
            'extensions',
        ]),
        terminals = TERMINALS,
    } = {}) {
        super._init();

        this.remote = null;
        this.installed = {snap: null, extension: null};
        this.snapInstalled = false;
        this.extensionRestartPending = false;
        this.checking = false;
        this.lastChecked = 0;
        this.lastError = null;

        this._extensionPath = extensionPath;
        this._metadata = metadata;
        this._notify = notify;
        this._snapMountRoots = [...snapMountRoots];
        this._userExtensionsDir = userExtensionsDir;
        this._terminals = terminals;
        this._cancellable = new Gio.Cancellable();
        this._session = new Soup.Session({
            user_agent: `${USER_AGENT} `,
            timeout: 60,
        });
        this._checkPromise = null;
        this._firstCheckId = 0;
        this._checkIntervalId = 0;
        this._destroyed = false;

        this.refreshInstalled();

        if (autoCheck) {
            this._firstCheckId = GLib.timeout_add_seconds(
                GLib.PRIORITY_LOW,
                FIRST_CHECK_DELAY,
                () => {
                    this._firstCheckId = 0;
                    if (!this._destroyed)
                        this.check();
                    return GLib.SOURCE_REMOVE;
                }
            );
            this._checkIntervalId = GLib.timeout_add_seconds(
                GLib.PRIORITY_LOW,
                CHECK_INTERVAL,
                () => {
                    if (!this._destroyed)
                        this.check();
                    return GLib.SOURCE_CONTINUE;
                }
            );
        }
    }

    get snapUpdateAvailable() {
        return typeof this.remote?.commit === 'string' &&
            !commitsMatch(this.installed.snap, this.remote.commit) &&
            typeof this.installed.snap === 'string' &&
            this.installed.snap.length > 0;
    }

    get extensionUpdateAvailable() {
        return typeof this.remote?.commit === 'string' &&
            !commitsMatch(this.installed.extension, this.remote.commit) &&
            typeof this.installed.extension === 'string' &&
            this.installed.extension.length > 0;
    }

    get updateAvailable() {
        return this.snapUpdateAvailable || this.extensionUpdateAvailable;
    }

    refreshInstalled() {
        if (this._destroyed)
            return;

        let snapInstalled = false;
        let snapCommit = null;
        for (const root of this._snapMountRoots) {
            const path = GLib.build_filenamev([
                root, SNAP_NAME, 'current', 'meta', 'snap.yaml',
            ]);
            try {
                const [ok, contents] = GLib.file_get_contents(path);
                if (!ok)
                    continue;

                snapInstalled = true;
                snapCommit = this._snapCommit(bytesToString(contents));
                break;
            } catch (_error) {
                // Try the next standard snap mount root.
            }
        }

        const uuid = typeof this._metadata?.uuid === 'string' &&
            this._metadata.uuid
            ? this._metadata.uuid
            : 'singstone@nsg.github.io';
        const installMetadataPath = GLib.build_filenamev([
            this._userExtensionsDir, uuid, 'metadata.json',
        ]);
        let extensionCommit = null;
        let metadataPath = installMetadataPath;
        if (!GLib.file_test(metadataPath, GLib.FileTest.EXISTS) &&
            this._extensionPath) {
            metadataPath = GLib.build_filenamev([
                this._extensionPath, 'metadata.json',
            ]);
        }
        if (GLib.file_test(metadataPath, GLib.FileTest.EXISTS)) {
            try {
                const [ok, contents] = GLib.file_get_contents(metadataPath);
                if (ok) {
                    const metadata = JSON.parse(bytesToString(contents));
                    if (typeof metadata.commit === 'string' && metadata.commit)
                        extensionCommit = metadata.commit;
                }
            } catch (_error) {
                // Fall back to the metadata loaded with the running extension.
            }
        }
        if (!extensionCommit &&
            typeof this._metadata?.commit === 'string' &&
            this._metadata.commit)
            extensionCommit = this._metadata.commit;

        const runningCommit = typeof this._metadata?.commit === 'string' &&
            this._metadata.commit
            ? this._metadata.commit
            : null;
        const extensionRestartPending = Boolean(
            extensionCommit && runningCommit &&
            !commitsMatch(extensionCommit, runningCommit)
        );

        const changed = this.snapInstalled !== snapInstalled ||
            this.installed.snap !== snapCommit ||
            this.installed.extension !== extensionCommit ||
            this.extensionRestartPending !== extensionRestartPending;
        if (!changed)
            return;

        this.snapInstalled = snapInstalled;
        this.extensionRestartPending = extensionRestartPending;
        this.installed = {snap: snapCommit, extension: extensionCommit};
        this._emitChanged();
    }

    async check({manual = false} = {}) {
        if (this._destroyed)
            return;
        if (this._checkPromise)
            return this._checkPromise;

        this.checking = true;
        this._emitChanged();
        this.refreshInstalled();
        this._checkPromise = this._performCheck(manual);
        return this._checkPromise;
    }

    checkIfStale() {
        if (this._destroyed)
            return;

        // The update runs in a terminal, so nothing else reports its result.
        this.refreshInstalled();
        const age = GLib.get_monotonic_time() - this.lastChecked;
        if (this.lastChecked === 0 || age >= STALE_AFTER)
            this.check();
    }

    // Downloading and installing happen in a terminal running update.sh: it
    // shows its own progress and asks for the password there.
    update() {
        if (this._destroyed)
            return;

        const script = GLib.build_filenamev([
            this._extensionPath, UPDATE_SCRIPT,
        ]);
        const terminal = this._terminals.find(
            ([program]) => GLib.find_program_in_path(program) !== null
        );
        if (!terminal) {
            this._notify(
                `No terminal found. Run this to update: bash ${script}`
            );
            return;
        }

        try {
            Gio.Subprocess.new(
                [...terminal, 'bash', script],
                Gio.SubprocessFlags.NONE
            );
        } catch (error) {
            this._notify(
                `Could not open a terminal: ${this._errorMessage(error)}`
            );
        }
    }

    destroy() {
        if (this._destroyed)
            return;

        this._destroyed = true;
        this._cancellable.cancel();
        if (this._firstCheckId) {
            GLib.source_remove(this._firstCheckId);
            this._firstCheckId = 0;
        }
        if (this._checkIntervalId) {
            GLib.source_remove(this._checkIntervalId);
            this._checkIntervalId = 0;
        }

        this._checkPromise = null;
        this._session = null;
        this._cancellable = null;
        this._extensionPath = null;
        this._metadata = null;
        this._notify = null;
        this._snapMountRoots = null;
        this._userExtensionsDir = null;
        this._terminals = null;
    }

    async _performCheck(manual) {
        try {
            const message = Soup.Message.new('GET', RELEASE_URL);
            const headers = message.get_request_headers();
            headers.append('Accept', 'application/vnd.github+json');
            headers.append('X-GitHub-Api-Version', '2022-11-28');
            headers.append('User-Agent', USER_AGENT);

            const input = await this._session.send_async(
                message,
                GLib.PRIORITY_DEFAULT,
                this._cancellable
            );
            if (this._destroyed)
                return;

            const status = message.get_status();
            if (status !== 200) {
                const remaining = message.get_response_headers()
                    .get_one('x-ratelimit-remaining');
                const rateLimited = (status === 403 || status === 429) &&
                    remaining === '0';
                throw new Error(rateLimited
                    ? `GitHub API rate limit (HTTP status ${status})`
                    : `GitHub API returned HTTP status ${status}`);
            }

            const body = await this._readAll(input);
            if (this._destroyed)
                return;
            const release = JSON.parse(bytesToString(body));
            const commit = typeof release.target_commitish === 'string' &&
                /^[0-9a-f]{7,40}$/i.test(release.target_commitish)
                ? release.target_commitish
                : null;
            this.remote = {commit};
            this.lastError = null;

            if (manual) {
                const parts = [];
                if (this.snapInstalled && this.installed.snap === null)
                    parts.push('Singstone version unknown');
                else if (this.snapUpdateAvailable)
                    parts.push(`Singstone ${shortCommit(commit)} available`);
                if (this.installed.extension === null)
                    parts.push('Extension version unknown');
                else if (this.extensionUpdateAvailable)
                    parts.push(`Extension ${shortCommit(commit)} available`);
                this._notify(parts.length > 0
                    ? parts.join('. ')
                    : 'Singstone and the extension are up to date');
            }
        } catch (error) {
            if (this._destroyed)
                return;

            const message = this._errorMessage(error);
            this.lastError = message;
            if (manual)
                this._notify(`Update check failed: ${message}`);
        } finally {
            if (!this._destroyed) {
                this.checking = false;
                this.lastChecked = GLib.get_monotonic_time();
                this._checkPromise = null;
                this._emitChanged();
            }
        }
    }

    async _readAll(input) {
        const chunks = [];
        let length = 0;
        while (true) {
            const bytes = await input.read_bytes_async(
                64 * 1024,
                GLib.PRIORITY_DEFAULT,
                this._cancellable
            );
            if (this._destroyed)
                throw new Error('Update manager was destroyed');

            const size = bytes.get_size();
            if (size === 0)
                break;
            const chunk = bytes.get_data();
            chunks.push(chunk);
            length += size;
        }

        const body = new Uint8Array(length);
        let offset = 0;
        for (const chunk of chunks) {
            body.set(chunk, offset);
            offset += chunk.length;
        }
        return body;
    }

    _snapCommit(contents) {
        const match = contents.match(/^version:\s*(.*?)\s*$/m);
        if (!match)
            return null;

        let version = match[1];
        if ((version.startsWith("'") && version.endsWith("'")) ||
            (version.startsWith('"') && version.endsWith('"')))
            version = version.slice(1, -1);
        const marker = version.indexOf('+git.');
        if (marker < 0)
            return null;
        return version.slice(marker + 5) || null;
    }

    _errorMessage(error) {
        return error?.message ?? `${error}`;
    }

    _emitChanged() {
        if (!this._destroyed)
            this.emit('changed');
    }
});
