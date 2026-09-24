import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Soup from 'gi://Soup?version=3.0';

Gio._promisify(Soup.Session.prototype, 'send_async', 'send_finish');
Gio._promisify(Gio.File.prototype, 'replace_async', 'replace_finish');
Gio._promisify(Gio.File.prototype, 'load_contents_async',
    'load_contents_finish');
Gio._promisify(Gio.InputStream.prototype, 'read_bytes_async',
    'read_bytes_finish');
Gio._promisify(Gio.OutputStream.prototype, 'write_bytes_async',
    'write_bytes_finish');
Gio._promisify(Gio.OutputStream.prototype, 'close_async', 'close_finish');
Gio._promisify(Gio.Subprocess.prototype, 'communicate_utf8_async',
    'communicate_utf8_finish');

const REPO = 'nsg/singstone';
const RELEASE_URL = `https://api.github.com/repos/${REPO}/releases/tags/latest`;
const SNAP_NAME = 'singstone';
const SNAP_ASSET = 'singstone_amd64.snap';
const EXTENSION_ASSET = 'singstone-gnome-shell-extension.zip';
const SNAP_MOUNT_ROOTS = ['/snap', '/var/lib/snapd/snap'];
const USER_AGENT = 'singstone-gnome-shell-extension';
const FIRST_CHECK_DELAY = 60;
const CHECK_INTERVAL = 6 * 60 * 60;
const STALE_AFTER = 15 * 60 * GLib.USEC_PER_SEC;
const DOWNLOAD_CHUNK_SIZE = 256 * 1024;
const DOWNLOAD_STALL_TIMEOUT = 60;
const DOWNLOAD_MAX_ATTEMPTS = 5;
const DOWNLOAD_RETRY_DELAY = 2;
const PROGRESS_INTERVAL = 250 * 1000;

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
        cacheDir = GLib.build_filenamev([
            GLib.get_user_cache_dir(),
            'singstone-gnome-shell-extension',
        ]),
        snapMountRoots = SNAP_MOUNT_ROOTS,
        userExtensionsDir = GLib.build_filenamev([
            GLib.get_user_data_dir(),
            'gnome-shell',
            'extensions',
        ]),
    } = {}) {
        super._init();

        this.remote = null;
        this.installed = {snap: null, extension: null};
        this.snapInstalled = false;
        this.extensionRestartPending = false;
        this.checking = false;
        this.lastChecked = 0;
        this.lastError = null;
        this.snapTask = null;
        this.extensionTask = null;

        this._extensionPath = extensionPath;
        this._metadata = metadata;
        this._notify = notify;
        this._cacheDir = cacheDir;
        this._snapMountRoots = [...snapMountRoots];
        this._userExtensionsDir = userExtensionsDir;
        this._cancellable = new Gio.Cancellable();
        this._session = new Soup.Session({
            user_agent: `${USER_AGENT} `,
            timeout: 60,
        });
        this._checkPromise = null;
        this._firstCheckId = 0;
        this._checkIntervalId = 0;
        this._processes = new Set();
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

        const age = GLib.get_monotonic_time() - this.lastChecked;
        if (this.lastChecked === 0 || age >= STALE_AFTER)
            this.check();
    }

    async updateSnap({beforeInstall = null, afterInstall = null} = {}) {
        if (this._destroyed || this.snapTask)
            return;

        const path = GLib.build_filenamev([this._cacheDir, SNAP_ASSET]);
        this.snapTask = {phase: 'checking', progress: -1};
        this._emitChanged();

        let context = null;
        try {
            if (this._checkPromise)
                await this._checkPromise;
            else if (!this.remote?.snap)
                await this.check();
            if (this._destroyed)
                return;

            const asset = this.remote?.snap;
            if (!asset) {
                this._notify('Singstone update is not available');
                return;
            }

            const commit = this.remote.commit;
            this.snapTask = {
                phase: 'downloading',
                progress: asset.size > 0 ? 0 : -1,
            };
            this._emitChanged();
            this._ensureCacheDir();
            await this._download(asset, path, progress => {
                if (this._destroyed || !this.snapTask)
                    return;
                this.snapTask = {phase: 'downloading', progress};
                this._emitChanged();
            });
            if (this._destroyed)
                return;

            this.snapTask = {phase: 'stopping', progress: 1};
            this._emitChanged();
            context = beforeInstall ? await beforeInstall() : null;
            if (this._destroyed)
                return;

            this.snapTask = {phase: 'installing', progress: 1};
            this._emitChanged();
            await this._runInstaller([
                'snap', 'install', '--dangerous', path,
            ]);
            if (this._destroyed)
                return;

            GLib.unlink(path);
            this.refreshInstalled();
            afterInstall?.(context);
            this._notify(context
                ? `Singstone updated to ${shortCommit(commit)}`
                : `Singstone updated to ${shortCommit(commit)}. ` +
                    'Start it to use the new version.');
        } catch (error) {
            this._unlink(path);
            if (!this._destroyed) {
                // Reopen the app when it was closed for an install that failed.
                afterInstall?.(context);
                this._notify(
                    `Singstone update failed: ${this._errorMessage(error)}`
                );
            }
        } finally {
            if (!this._destroyed) {
                this.snapTask = null;
                this._emitChanged();
            }
        }
    }

    async updateExtension() {
        if (this._destroyed || this.extensionTask)
            return;

        const path = GLib.build_filenamev([this._cacheDir, EXTENSION_ASSET]);
        this.extensionTask = {phase: 'checking', progress: -1};
        this._emitChanged();

        try {
            if (this._checkPromise)
                await this._checkPromise;
            else if (!this.remote?.extension)
                await this.check();
            if (this._destroyed)
                return;

            const asset = this.remote?.extension;
            if (!asset) {
                this._notify('Extension update is not available');
                return;
            }

            const commit = this.remote.commit;
            this.extensionTask = {
                phase: 'downloading',
                progress: asset.size > 0 ? 0 : -1,
            };
            this._emitChanged();
            this._ensureCacheDir();
            await this._download(asset, path, progress => {
                if (this._destroyed || !this.extensionTask)
                    return;
                this.extensionTask = {phase: 'downloading', progress};
                this._emitChanged();
            });
            if (this._destroyed)
                return;

            this.extensionTask = {phase: 'installing', progress: 1};
            this._emitChanged();
            await this._runInstaller([
                'gnome-extensions', 'install', '--force', path,
            ]);
            if (this._destroyed)
                return;

            GLib.unlink(path);
            this.refreshInstalled();
            this._notify(
                `Extension updated to ${shortCommit(commit)}. ` +
                'Log out and back in to load it.'
            );
        } catch (error) {
            this._unlink(path);
            if (!this._destroyed) {
                this._notify(
                    `Extension update failed: ${this._errorMessage(error)}`
                );
            }
        } finally {
            if (!this._destroyed) {
                this.extensionTask = null;
                this._emitChanged();
            }
        }
    }

    async _download(asset, path, onProgress = () => {}) {
        if (this._destroyed)
            throw new Error('Update manager was destroyed');

        const managerCancellable = this._cancellable;
        const useApiUrl = typeof asset.apiUrl === 'string';
        const url = useApiUrl ? asset.apiUrl : asset.url;
        let size = Number(asset.size);
        const file = Gio.File.new_for_path(path);
        let output = await file.replace_async(
            null,
            false,
            Gio.FileCreateFlags.REPLACE_DESTINATION,
            GLib.PRIORITY_DEFAULT,
            managerCancellable
        );
        let written = 0;
        let lastProgress = GLib.get_monotonic_time();
        let lastError = null;
        try {
            if (this._destroyed)
                throw new Error('Update manager was destroyed');

            for (let number = 1; number <= DOWNLOAD_MAX_ATTEMPTS; number++) {
                const attempt = new Gio.Cancellable();
                const cancellationId = managerCancellable.connect(
                    () => attempt.cancel()
                );
                let watchdogId = 0;
                let stalled = false;
                let retryable = true;
                let complete = false;

                const armWatchdog = () => {
                    if (watchdogId)
                        GLib.source_remove(watchdogId);
                    watchdogId = GLib.timeout_add_seconds(
                        GLib.PRIORITY_DEFAULT,
                        DOWNLOAD_STALL_TIMEOUT,
                        () => {
                            watchdogId = 0;
                            stalled = true;
                            attempt.cancel();
                            return GLib.SOURCE_REMOVE;
                        }
                    );
                };

                try {
                    const message = Soup.Message.new('GET', url);
                    if (typeof message.set_force_http1 === 'function')
                        message.set_force_http1(true);
                    const headers = message.get_request_headers();
                    if (useApiUrl) {
                        headers.append(
                            'Accept', 'application/octet-stream'
                        );
                    }
                    const requestedRange = written > 0;
                    if (requestedRange)
                        headers.append('Range', `bytes=${written}-`);

                    armWatchdog();
                    const input = await this._session.send_async(
                        message,
                        GLib.PRIORITY_DEFAULT,
                        attempt
                    );
                    if (this._destroyed)
                        throw new Error('Update manager was destroyed');

                    const status = message.get_status();
                    if (status !== 200 && status !== 206 &&
                        !(status === 416 && written === size)) {
                        retryable = false;
                        throw new Error(
                            `Download failed with HTTP status ${status}`
                        );
                    }
                    if ((!Number.isFinite(size) || size <= 0) &&
                        number === 1 && status === 200) {
                        size = Number(message.get_response_headers()
                            .get_content_length());
                    }
                    if (!Number.isFinite(size) || size <= 0) {
                        retryable = false;
                        throw new Error('Download size is unknown');
                    }

                    if (status === 416 && written === size) {
                        complete = true;
                    } else if (status === 206) {
                        // The server accepted the requested range.
                    } else if (status === 200 && requestedRange) {
                        output.truncate(0, null);
                        output.seek(0, GLib.SeekType.SET, null);
                        written = 0;
                    }

                    if (number > 1) {
                        onProgress(Math.min(written / size, 1));
                        lastProgress = GLib.get_monotonic_time();
                    }

                    while (!complete) {
                        const bytes = await input.read_bytes_async(
                            DOWNLOAD_CHUNK_SIZE,
                            GLib.PRIORITY_DEFAULT,
                            attempt
                        );
                        if (this._destroyed) {
                            throw new Error(
                                'Update manager was destroyed'
                            );
                        }

                        const chunkSize = bytes.get_size();
                        if (chunkSize === 0) {
                            if (written === size)
                                complete = true;
                            else
                                throw new Error(
                                    `Downloaded ${written} bytes, ` +
                                    `expected ${size}`
                                );
                            break;
                        }
                        armWatchdog();

                        const data = bytes.get_data();
                        let offset = 0;
                        while (offset < chunkSize) {
                            const remaining = offset === 0
                                ? bytes
                                : new GLib.Bytes(data.slice(offset));
                            const count = await output.write_bytes_async(
                                remaining,
                                GLib.PRIORITY_DEFAULT,
                                attempt
                            );
                            if (this._destroyed) {
                                throw new Error(
                                    'Update manager was destroyed'
                                );
                            }
                            if (count <= 0) {
                                throw new Error(
                                    'Download failed while writing the file'
                                );
                            }
                            offset += count;
                            written += count;
                        }

                        const now = GLib.get_monotonic_time();
                        if (now - lastProgress >= PROGRESS_INTERVAL) {
                            onProgress(Math.min(written / size, 1));
                            lastProgress = now;
                        }
                    }
                } catch (error) {
                    if (this._destroyed ||
                        managerCancellable.is_cancelled()) {
                        throw error;
                    }
                    if (!retryable)
                        throw error;
                    lastError = stalled
                        ? new Error(
                            `No download data received for ` +
                            `${DOWNLOAD_STALL_TIMEOUT} seconds`
                        )
                        : error;
                } finally {
                    if (watchdogId)
                        GLib.source_remove(watchdogId);
                    managerCancellable.disconnect(cancellationId);
                }

                if (complete)
                    break;
                if (number === DOWNLOAD_MAX_ATTEMPTS) {
                    throw new Error(
                        `Download stalled after ${DOWNLOAD_MAX_ATTEMPTS} ` +
                        `attempts: ${this._errorMessage(lastError)}`
                    );
                }

                let retryTimeoutId = 0;
                let retryCancellationId = 0;
                try {
                    await new Promise((resolve, reject) => {
                        retryCancellationId = managerCancellable.connect(
                            () => {
                                if (retryTimeoutId) {
                                    GLib.source_remove(retryTimeoutId);
                                    retryTimeoutId = 0;
                                }
                                reject(new Error(
                                    'Update manager was destroyed'
                                ));
                            }
                        );
                        retryTimeoutId = GLib.timeout_add_seconds(
                            GLib.PRIORITY_DEFAULT,
                            DOWNLOAD_RETRY_DELAY,
                            () => {
                                retryTimeoutId = 0;
                                resolve();
                                return GLib.SOURCE_REMOVE;
                            }
                        );
                    });
                } finally {
                    if (retryTimeoutId)
                        GLib.source_remove(retryTimeoutId);
                    if (retryCancellationId) {
                        managerCancellable.disconnect(
                            retryCancellationId
                        );
                    }
                }
            }

            await output.close_async(
                GLib.PRIORITY_DEFAULT,
                managerCancellable
            );
            output = null;
            if (this._destroyed)
                throw new Error('Update manager was destroyed');

            if (size > 0 && written !== size) {
                throw new Error(
                    `Downloaded ${written} bytes, expected ${size}`
                );
            }
            onProgress(size > 0 ? 1 : -1);
            return written;
        } finally {
            if (output) {
                try {
                    await output.close_async(
                        GLib.PRIORITY_DEFAULT,
                        managerCancellable
                    );
                } catch (_error) {
                    // Preserve the original download error.
                }
            }
        }
    }

    destroy() {
        if (this._destroyed)
            return;

        this._destroyed = true;
        for (const process of this._processes)
            process.send_signal(2);
        this._cancellable.cancel();
        if (this._firstCheckId) {
            GLib.source_remove(this._firstCheckId);
            this._firstCheckId = 0;
        }
        if (this._checkIntervalId) {
            GLib.source_remove(this._checkIntervalId);
            this._checkIntervalId = 0;
        }

        this.snapTask = null;
        this.extensionTask = null;
        this._checkPromise = null;
        this._session = null;
        this._cancellable = null;
        this._extensionPath = null;
        this._metadata = null;
        this._notify = null;
        this._cacheDir = null;
        this._snapMountRoots = null;
        this._userExtensionsDir = null;
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
            const assets = Array.isArray(release.assets) ? release.assets : [];
            const commit = typeof release.target_commitish === 'string' &&
                /^[0-9a-f]{7,40}$/i.test(release.target_commitish)
                ? release.target_commitish
                : null;
            this.remote = {
                commit,
                snap: this._releaseAsset(assets, SNAP_ASSET),
                extension: this._releaseAsset(assets, EXTENSION_ASSET),
            };
            this.lastError = null;

            if (manual) {
                const parts = [];
                if (this.snapInstalled && this.installed.snap === null) {
                    parts.push(
                        'Singstone version unknown; use Reinstall Singstone'
                    );
                } else if (this.snapUpdateAvailable) {
                    parts.push(`Singstone ${shortCommit(commit)} available`);
                }
                if (this.installed.extension === null) {
                    parts.push(
                        'Extension version unknown; use Reinstall extension'
                    );
                } else if (this.extensionUpdateAvailable) {
                    parts.push(`Extension ${shortCommit(commit)} available`);
                }
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

    async _runInstaller(argv) {
        const process = Gio.Subprocess.new(
            argv,
            Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_PIPE
        );
        this._processes.add(process);
        let watchdogId = 0;
        const communication = process.communicate_utf8_async(
            null, this._cancellable
        ).finally(() => {
            this._processes.delete(process);
            if (watchdogId) {
                GLib.Source.remove(watchdogId);
                watchdogId = 0;
            }
        });
        const timeout = new Promise((_resolve, reject) => {
            watchdogId = GLib.timeout_add_seconds(
                GLib.PRIORITY_DEFAULT,
                600,
                () => {
                    watchdogId = 0;
                    process.send_signal(2);
                    reject(new Error(
                        `${argv[0]} timed out after 10 minutes`
                    ));
                    return GLib.SOURCE_REMOVE;
                }
            );
        });
        // After a watchdog timeout the interrupted process still settles
        // `communication`; swallow that late rejection.
        communication.catch(() => {});
        const [, stderr] = await Promise.race([communication, timeout]);
        if (this._destroyed)
            throw new Error('Update manager was destroyed');
        if (process.get_successful())
            return;

        const lastLine = stderr
            ?.split('\n')
            .map(line => line.trim())
            .filter(Boolean)
            .at(-1);
        throw new Error(lastLine || `exit status ${process.get_exit_status()}`);
    }

    _releaseAsset(assets, name) {
        const asset = assets.find(candidate => candidate?.name === name);
        if (!asset || typeof asset.browser_download_url !== 'string')
            return null;

        return {
            url: asset.browser_download_url,
            apiUrl: typeof asset.url === 'string' ? asset.url : null,
            size: Number.isFinite(Number(asset.size)) ? Number(asset.size) : 0,
        };
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

    _ensureCacheDir() {
        if (GLib.mkdir_with_parents(this._cacheDir, 0o700) !== 0)
            throw new Error(`Cannot create cache directory ${this._cacheDir}`);
    }

    _unlink(path) {
        try {
            GLib.unlink(path);
        } catch (_error) {
            // The download may have failed before creating the file.
        }
    }

    _errorMessage(error) {
        return error?.message ?? `${error}`;
    }

    _emitChanged() {
        if (!this._destroyed)
            this.emit('changed');
    }
});
