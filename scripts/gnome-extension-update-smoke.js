#!/usr/bin/gjs -m

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import System from 'system';

import {UpdateManager} from '../gnome-shell-extension/singstone@nsg.github.io/updater.js';

const loop = new GLib.MainLoop(null, false);
const tmp = GLib.dir_make_tmp('singstone-update-smoke-XXXXXX');
const metadataPath = GLib.build_filenamev([tmp, 'metadata.json']);
const downloadPath = GLib.build_filenamev([
    tmp, 'singstone-gnome-shell-extension.zip',
]);
const metadata = ARGV[0] ? {commit: ARGV[0]} : {};
GLib.file_set_contents(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

const snapMountRoot = GLib.getenv('SINGSTONE_SNAP_MOUNT_ROOT');
let manager = null;
let exitStatus = 1;

async function run() {
    manager = new UpdateManager({
        extensionPath: tmp,
        autoCheck: false,
        cacheDir: tmp,
        userExtensionsDir: tmp,
        notify: message => console.log(`notify: ${message}`),
        snapMountRoots: snapMountRoot ? [snapMountRoot] : undefined,
    });
    manager.connect('changed', source => {
        console.log(
            `changed: checking=${source.checking} ` +
            `snapTask=${source.snapTask?.phase ?? 'idle'} ` +
            `extensionTask=${source.extensionTask?.phase ?? 'idle'}`
        );
    });

    await manager.check({manual: true});
    console.log(JSON.stringify({
        remote: manager.remote,
        installed: manager.installed,
        snapInstalled: manager.snapInstalled,
        snapUpdateAvailable: manager.snapUpdateAvailable,
        extensionUpdateAvailable: manager.extensionUpdateAvailable,
        lastError: manager.lastError,
    }));

    if (!manager.lastError && ARGV[1] === 'download-extension') {
        const asset = manager.remote?.extension;
        if (!asset)
            throw new Error('The release has no extension asset');
        const written = await manager._download(
            asset,
            downloadPath,
            progress => console.log(`download-progress=${progress}`)
        );
        const [contents] = await Gio.File.new_for_path(downloadPath)
            .load_contents_async(null);
        if (contents.length !== written)
            throw new Error(`Read ${contents.length} bytes after writing ${written}`);
        console.log(`downloaded-extension-bytes=${written}`);
    }

    exitStatus = manager.lastError ? 1 : 0;
}

GLib.idle_add(GLib.PRIORITY_DEFAULT_IDLE, () => {
    run().catch(error => {
        console.error(`error: ${error.message ?? error}`);
        exitStatus = 1;
    }).finally(() => {
        manager?.destroy();
        for (const path of [downloadPath, metadataPath]) {
            try {
                GLib.unlink(path);
            } catch (_error) {
                // The optional download may not have been created.
            }
        }
        GLib.rmdir(tmp);
        loop.quit();
    });
    return GLib.SOURCE_REMOVE;
});

loop.run();
System.exit(exitStatus);
