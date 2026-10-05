#!/usr/bin/gjs -m

import GLib from 'gi://GLib';
import System from 'system';

import {UpdateManager} from '../gnome-shell-extension/singstone@nsg.github.io/updater.js';

const loop = new GLib.MainLoop(null, false);
const tmp = GLib.dir_make_tmp('singstone-update-smoke-XXXXXX');
const metadataPath = GLib.build_filenamev([tmp, 'metadata.json']);
const metadata = ARGV[0] ? {commit: ARGV[0]} : {};
GLib.file_set_contents(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

const snapMountRoot = GLib.getenv('SINGSTONE_SNAP_MOUNT_ROOT');
let manager = null;
let exitStatus = 1;

async function run() {
    manager = new UpdateManager({
        extensionPath: tmp,
        autoCheck: false,
        userExtensionsDir: tmp,
        notify: message => console.log(`notify: ${message}`),
        snapMountRoots: snapMountRoot ? [snapMountRoot] : undefined,
    });
    manager.connect('changed', source => {
        console.log(
            `changed: checking=${source.checking}`
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

    exitStatus = manager.lastError ? 1 : 0;
}

GLib.idle_add(GLib.PRIORITY_DEFAULT_IDLE, () => {
    run().catch(error => {
        console.error(`error: ${error.message ?? error}`);
        exitStatus = 1;
    }).finally(() => {
        manager?.destroy();
        GLib.unlink(metadataPath);
        GLib.rmdir(tmp);
        loop.quit();
    });
    return GLib.SOURCE_REMOVE;
});

loop.run();
System.exit(exitStatus);
