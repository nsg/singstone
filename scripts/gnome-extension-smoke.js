#!/usr/bin/gjs -m

// Usage: gjs -m scripts/gnome-extension-smoke.js [start|stop] [--no-signals]

import GLib from 'gi://GLib';

import {RecorderClient} from '../gnome-shell-extension/singstone@nsg.github.io/recorder.js';

const subscribeSignals = !ARGV.includes('--no-signals');
const command = ARGV.find(argument => argument !== '--no-signals');
const loop = new GLib.MainLoop(null, false);
const client = new RecorderClient({
    onError: message => console.error(`error: ${message}`),
    subscribeSignals,
});

client.connect('available-changed', (_client, available) => {
    console.log(`available=${available}`);
});
client.connect('status-changed', source => {
    console.log(
        `status=${JSON.stringify(source.status)} source=${source.statusSource}`
    );
});
console.log(`available=${client.available}`);

if (command === 'start' || command === 'stop') {
    GLib.timeout_add(GLib.PRIORITY_DEFAULT, 500, () => {
        client[command]();
        return GLib.SOURCE_REMOVE;
    });
}

GLib.timeout_add(GLib.PRIORITY_DEFAULT, 6000, () => {
    client.destroy();
    loop.quit();
    return GLib.SOURCE_REMOVE;
});

loop.run();
