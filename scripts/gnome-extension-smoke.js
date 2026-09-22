#!/usr/bin/gjs -m

import GLib from 'gi://GLib';

import {RecorderClient} from '../gnome-shell-extension/singstone@nsg.github.io/recorder.js';

const loop = new GLib.MainLoop(null, false);
const client = new RecorderClient({
    onError: message => console.error(`error: ${message}`),
});

client.connect('available-changed', (_client, available) => {
    console.log(`available=${available}`);
});
client.connect('status-changed', source => {
    console.log(`status=${JSON.stringify(source.status)}`);
});
console.log(`available=${client.available}`);

const command = ARGV[0];
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
