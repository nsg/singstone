import Clutter from 'gi://Clutter';
import GObject from 'gi://GObject';
import Shell from 'gi://Shell';
import St from 'gi://St';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

import {RecorderClient} from './recorder.js';
import {UpdateManager} from './updater.js';

const METER_WIDTH = 40;
const METER_HEIGHT = 5;

class LevelMeter {
    constructor(iconName) {
        this.actor = new St.BoxLayout({
            style_class: 'singstone-meter-row',
            x_align: Clutter.ActorAlign.START,
        });
        this.actor.add_child(new St.Icon({
            icon_name: iconName,
            icon_size: 12,
        }));

        this._track = new St.Widget({
            style_class: 'singstone-meter-track',
            layout_manager: new Clutter.FixedLayout(),
            clip_to_allocation: true,
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._track.set_size(METER_WIDTH, METER_HEIGHT);
        this._fill = new St.Widget({
            style_class: 'singstone-meter-fill',
            x_align: Clutter.ActorAlign.START,
            y_align: Clutter.ActorAlign.START,
        });
        this._fill.set_size(0, METER_HEIGHT);
        this._track.add_child(this._fill);
        this.actor.add_child(this._track);
        this._displayedLevel = 0;
    }

    setLevel(level) {
        const normalized = Math.max(0, Math.min(1, Number(level) || 0));
        this._displayedLevel = Math.max(
            normalized,
            this._displayedLevel * 0.7
        );
        this._fill.set_width(Math.round(this._displayedLevel * METER_WIDTH));
    }

    reset() {
        this._displayedLevel = 0;
        this._fill.set_width(0);
    }
}

const SingstoneButton = GObject.registerClass(
class SingstoneButton extends PanelMenu.Button {
    _init(client, updater) {
        super._init(0.0, 'Singstone');
        this._client = client;
        this._updater = updater;

        this._box = new St.BoxLayout({
            style_class: 'panel-status-menu-box singstone-indicator',
        });
        this.add_child(this._box);

        this._recordIcon = new St.Icon({
            icon_name: 'media-record-symbolic',
            style_class: 'system-status-icon singstone-record-icon',
        });
        this._box.add_child(this._recordIcon);

        this._elapsedLabel = new St.Label({
            style_class: 'singstone-elapsed',
            y_align: Clutter.ActorAlign.CENTER,
            text: 'Record',
        });
        this._box.add_child(this._elapsedLabel);

        this._meters = new St.BoxLayout({style_class: 'singstone-meters'});
        if (this._meters.orientation !== undefined)
            this._meters.orientation = Clutter.Orientation.VERTICAL;
        else
            this._meters.vertical = true;
        this._micMeter = new LevelMeter('audio-input-microphone-symbolic');
        this._systemMeter = new LevelMeter('audio-speakers-symbolic');
        this._meters.add_child(this._micMeter.actor);
        this._meters.add_child(this._systemMeter.actor);
        this._box.add_child(this._meters);

        this._stopIcon = new St.Icon({
            icon_name: 'media-playback-stop-symbolic',
            style_class: 'system-status-icon',
        });
        this._box.add_child(this._stopIcon);

        this._updateIcon = new St.Icon({
            icon_name: 'software-update-available-symbolic',
            style_class: 'system-status-icon singstone-update-icon',
            reactive: true,
            track_hover: true,
        });
        this._updateIcon.connect('button-press-event', () => {
            this.menu.toggle();
            return Clutter.EVENT_STOP;
        });
        this._updateIcon.connect('touch-event', (_actor, event) => {
            if (event.type() === Clutter.EventType.TOUCH_BEGIN)
                this.menu.toggle();
            return Clutter.EVENT_STOP;
        });
        this._box.add_child(this._updateIcon);

        this._recordingItem = new PopupMenu.PopupMenuItem('Start recording');
        this._recordingItem.connect('activate', () => this._toggleRecording());
        this.menu.addMenuItem(this._recordingItem);
        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        const openItem = new PopupMenu.PopupMenuItem('Open Singstone');
        openItem.connect('activate', () => this._client.openApp());
        this.menu.addMenuItem(openItem);
        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._snapUpdateItem = new PopupMenu.PopupMenuItem('Reinstall Singstone');
        this._snapUpdateItem.connect('activate', () => {
            if (this._client.status.recording) {
                Main.notify(
                    'Singstone',
                    'Stop the recording before updating Singstone'
                );
                return;
            }
            this._updater.updateSnap({
                beforeInstall: async () => {
                    if (!this._client.available)
                        return false;
                    this._client.quit();
                    await this._client.waitUntilUnavailable(15_000);
                    return true;
                },
                afterInstall: wasRunning => {
                    if (wasRunning)
                        this._client.openApp();
                },
            });
        });
        this.menu.addMenuItem(this._snapUpdateItem);

        this._extensionUpdateItem = new PopupMenu.PopupMenuItem(
            'Reinstall extension'
        );
        this._extensionUpdateItem.connect(
            'activate', () => this._updater.updateExtension()
        );
        this.menu.addMenuItem(this._extensionUpdateItem);

        this._checkUpdateItem = new PopupMenu.PopupMenuItem(
            'Check for updates'
        );
        this._checkUpdateItem.connect(
            'activate', () => this._updater.check({manual: true})
        );
        this.menu.addMenuItem(this._checkUpdateItem);

        this._statusChangedId = this._client.connect(
            'status-changed', () => this._sync()
        );
        this._updatesChangedId = this._updater.connect(
            'changed', () => this._syncUpdates()
        );
        this._menuOpenId = this.menu.connect(
            'open-state-changed', (_menu, open) => {
                if (open)
                    this._updater.checkIfStale();
            }
        );
        this.connect('destroy', () => {
            if (this._statusChangedId) {
                this._client.disconnect(this._statusChangedId);
                this._statusChangedId = 0;
            }
            if (this._updatesChangedId) {
                this._updater.disconnect(this._updatesChangedId);
                this._updatesChangedId = 0;
            }
            if (this._menuOpenId) {
                this.menu.disconnect(this._menuOpenId);
                this._menuOpenId = 0;
            }
        });
        this._sync();
        this._syncUpdates();
    }

    vfunc_event(event) {
        const type = event.type();
        const leftClick = type === Clutter.EventType.BUTTON_PRESS &&
            event.get_button() === 1;
        if (leftClick || type === Clutter.EventType.TOUCH_BEGIN) {
            this._toggleRecording();
            return Clutter.EVENT_STOP;
        }

        return super.vfunc_event(event);
    }

    _toggleRecording() {
        const {recording, stopping} = this._client.status;
        if (stopping)
            return;
        if (recording)
            this._client.stop();
        else
            this._client.start();
    }

    _sync() {
        const status = this._client.status;
        if (!status.recording) {
            this._recordIcon.remove_style_class_name(
                'singstone-record-icon-live'
            );
            this._elapsedLabel.text = 'Record';
            this._meters.hide();
            this._stopIcon.hide();
            this._micMeter.reset();
            this._systemMeter.reset();
            this._recordingItem.label.text = 'Start recording';
            this._recordingItem.setSensitive(true);
            return;
        }

        this._recordIcon.add_style_class_name('singstone-record-icon-live');
        this._elapsedLabel.text = status.stopping
            ? 'Stopping…'
            : formatElapsed(status.elapsed);
        this._meters.show();
        this._stopIcon.visible = !status.stopping;

        this._micMeter.actor.visible = status.mic;
        this._systemMeter.actor.visible = status.system;
        if (status.mic)
            this._micMeter.setLevel(status.mic_level);
        else
            this._micMeter.reset();
        if (status.system)
            this._systemMeter.setLevel(status.system_level);
        else
            this._systemMeter.reset();

        this._recordingItem.label.text = 'Stop recording';
        this._recordingItem.setSensitive(!status.stopping);
    }

    _syncUpdates() {
        const snapTask = this._updater.snapTask;
        this._snapUpdateItem.visible = this._updater.snapInstalled;
        if (snapTask?.phase === 'checking') {
            this._snapUpdateItem.label.text = 'Checking for updates…';
            this._snapUpdateItem.setSensitive(false);
        } else if (snapTask?.phase === 'downloading') {
            this._snapUpdateItem.label.text = downloadLabel(
                'Downloading Singstone…', snapTask.progress
            );
            this._snapUpdateItem.setSensitive(false);
        } else if (snapTask?.phase === 'installing') {
            this._snapUpdateItem.label.text = 'Installing Singstone…';
            this._snapUpdateItem.setSensitive(false);
        } else if (snapTask?.phase === 'stopping') {
            this._snapUpdateItem.label.text = 'Stopping Singstone…';
            this._snapUpdateItem.setSensitive(false);
        } else {
            this._snapUpdateItem.label.text = this._updater.snapUpdateAvailable
                ? `Update Singstone to ${shortCommit(this._updater.remote.commit)}`
                : 'Reinstall Singstone';
            this._snapUpdateItem.setSensitive(!this._updater.checking);
        }

        const extensionTask = this._updater.extensionTask;
        if (extensionTask?.phase === 'checking') {
            this._extensionUpdateItem.label.text = 'Checking for updates…';
            this._extensionUpdateItem.setSensitive(false);
        } else if (extensionTask?.phase === 'downloading') {
            this._extensionUpdateItem.label.text = downloadLabel(
                'Downloading extension…', extensionTask.progress
            );
            this._extensionUpdateItem.setSensitive(false);
        } else if (extensionTask?.phase === 'installing') {
            this._extensionUpdateItem.label.text = 'Installing extension…';
            this._extensionUpdateItem.setSensitive(false);
        } else if (this._updater.extensionRestartPending &&
            !this._updater.extensionUpdateAvailable) {
            this._extensionUpdateItem.label.text =
                'Extension updated; log out to load it';
            this._extensionUpdateItem.setSensitive(false);
        } else {
            this._extensionUpdateItem.label.text =
                this._updater.extensionUpdateAvailable
                    ? `Update extension to ${shortCommit(this._updater.remote.commit)}`
                    : 'Reinstall extension';
            this._extensionUpdateItem.setSensitive(!this._updater.checking);
        }

        this._checkUpdateItem.label.text = this._updater.checking
            ? 'Checking for updates…'
            : 'Check for updates';
        this._checkUpdateItem.setSensitive(!this._updater.checking);
        this._updateIcon.visible = this._updater.snapUpdateAvailable ||
            this._updater.extensionUpdateAvailable ||
            this._updater.extensionRestartPending;
    }
});

function shortCommit(commit) {
    return commit?.slice(0, 7) ?? '';
}

function downloadLabel(label, progress) {
    if (progress < 0)
        return label;
    return `${label} ${Math.round(progress * 100)}%`;
}

function formatElapsed(elapsed) {
    const seconds = Math.max(0, Math.floor(Number(elapsed) || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor(seconds % 3600 / 60);
    const remainder = seconds % 60;
    const pair = value => `${value}`.padStart(2, '0');

    return hours > 0
        ? `${pair(hours)}:${pair(minutes)}:${pair(remainder)}`
        : `${pair(minutes)}:${pair(remainder)}`;
}

export default class SingstoneExtension extends Extension {
    enable() {
        this._updater = new UpdateManager({
            extensionPath: this.path,
            metadata: this.metadata,
            notify: message => Main.notify('Singstone', message),
        });
        this._client = new RecorderClient({
            onError: message => Main.notify('Singstone', message),
            launch: desktopId => {
                const app = Shell.AppSystem.get_default().lookup_app(desktopId);
                if (!app)
                    return false;
                app.activate();
                return true;
            },
        });
        this._button = new SingstoneButton(this._client, this._updater);
        Main.panel.addToStatusArea('singstone', this._button, 0, 'right');
    }

    disable() {
        this._button?.destroy();
        this._updater?.destroy();
        this._client?.destroy();
        this._button = null;
        this._updater = null;
        this._client = null;
    }
}
