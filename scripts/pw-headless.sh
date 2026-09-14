#!/usr/bin/env bash
# Start a headless PipeWire daemon (no hardware) with test nodes, for testing
# capture in containers and CI.
#
#   source scripts/pw-headless.sh   # exports XDG_RUNTIME_DIR for this shell
#
# Nodes:
#   test-sink     Audio/Sink    play into it; the recorder captures its monitor (--system test-sink)
#   test-mic-in   Audio/Sink    play into it to feed the virtual microphone
#   test-mic      Audio/Source  the virtual microphone (--mic test-mic), fed by a pw-loopback
set -u
: "${XDG_RUNTIME_DIR:=/tmp/singstone-pw-$(id -u)}"
export XDG_RUNTIME_DIR PIPEWIRE_RUNTIME_DIR="$XDG_RUNTIME_DIR"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

if ! pgrep -x pipewire >/dev/null; then
  (setsid pipewire >"$XDG_RUNTIME_DIR/pipewire.log" 2>&1 &)
  sleep 1
fi
if ! pgrep -x wireplumber >/dev/null; then
  (setsid wireplumber >"$XDG_RUNTIME_DIR/wireplumber.log" 2>&1 &)
  sleep 2
fi

if ! pw-cli ls Node 2>/dev/null | grep -q 'node.name = "test-sink"'; then
  pw-cli create-node adapter '{ factory.name=support.null-audio-sink node.name=test-sink node.description="Test Sink" media.class=Audio/Sink object.linger=true audio.position=[FL,FR] }' >/dev/null
fi
if ! pgrep -x pw-loopback >/dev/null; then
  (setsid pw-loopback -m '[ MONO ]' \
    --capture-props='media.class=Audio/Sink node.name=test-mic-in node.description="Test Mic In"' \
    --playback-props='media.class=Audio/Source node.name=test-mic node.description="Test Mic"' \
    >"$XDG_RUNTIME_DIR/loopback.log" 2>&1 &)
  sleep 1
fi
echo "PipeWire ready in $XDG_RUNTIME_DIR (nodes: test-sink, test-mic-in -> test-mic)"
