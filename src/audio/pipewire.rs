use pipewire as pw;
use pw::properties::properties;
use std::io;

pub fn new_capture_stream(
    core: &pw::core::CoreRc,
    name: &str,
    target: &str,
    capture_sink: bool,
) -> Result<pw::stream::StreamRc, pw::Error> {
    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Communication",
    };
    if target != "default" {
        props.insert(*pw::keys::TARGET_OBJECT, target);
        props.insert(*pw::keys::NODE_DONT_RECONNECT, "true");
    }
    if capture_sink {
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }
    pw::stream::StreamRc::new(core.clone(), name, props)
}

pub fn connect(stream: &pw::stream::Stream) -> Result<(), Box<dyn std::error::Error>> {
    let mut audio_info = pw::spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(pw::spa::param::audio::AudioFormat::F32LE);
    audio_info.set_rate(crate::types::SAMPLE_RATE);
    audio_info.set_channels(1);
    let mut position = [0u32; pw::spa::sys::SPA_AUDIO_MAX_CHANNELS as usize];
    position[0] = pw::spa::sys::SPA_AUDIO_CHANNEL_MONO;
    audio_info.set_position(position);
    let object = pw::spa::pod::Object {
        type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: pw::spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values = pw::spa::pod::serialize::PodSerializer::serialize(
        io::Cursor::new(Vec::new()),
        &pw::spa::pod::Value::Object(object),
    )?
    .0
    .into_inner();
    let pod = pw::spa::pod::Pod::from_bytes(&values)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid format pod"))?;
    let mut params = [pod];
    stream.connect(
        pw::spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;
    Ok(())
}

pub fn monotonic_ns() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    (now.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(now.tv_nsec as u64)
}
