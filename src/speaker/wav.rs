use crate::types::SAMPLE_RATE;
use std::fs;
use std::io;
use std::path::Path;

pub fn read_audio(path: &Path) -> io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.starts_with(b"RIFF") {
        read_wav_bytes(&bytes)
    } else {
        if bytes.len() % 4 != 0 {
            return Err(invalid("raw f32le file length is not a multiple of 4"));
        }
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

fn read_wav_bytes(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if bytes.len() < 12 || &bytes[8..12] != b"WAVE" {
        return Err(invalid("invalid RIFF/WAVE header"));
    }
    let mut format = None;
    let mut data = None;
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| invalid("invalid WAV chunk"))?,
        ) as usize;
        let start = offset + 8;
        let end = start
            .checked_add(size)
            .ok_or_else(|| invalid("WAV chunk size overflow"))?;
        if end > bytes.len() {
            return Err(invalid("truncated WAV chunk"));
        }
        if id == b"fmt " {
            format = Some(parse_format(&bytes[start..end])?);
        } else if id == b"data" {
            data = Some(&bytes[start..end]);
        }
        offset = end + (size & 1);
    }
    let format = format.ok_or_else(|| invalid("WAV has no fmt chunk"))?;
    let data = data.ok_or_else(|| invalid("WAV has no data chunk"))?;
    if format.channels != 1 {
        return Err(invalid(format!(
            "WAV must be mono, found {} channels",
            format.channels
        )));
    }
    if format.sample_rate != SAMPLE_RATE {
        return Err(invalid(format!(
            "WAV must be 16000 Hz, found {} Hz",
            format.sample_rate
        )));
    }
    match (format.encoding, format.bits) {
        (1, 16) => exact_chunks(data, 2)?
            .map(|b| Ok(i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0))
            .collect(),
        (1, 32) => exact_chunks(data, 4)?
            .map(|b| Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2147483648.0))
            .collect(),
        (3, 32) => exact_chunks(data, 4)?
            .map(|b| Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        _ => Err(invalid(format!(
            "unsupported WAV format {}-bit encoding {} (expected PCM16, PCM32, or float32)",
            format.bits, format.encoding
        ))),
    }
}

struct WaveFormat {
    encoding: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
}

fn parse_format(bytes: &[u8]) -> io::Result<WaveFormat> {
    if bytes.len() < 16 {
        return Err(invalid("truncated WAV fmt chunk"));
    }
    Ok(WaveFormat {
        encoding: u16::from_le_bytes([bytes[0], bytes[1]]),
        channels: u16::from_le_bytes([bytes[2], bytes[3]]),
        sample_rate: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        bits: u16::from_le_bytes([bytes[14], bytes[15]]),
    })
}

fn exact_chunks(bytes: &[u8], size: usize) -> io::Result<std::slice::ChunksExact<'_, u8>> {
    let chunks = bytes.chunks_exact(size);
    if chunks.remainder().is_empty() {
        Ok(chunks)
    } else {
        Err(invalid("WAV data is not aligned to its sample size"))
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(encoding: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt \x10\0\0\0");
        out.extend_from_slice(&encoding.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        let width = u32::from(bits / 8) * u32::from(channels);
        out.extend_from_slice(&(rate * width).to_le_bytes());
        out.extend_from_slice(&(width as u16).to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_pcm16() {
        let bytes = wav(1, 1, 16_000, 16, &[0, 0, 0xff, 0x7f, 0, 0x80]);
        let samples = read_wav_bytes(&bytes).expect("valid WAV");
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - 0.99997).abs() < 0.0001);
        assert_eq!(samples[2], -1.0);
    }

    #[test]
    fn reads_float32() {
        let data = [0.25f32.to_le_bytes(), (-0.5f32).to_le_bytes()].concat();
        assert_eq!(
            read_wav_bytes(&wav(3, 1, 16_000, 32, &data)).expect("valid WAV"),
            vec![0.25, -0.5]
        );
    }

    #[test]
    fn rejects_stereo_and_wrong_rate() {
        assert!(
            read_wav_bytes(&wav(1, 2, 16_000, 16, &[]))
                .unwrap_err()
                .to_string()
                .contains("mono")
        );
        assert!(
            read_wav_bytes(&wav(1, 1, 44_100, 16, &[]))
                .unwrap_err()
                .to_string()
                .contains("16000")
        );
    }
}
