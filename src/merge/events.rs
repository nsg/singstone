use crate::merge::process::ProcessingProgress;
use crate::transcription::TranscriptionProgress;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProcessingEvent {
    Progress { progress: ProcessingProgress },
    Transcription { progress: TranscriptionProgress },
    Failure { error: String },
}

pub fn encode(event: &ProcessingEvent) -> serde_json::Result<String> {
    serde_json::to_string(event)
}

pub fn parse(line: &str) -> serde_json::Result<ProcessingEvent> {
    serde_json::from_str(line)
}

pub fn write_line(writer: &mut impl Write, event: &ProcessingEvent) -> io::Result<()> {
    writer.write_all(encode(event)?.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge::process::ProcessingStage;

    #[test]
    fn event_line_round_trips() {
        let event = ProcessingEvent::Progress {
            progress: ProcessingProgress {
                stage: ProcessingStage::TranscribingSystem,
                fraction: Some(0.42),
            },
        };

        assert_eq!(
            parse(&encode(&event).expect("encode event")).unwrap(),
            event
        );
    }

    #[test]
    fn malformed_event_line_is_rejected() {
        assert!(parse("not json").is_err());
    }
}
