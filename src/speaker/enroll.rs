use crate::cli::{EnrollArgs, SpeakersArgs};
use crate::models;
use crate::speaker::database::{self, SpeakerDatabase, SpeakerRecord};
use crate::speaker::embedding::EmbeddingExtractor;
use crate::speaker::wav;
use crate::types::SAMPLE_RATE;
use std::io;

pub fn run(args: EnrollArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.name.trim().is_empty() {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "speaker name cannot be empty").into(),
        );
    }
    models::verify_model(
        &args.embedding_model,
        "speaker-embedding",
        args.models_lock.as_deref(),
        args.allow_unverified_models,
    )?;
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let extractor = EmbeddingExtractor::new(&args.embedding_model, threads)?;
    let identity = database::identity(&args.embedding_model, extractor.dimension())?;
    let path = args.speakers_db.unwrap_or_else(database::default_path);
    let mut database = match SpeakerDatabase::load_checked(&path, &identity) {
        Ok(database) => database,
        Err(error) if error.kind() == io::ErrorKind::NotFound => SpeakerDatabase::empty(identity),
        Err(error) => return Err(error.into()),
    };
    let mut embeddings = Vec::new();
    for path in &args.samples {
        let samples = wav::read_audio(path)?;
        if samples.len() < 2 * SAMPLE_RATE as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} contains less than 2 seconds of audio", path.display()),
            )
            .into());
        }
        let embedding = extractor.embed(&samples).ok_or_else(|| {
            io::Error::other(format!(
                "could not compute an embedding for {}",
                path.display()
            ))
        })?;
        embeddings.push(embedding);
    }
    let record = database
        .speakers
        .entry(args.name.clone())
        .or_insert_with(SpeakerRecord::default);
    if args.replace {
        record.embeddings.clear();
    }
    record.embeddings.extend(embeddings);
    let count = record.embeddings.len();
    database.save(&path)?;
    eprintln!(
        "enrolled {}: {} embedding(s) in {}",
        args.name,
        count,
        path.display()
    );
    Ok(())
}

pub fn list(args: SpeakersArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = args.speakers_db.unwrap_or_else(database::default_path);
    let database = match SpeakerDatabase::load(&path) {
        Ok(database) => database,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("No enrolled speakers ({})", path.display());
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    println!(
        "Model: {} (sha256 {}, dimension {})",
        database.embedding_model.name,
        database.embedding_model.sha256,
        database.embedding_model.dimension
    );
    for (name, speaker) in database.speakers {
        println!("{name}\t{} embedding(s)", speaker.embeddings.len());
    }
    Ok(())
}
