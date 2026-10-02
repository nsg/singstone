use crate::cli::SpeakersArgs;
use crate::speaker::database::{self, SpeakerDatabase};
use std::io;

pub fn list(args: SpeakersArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = args.speakers_db.unwrap_or_else(database::default_path);
    let database = match SpeakerDatabase::load(&path) {
        Ok(database) => database,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("No learned speakers ({})", path.display());
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
