use crate::cli::{SpeakersArgs, SpeakersCommand, TidyArgs};
use crate::speaker::database::{self, PruneReason, SpeakerDatabase, unix_now};
use crate::speaker::tidy::{self, MIN_TIDIED_SAMPLES, Plan, Summary, Verdict};
use std::io;
use std::path::Path;

pub fn run(args: SpeakersArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = args.speakers_db.unwrap_or_else(database::default_path);
    let database = match SpeakerDatabase::load(&path) {
        Ok(database) => database,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("No learned speakers ({})", path.display());
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    match args.command {
        None => list(&database),
        Some(SpeakersCommand::Tidy(tidy)) => run_tidy(&path, &database, &tidy)?,
    }
    Ok(())
}

fn list(database: &SpeakerDatabase) {
    println!(
        "Model: {} (sha256 {}, dimension {})",
        database.embedding_model.name,
        database.embedding_model.sha256,
        database.embedding_model.dimension
    );
    for (name, speaker) in &database.speakers {
        println!("{name}\t{} embedding(s)", speaker.samples.len());
    }
}

fn run_tidy(path: &Path, database: &SpeakerDatabase, args: &TidyArgs) -> io::Result<()> {
    if args.dry_run {
        let plan = tidy::plan(database, args.speaker_threshold, unix_now());
        print_plan(&plan);
        println!("{}", outcome(&plan.summary(), true));
    } else {
        let (plan, summary) = tidy::run(path, args.speaker_threshold)?;
        print_plan(&plan);
        println!("{}", outcome(&summary, false));
    }
    Ok(())
}

fn print_plan(plan: &Plan) {
    for person in &plan.people {
        let samples = format!("{} sample(s)", person.samples);
        if !person.tidied() {
            println!(
                "{}\t{samples}, not tidied below {MIN_TIDIED_SAMPLES}",
                person.name
            );
        } else if person.findings.is_empty() {
            println!("{}\t{samples}, nothing to tidy", person.name);
        } else {
            println!("{}\t{samples}", person.name);
        }
        for finding in &person.findings {
            let own = finding
                .own_score
                .map_or_else(|| "-".to_owned(), |score| format!("{score:.3}"));
            let reason = match &finding.reason {
                PruneReason::SoundsLike { rival } => {
                    format!("sounds like {rival} ({:.3} vs own {own})", finding.limit)
                }
                PruneReason::Isolated => {
                    format!("isolated (own {own} below {:.3})", finding.limit)
                }
                PruneReason::User => "removed by the user".to_owned(),
            };
            let verdict = match finding.verdict {
                Verdict::Remove => "remove".to_owned(),
                Verdict::Deferred => "deferred: over this pass's limit".to_owned(),
                Verdict::Settling(sessions) => {
                    format!("settling: {sessions} more session(s) to go")
                }
                Verdict::Recurring => "protected: recurs across sessions".to_owned(),
                Verdict::Kept => "kept: restored by the user".to_owned(),
            };
            println!("  sample {}\t{reason}\t{verdict}", finding.sample + 1);
        }
        if person.expired > 0 {
            println!(
                "  {} removed sample(s) past the restore period",
                person.expired
            );
        }
    }
}

fn outcome(summary: &Summary, dry_run: bool) -> String {
    let mut lines = vec![match (summary.describe(), dry_run) {
        (None, _) => "Nothing to tidy.".to_owned(),
        (Some(removed), true) => format!("Would tidy {removed}."),
        (Some(removed), false) => format!("Tidied {removed}."),
    }];
    if summary.expired > 0 {
        lines.push(format!(
            "{} {} removed sample(s) past the restore period.",
            if dry_run { "Would delete" } else { "Deleted" },
            summary.expired
        ));
    }
    lines.join("\n")
}
