//! Keeps learned voices clean. Chunks that sound like someone else are not
//! learned, and a tidy pass sets aside samples that sound like someone else
//! or sit isolated from the rest of their voice.
//!
//! A sample is only judged once its voice has learned in a few later
//! sessions, so a new recording condition, such as a second microphone, can
//! build up first; one that recurs across sessions is kept however far it
//! lies from the rest. Set-aside samples can be restored for a while before
//! they are deleted.

use crate::speaker::classify::{self, Flag, Review};
use crate::speaker::database::{PruneReason, SpeakerDatabase, SpeakerRecord, unix_now};
use crate::speaker::embedding::{mean_normalized, topk_mean};
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// People with fewer samples are neither judged nor judged against.
pub const MIN_JUDGED_SAMPLES: usize = 10;
/// How much better another person must match for a sample to count as theirs.
pub const CLEARLY_BETTER: f32 = 0.05;
/// Least gap below a voice's median own score that counts as isolated.
pub const ISOLATION_MIN_GAP: f32 = 0.15;
/// Closest own samples that form a sample's neighbourhood; those at least
/// as similar as both the recognition threshold and the isolation floor
/// count as close.
pub const NEIGHBORHOOD: usize = 5;
/// Sessions a close neighbourhood must span to be a recurring condition.
pub const RECURRING_SESSIONS: usize = 2;
/// Later learning sessions a sample waits before it may be removed.
pub const SETTLING_SESSIONS: usize = 3;
/// Voices with fewer samples are not tidied.
pub const MIN_TIDIED_SAMPLES: usize = 20;
/// Share of a voice's samples one pass may remove.
pub const MAX_REMOVED_SHARE: f64 = 0.10;
/// How long removed samples can be restored, in seconds.
pub const RESTORE_PERIOD: u64 = 30 * 24 * 60 * 60;

/// Whether `vector` may be learned as `name`: not when another known person
/// matches it clearly better and well enough to be recognized.
pub fn admits(database: &SpeakerDatabase, name: &str, vector: &[f32], threshold: f32) -> bool {
    let Some(own) = database
        .speakers
        .get(name)
        .filter(|record| record.samples.len() >= MIN_JUDGED_SAMPLES)
        .and_then(|record| topk_mean(vector, &record.samples))
    else {
        return true;
    };
    !database
        .speakers
        .iter()
        .filter(|(other, record)| {
            other.as_str() != name && record.samples.len() >= MIN_JUDGED_SAMPLES
        })
        .filter_map(|(_, record)| topk_mean(vector, &record.samples))
        .any(|score| score >= threshold && score - own >= CLEARLY_BETTER)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Removed by this pass.
    Remove,
    /// Due, but over this pass's limit for the voice.
    Deferred,
    /// Still settling until the voice learns in this many more sessions.
    Settling(usize),
    /// Close to samples from other sessions: a recurring condition.
    Recurring,
    /// Restored by the user.
    Kept,
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub sample: usize,
    pub embedding: Vec<f32>,
    pub reason: PruneReason,
    pub verdict: Verdict,
    pub own_score: Option<f32>,
    /// The rival's score for a sample that sounds like someone else, the
    /// isolation floor for an isolated one.
    pub limit: f32,
}

#[derive(Debug, Clone)]
pub struct PersonPlan {
    pub name: String,
    pub samples: usize,
    pub findings: Vec<Finding>,
    /// Removed samples past their restore period.
    pub expired: usize,
}

impl PersonPlan {
    pub fn tidied(&self) -> bool {
        self.samples >= MIN_TIDIED_SAMPLES
    }
}

/// What one pass would do, decided from a single snapshot.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub people: Vec<PersonPlan>,
}

impl Plan {
    fn has_work(&self) -> bool {
        self.people.iter().any(|person| {
            person.expired > 0
                || person
                    .findings
                    .iter()
                    .any(|finding| finding.verdict == Verdict::Remove)
        })
    }

    /// The pass's removals, as `apply` would count them on the snapshot.
    pub fn summary(&self) -> Summary {
        let mut summary = Summary::default();
        for person in &self.people {
            summary.expired += person.expired;
            for finding in &person.findings {
                if finding.verdict == Verdict::Remove {
                    summary.count(&finding.reason);
                }
            }
        }
        summary
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    pub sounds_like: usize,
    pub isolated: usize,
    /// Removed samples deleted for good.
    pub expired: usize,
}

impl Summary {
    pub fn removed(&self) -> usize {
        self.sounds_like + self.isolated
    }

    fn count(&mut self, reason: &PruneReason) {
        match reason {
            PruneReason::SoundsLike { .. } => self.sounds_like += 1,
            PruneReason::Isolated => self.isolated += 1,
            PruneReason::User => {}
        }
    }

    /// "12 samples: 8 sounded like someone else, 4 were isolated"
    pub fn describe(&self) -> Option<String> {
        if self.removed() == 0 {
            return None;
        }
        let mut parts = Vec::new();
        if self.sounds_like > 0 {
            parts.push(format!("{} sounded like someone else", self.sounds_like));
        }
        if self.isolated > 0 {
            parts.push(format!(
                "{} {} isolated",
                self.isolated,
                if self.isolated == 1 { "was" } else { "were" }
            ));
        }
        Some(format!(
            "{} sample{}: {}",
            self.removed(),
            if self.removed() == 1 { "" } else { "s" },
            parts.join(", ")
        ))
    }
}

/// Plans a pass over every voice in `database`.
pub fn plan(database: &SpeakerDatabase, threshold: f32, now: u64) -> Plan {
    let (names, records) = reviewed_people(database);
    let samples = records
        .iter()
        .map(|record| record.samples.as_slice())
        .collect::<Vec<_>>();
    let centroids = samples
        .iter()
        .map(|samples| mean_normalized(samples).unwrap_or_default())
        .collect::<Vec<_>>();
    let review = classify::review(&samples, &centroids, threshold);
    plan_reviewed(database, &names, &review, threshold, now)
}

/// The people `plan` reviews, in review order: everyone with a sample.
pub fn reviewed_people(database: &SpeakerDatabase) -> (Vec<String>, Vec<&SpeakerRecord>) {
    database
        .speakers
        .iter()
        .filter(|(_, record)| !record.samples.is_empty())
        .map(|(name, record)| (name.clone(), record))
        .unzip()
}

/// Plans a pass from a review of `reviewed_people(database)`.
pub fn plan_reviewed(
    database: &SpeakerDatabase,
    names: &[String],
    review: &Review,
    threshold: f32,
    now: u64,
) -> Plan {
    let mut people = names
        .iter()
        .zip(&review.people)
        .map(|(name, person)| {
            let record = &database.speakers[name];
            let mut findings = Vec::new();
            if record.samples.len() >= MIN_TIDIED_SAMPLES {
                for (index, judged) in person.samples.iter().enumerate() {
                    if let Some(finding) =
                        judge(record, names, review, person, index, judged, threshold)
                    {
                        findings.push(finding);
                    }
                }
                limit_removals(&mut findings, record.samples.len());
            }
            PersonPlan {
                name: name.clone(),
                samples: record.samples.len(),
                findings,
                expired: expired(record, now),
            }
        })
        .collect::<Vec<_>>();
    for (name, record) in &database.speakers {
        if record.samples.is_empty() && !record.pruned.is_empty() {
            people.push(PersonPlan {
                name: name.clone(),
                samples: 0,
                findings: Vec::new(),
                expired: expired(record, now),
            });
        }
    }
    Plan { people }
}

fn judge(
    record: &SpeakerRecord,
    names: &[String],
    review: &Review,
    person: &classify::PersonReview,
    index: usize,
    judged: &classify::SampleReview,
    threshold: f32,
) -> Option<Finding> {
    let sample = &record.samples[index];
    let sounds_like = judged.flag == Flag::Stray
        && judged
            .own_score
            .zip(judged.rival)
            .is_some_and(|(own, (rival, score))| {
                score - own >= CLEARLY_BETTER
                    && review.people[rival].samples.len() >= MIN_JUDGED_SAMPLES
            });
    let (reason, limit) = if sounds_like {
        let (rival, score) = judged.rival?;
        (
            PruneReason::SoundsLike {
                rival: names[rival].clone(),
            },
            score,
        )
    } else if judged.isolated {
        (PruneReason::Isolated, person.isolation_floor?)
    } else {
        return None;
    };
    let close = threshold.max(limit);
    let recurring = || {
        let mut sessions = judged
            .neighbors
            .iter()
            .filter(|(_, similarity)| *similarity >= close)
            .filter_map(|(neighbor, _)| record.samples[*neighbor].session.as_deref())
            .chain(sample.session.as_deref())
            .collect::<Vec<_>>();
        sessions.sort_unstable();
        sessions.dedup();
        sessions.len() >= RECURRING_SESSIONS
    };
    let since = sample
        .session
        .as_deref()
        .and_then(|session| record.sessions_since(session))
        .unwrap_or(SETTLING_SESSIONS);
    let verdict = if sample.kept_by_user {
        Verdict::Kept
    } else if reason == PruneReason::Isolated && recurring() {
        Verdict::Recurring
    } else if since < SETTLING_SESSIONS {
        Verdict::Settling(SETTLING_SESSIONS - since)
    } else {
        Verdict::Remove
    };
    Some(Finding {
        sample: index,
        embedding: sample.embedding.clone(),
        reason,
        verdict,
        own_score: judged.own_score,
        limit,
    })
}

/// How far past its limit a finding lies; larger goes first.
fn severity(finding: &Finding) -> f32 {
    finding.limit - finding.own_score.unwrap_or(0.0)
}

fn removal_limit(samples: usize) -> usize {
    ((samples as f64 * MAX_REMOVED_SHARE) as usize).min(samples.saturating_sub(MIN_TIDIED_SAMPLES))
}

/// Keeps the most severe removals within the pass's limit and defers the rest.
fn limit_removals(findings: &mut [Finding], samples: usize) {
    findings.sort_by(|a, b| {
        severity(b)
            .total_cmp(&severity(a))
            .then(a.sample.cmp(&b.sample))
    });
    let mut allowed = removal_limit(samples);
    for finding in findings.iter_mut() {
        if finding.verdict == Verdict::Remove {
            if allowed == 0 {
                finding.verdict = Verdict::Deferred;
            } else {
                allowed -= 1;
            }
        }
    }
}

fn expired(record: &SpeakerRecord, now: u64) -> usize {
    record
        .pruned
        .iter()
        .filter(|pruned| now.saturating_sub(pruned.pruned_at) >= RESTORE_PERIOD)
        .count()
}

/// Carries out `plan` on `database`, which may have changed since the plan
/// was made: samples are found by their vectors and limits are rechecked.
pub fn apply(database: &mut SpeakerDatabase, plan: &Plan, now: u64) -> Summary {
    let mut summary = Summary::default();
    for person in &plan.people {
        let Some(record) = database.speakers.get_mut(&person.name) else {
            continue;
        };
        let mut allowed = removal_limit(record.samples.len());
        for finding in &person.findings {
            if finding.verdict != Verdict::Remove || allowed == 0 {
                continue;
            }
            if record.set_aside(&finding.embedding, finding.reason.clone(), now) {
                summary.count(&finding.reason);
                allowed -= 1;
            }
        }
    }
    for record in database.speakers.values_mut() {
        let before = record.pruned.len();
        record
            .pruned
            .retain(|pruned| now.saturating_sub(pruned.pruned_at) < RESTORE_PERIOD);
        summary.expired += before - record.pruned.len();
    }
    database.refresh_centroids();
    summary
}

/// Tidies the database at `path`: plans from a snapshot, then applies the
/// plan under the database lock so concurrent changes are kept.
pub fn run(path: &Path, threshold: f32) -> io::Result<(Plan, Summary)> {
    let snapshot = match SpeakerDatabase::load(path) {
        Ok(database) => database,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok((Plan::default(), Summary::default()));
        }
        Err(error) => return Err(error),
    };
    let now = unix_now();
    let plan = plan(&snapshot, threshold, now);
    if !plan.has_work() {
        return Ok((plan, Summary::default()));
    }
    let lock = SpeakerDatabase::lock(path)?;
    let mut database = SpeakerDatabase::load_locked(path, &lock)?;
    let summary = apply(&mut database, &plan, now);
    if summary != Summary::default() {
        database.save(path)?;
    }
    Ok((plan, summary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speaker::database::{EmbeddingModelIdentity, PrunedSample, Sample};
    use crate::types::AudioSource;

    const DIMENSION: usize = 16;
    const NOW: u64 = 100 * 24 * 60 * 60;

    fn unit(vector: Vec<f32>) -> Vec<f32> {
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        vector.into_iter().map(|v| v / norm).collect()
    }

    /// A direction mixing two axes, `weight` of the second.
    fn toward(first: usize, second: usize, weight: f32) -> Vec<f32> {
        unit(
            (0..DIMENSION)
                .map(|i| {
                    f32::from(u8::from(i == first)) + weight * f32::from(u8::from(i == second))
                })
                .collect(),
        )
    }

    /// `count` samples scattered closely around `center`.
    fn around(center: &[f32], count: usize, seed: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|n| {
                unit(
                    center
                        .iter()
                        .enumerate()
                        .map(|(i, c)| c + 0.04 * (((n + seed) * 31 + i * 17) as f32 * 0.618).sin())
                        .collect(),
                )
            })
            .collect()
    }

    fn voice(axis: usize, count: usize) -> Vec<Vec<f32>> {
        around(&toward(axis, axis, 0.0), count, axis * 100)
    }

    fn database(people: &[(&str, Vec<Vec<f32>>)]) -> SpeakerDatabase {
        let mut database = SpeakerDatabase::empty(EmbeddingModelIdentity {
            name: "model".into(),
            sha256: "hash".into(),
            dimension: DIMENSION,
        });
        for (name, vectors) in people {
            database
                .speakers
                .insert((*name).into(), SpeakerRecord::settled(vectors.clone()));
        }
        database.refresh_centroids();
        database
    }

    /// Me with 24 settled samples plus `extra`, and Bob with 12.
    fn me_and_bob(extra: Vec<Sample>) -> SpeakerDatabase {
        let mut database = database(&[("Me", voice(0, 24)), ("Bob", voice(4, 12))]);
        let me = database.speakers.get_mut("Me").unwrap();
        for sample in extra {
            if let Some(session) = &sample.session
                && !me.sessions.contains(session)
            {
                me.sessions.push(session.clone());
            }
            me.samples.push(sample);
        }
        database.refresh_centroids();
        database
    }

    fn learned(embedding: Vec<f32>, session: &str) -> Sample {
        Sample::learned(embedding, session, AudioSource::Mic, 0)
    }

    /// Me's findings in sample order.
    fn findings(database: &SpeakerDatabase) -> Vec<(usize, PruneReason, Verdict)> {
        let mut findings = plan(database, 0.72, NOW)
            .people
            .into_iter()
            .find(|person| person.name == "Me")
            .unwrap()
            .findings
            .into_iter()
            .map(|finding| (finding.sample, finding.reason, finding.verdict))
            .collect::<Vec<_>>();
        findings.sort_by_key(|finding| finding.0);
        findings
    }

    #[test]
    fn admission_skips_a_chunk_that_clearly_matches_another_known_person() {
        let chunk = voice(1, 11).pop().unwrap();
        let known = database(&[("Alice", voice(0, 10)), ("Bob", voice(1, 10))]);
        assert!(!admits(&known, "Alice", &chunk, 0.72));
        assert!(admits(&known, "Bob", &chunk, 0.72));
        let new_rival = database(&[("Alice", voice(0, 10)), ("Bob", voice(1, 9))]);
        assert!(admits(&new_rival, "Alice", &chunk, 0.72));
        let new_voice = database(&[("Alice", voice(0, 9)), ("Bob", voice(1, 10))]);
        assert!(admits(&new_voice, "Alice", &chunk, 0.72));
    }

    #[test]
    fn removes_a_sample_that_sounds_like_a_known_rival() {
        let bobs = voice(4, 13).pop().unwrap();
        let database = me_and_bob(vec![Sample::settled(bobs)]);
        assert_eq!(
            findings(&database),
            [(
                24,
                PruneReason::SoundsLike {
                    rival: "Bob".into()
                },
                Verdict::Remove
            )]
        );
        let mut new_rival = database.clone();
        new_rival
            .speakers
            .get_mut("Bob")
            .unwrap()
            .samples
            .truncate(9);
        assert!(
            !findings(&new_rival)
                .iter()
                .any(|f| matches!(f.1, PruneReason::SoundsLike { .. }))
        );
    }

    #[test]
    fn removes_an_isolated_sample() {
        let database = me_and_bob(vec![Sample::settled(toward(0, 9, 1.0))]);
        assert_eq!(
            findings(&database),
            [(24, PruneReason::Isolated, Verdict::Remove)]
        );
    }

    #[test]
    fn keeps_an_isolated_group_that_recurs_across_sessions() {
        let room = around(&toward(0, 9, 1.0), 2, 7);
        let recurring = me_and_bob(vec![
            learned(room[0].clone(), "a"),
            learned(room[1].clone(), "b"),
            learned(toward(0, 1, 0.1), "c"),
            learned(toward(0, 2, 0.1), "d"),
            learned(toward(0, 3, 0.1), "e"),
        ]);
        assert_eq!(
            findings(&recurring)
                .into_iter()
                .map(|f| f.2)
                .collect::<Vec<_>>(),
            [Verdict::Recurring, Verdict::Recurring]
        );
        let mut once = recurring.clone();
        once.speakers.get_mut("Me").unwrap().samples[25].session = Some("a".into());
        assert_eq!(
            findings(&once),
            [
                (24, PruneReason::Isolated, Verdict::Remove),
                (25, PruneReason::Isolated, Verdict::Remove)
            ]
        );
    }

    #[test]
    fn a_sample_settles_after_three_later_sessions() {
        let mut database = me_and_bob(vec![
            learned(toward(0, 9, 1.0), "first"),
            learned(toward(0, 1, 0.1), "second"),
            learned(toward(0, 2, 0.1), "third"),
        ]);
        assert_eq!(
            findings(&database),
            [(24, PruneReason::Isolated, Verdict::Settling(1))]
        );
        let me = database.speakers.get_mut("Me").unwrap();
        me.samples.truncate(25);
        assert!(me.learn(learned(toward(0, 3, 0.1), "fourth")));
        assert_eq!(
            findings(&database),
            [(24, PruneReason::Isolated, Verdict::Remove)]
        );
    }

    #[test]
    fn a_pass_removes_at_most_a_tenth_and_keeps_twenty() {
        let far = (5..10)
            .map(|axis| Sample::settled(toward(0, axis + 5, 1.0)))
            .collect();
        let mut database = me_and_bob(far);
        let verdicts = findings(&database)
            .into_iter()
            .map(|f| f.2)
            .collect::<Vec<_>>();
        assert_eq!(
            verdicts.iter().filter(|v| **v == Verdict::Remove).count(),
            2
        );
        assert_eq!(
            verdicts.iter().filter(|v| **v == Verdict::Deferred).count(),
            3
        );
        assert_eq!(removal_limit(20), 0);
        assert_eq!(removal_limit(21), 1);
        let plan = plan(&database, 0.72, NOW);
        assert_eq!(apply(&mut database, &plan, NOW).isolated, 2);
        assert_eq!(database.speakers["Me"].samples.len(), 27);
        assert_eq!(database.speakers["Me"].pruned.len(), 2);
    }

    #[test]
    fn a_restored_sample_is_left_alone() {
        let mut kept = Sample::settled(toward(0, 9, 1.0));
        kept.kept_by_user = true;
        let database = me_and_bob(vec![kept]);
        assert_eq!(
            findings(&database),
            [(24, PruneReason::Isolated, Verdict::Kept)]
        );
    }

    #[test]
    fn removed_samples_are_deleted_after_the_restore_period() {
        let mut database = me_and_bob(Vec::new());
        let pruned = |days: u64| PrunedSample {
            reason: PruneReason::User,
            pruned_at: NOW - days * 24 * 60 * 60,
            sample: Sample::settled(toward(0, 9, 1.0)),
        };
        database.speakers.get_mut("Me").unwrap().pruned = vec![pruned(30), pruned(29)];
        let plan = plan(&database, 0.72, NOW);
        assert_eq!(apply(&mut database, &plan, NOW).expired, 1);
        assert_eq!(database.speakers["Me"].pruned, [pruned(29)]);
    }
}
