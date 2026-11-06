//! Picking the recordings to merge with the session on screen.

use super::format_duration;
use super::player::QueueReason;
use crate::join::{self, Piece};
use crate::session::Session;
use crate::split;
use adw::prelude::*;
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;

struct Candidate {
    piece: Piece,
    check: gtk::CheckButton,
    row: adw::ActionRow,
    reason: gtk::Label,
    /// Why it cannot be picked whatever else is picked.
    blocked: Option<String>,
}

/// Ask which unprocessed recordings next to `current` go into a merge with
/// it; `preview` gets them all, `current` included.
pub(super) fn pick(
    window: &adw::ApplicationWindow,
    current: &Path,
    queue_reason: QueueReason,
    preview: impl Fn(Vec<PathBuf>) + 'static,
) {
    let current = match Piece::open(current) {
        Ok(piece) => piece,
        Err(error) => {
            super::show_error(window, "Could not merge the session", &error.to_string());
            return;
        }
    };
    let root = current
        .session
        .dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    // The recording a session was first split from, through parts split
    // again.
    let origin = |session: &Session| {
        let mut found = None;
        let mut session = session.clone();
        for _ in 0..16 {
            let Some(from) = split::read_split_from(&session).ok().flatten() else {
                break;
            };
            let next = Session::open(root.join(&from.session));
            found = Some((from.session, from.title));
            match next {
                Ok(next) => session = next,
                Err(_) => break,
            }
        }
        found
    };
    let current_origin = origin(&current.session);
    let started = current.started_unix_ms().unwrap_or(0);
    let mut pieces = fs::read_dir(&root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter(|entry| entry.path() != current.session.dir)
        .filter_map(|entry| Piece::open(entry.path()).ok())
        .filter(|piece| join::refusal(&piece.session, &piece.manifest).is_none())
        .filter(|piece| piece.samples > 0)
        .collect::<Vec<_>>();
    pieces.sort_by_key(|piece| {
        piece
            .started_unix_ms()
            .map_or(u64::MAX, |start| start.abs_diff(started))
    });

    let list = gtk::ListBox::new();
    list.add_css_class("boxed-list");
    list.set_selection_mode(gtk::SelectionMode::None);
    let row = |piece: &Piece, tag: Option<String>| {
        let row = adw::ActionRow::builder()
            .title(glib_escape(&piece.title))
            .subtitle(format!(
                "{} · {}",
                piece.range_label(),
                format_duration(piece.length_ms())
            ))
            .build();
        let check = gtk::CheckButton::new();
        check.set_valign(gtk::Align::Center);
        row.add_prefix(&check);
        row.set_activatable_widget(Some(&check));
        if let Some(tag) = tag {
            let label = gtk::Label::new(Some(&tag));
            label.add_css_class("merge-tag");
            label.set_valign(gtk::Align::Center);
            row.add_suffix(&label);
        }
        let reason = gtk::Label::new(None);
        reason.add_css_class("caption");
        reason.add_css_class("dim-label");
        reason.set_valign(gtk::Align::Center);
        reason.set_visible(false);
        row.add_suffix(&reason);
        list.append(&row);
        (row, check, reason)
    };
    let (_, own_check, _) = row(&current, Some("this session".into()));
    own_check.set_active(true);
    own_check.set_sensitive(false);
    let candidates = Rc::new(RefCell::new(Vec::new()));
    for piece in pieces {
        let tag = origin(&piece.session)
            .filter(|origin| Some(origin) == current_origin.as_ref())
            .map(|(_, title)| format!("split from {title}"));
        let (row, check, reason) = row(&piece, tag);
        let blocked = queue_reason(&piece.session.dir)
            .map(str::to_owned)
            .or_else(|| {
                join::overlap(&piece, &current).then(|| format!("overlaps “{}”", current.title))
            });
        candidates.borrow_mut().push(Candidate {
            piece,
            check,
            row,
            reason,
            blocked,
        });
    }
    if candidates.borrow().is_empty() {
        let empty = gtk::Label::new(Some("There is no other recording to merge with."));
        empty.add_css_class("dim-label");
        empty.set_margin_top(12);
        empty.set_margin_bottom(12);
        list.append(&empty);
    }

    let dialog = adw::Dialog::builder()
        .title("Merge")
        .content_width(560)
        .follows_content_size(true)
        .presentation_mode(adw::DialogPresentationMode::Floating)
        .build();
    let body = gtk::Box::new(gtk::Orientation::Vertical, 10);
    body.set_margin_top(24);
    body.set_margin_bottom(24);
    body.set_margin_start(24);
    body.set_margin_end(24);
    body.set_size_request(540, -1);
    let heading = gtk::Label::new(Some(&format!("Merge “{}” with…", current.title)));
    heading.add_css_class("title-2");
    heading.set_wrap(true);
    heading.set_justify(gtk::Justification::Center);
    body.append(&heading);
    let lead = gtk::Label::new(Some(
        "Recordings that have not been processed yet, nearest in time first.",
    ));
    lead.add_css_class("dim-label");
    lead.set_wrap(true);
    lead.set_justify(gtk::Justification::Center);
    body.append(&lead);
    let scroll = gtk::ScrolledWindow::builder()
        .child(&list)
        .hscrollbar_policy(gtk::PolicyType::Never)
        .propagate_natural_height(true)
        .max_content_height(340)
        .build();
    body.append(&scroll);
    let summary = gtk::Label::new(None);
    summary.set_wrap(true);
    summary.set_xalign(0.0);
    summary.set_max_width_chars(60);
    summary.add_css_class("merge-summary");
    body.append(&summary);
    let actions = gtk::Box::new(gtk::Orientation::Horizontal, 10);
    actions.set_halign(gtk::Align::Center);
    actions.set_margin_top(6);
    let cancel = gtk::Button::with_label("Cancel");
    cancel.add_css_class("pill-button");
    let dialog_for_cancel = dialog.clone();
    cancel.connect_clicked(move |_| {
        dialog_for_cancel.close();
    });
    actions.append(&cancel);
    let confirm = gtk::Button::with_label("Preview merge");
    confirm.add_css_class("pill-button");
    confirm.add_css_class("suggested-action");
    actions.append(&confirm);
    body.append(&actions);
    dialog.set_child(Some(&body));
    dialog.set_default_widget(Some(&confirm));

    let current = Rc::new(current);
    let update = {
        let candidates = candidates.clone();
        let current = current.clone();
        let summary = summary.clone();
        let confirm = confirm.clone();
        move || {
            let candidates = candidates.borrow();
            let picked = candidates
                .iter()
                .filter(|candidate| candidate.check.is_active())
                .collect::<Vec<_>>();
            for candidate in candidates.iter() {
                let reason = candidate.blocked.clone().or_else(|| {
                    (!candidate.check.is_active())
                        .then(|| {
                            picked
                                .iter()
                                .find(|other| join::overlap(&candidate.piece, &other.piece))
                                .map(|other| format!("overlaps “{}”", other.piece.title))
                        })
                        .flatten()
                });
                candidate.row.set_sensitive(reason.is_none());
                candidate.reason.set_visible(reason.is_some());
                candidate.reason.set_label(reason.as_deref().unwrap_or(""));
                if reason.is_some() {
                    candidate.check.set_active(false);
                }
            }
            let mut chosen = picked
                .iter()
                .map(|candidate| candidate.piece.clone())
                .chain([(*current).clone()])
                .collect::<Vec<_>>();
            join::sort_by_start(&mut chosen);
            confirm.set_sensitive(chosen.len() >= 2);
            if chosen.len() < 2 {
                summary.set_label("Pick at least one more recording.");
                return;
            }
            let samples = chosen.iter().map(|piece| piece.samples).collect::<Vec<_>>();
            let offsets = join::offsets(&samples);
            let length = crate::types::samples_to_ms(
                offsets[offsets.len() - 1] + samples[samples.len() - 1],
            );
            summary.set_markup(&format!(
                "<b>Result:</b> one recording of {} from {} pieces, in recorded order: {}. You can change the order in the preview.",
                format_duration(length),
                chosen.len(),
                glib_escape(
                    &chosen
                        .iter()
                        .map(|piece| format!("“{}”", piece.title))
                        .collect::<Vec<_>>()
                        .join(" → ")
                )
            ));
        }
    };
    let update = Rc::new(update);
    for candidate in candidates.borrow().iter() {
        let update = update.clone();
        candidate.check.connect_toggled(move |_| {
            // Toggling inside the update calls back; wait for it to finish.
            let update = update.clone();
            gtk::glib::idle_add_local_once(move || update());
        });
    }
    update();

    let dialog_for_confirm = dialog.clone();
    confirm.connect_clicked(move |_| {
        dialog_for_confirm.close();
        let mut paths = candidates
            .borrow()
            .iter()
            .filter(|candidate| candidate.check.is_active())
            .map(|candidate| candidate.piece.session.dir.clone())
            .collect::<Vec<_>>();
        paths.push(current.session.dir.clone());
        if paths.len() >= 2 {
            preview(paths);
        }
    });
    dialog.present(Some(window));
}

fn glib_escape(text: &str) -> String {
    gtk::glib::markup_escape_text(text).to_string()
}
