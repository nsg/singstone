//! Sessions laid out on a month calendar, weeks starting Monday.

use super::{SessionSummary, format_duration, status_pill, weeks};
use adw::prelude::*;
use gtk4 as gtk;
use libadwaita as adw;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const WEEKDAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
/// The order statuses are explained in, following a session's life.
const STATUS_ORDER: [&str; 9] = [
    "Recording",
    "Recorded",
    "Queued",
    "Processing",
    "Failed",
    "Processed",
    "Done",
    "Archived",
    "Unavailable",
];

type Loader = Rc<dyn Fn() -> Vec<SessionSummary>>;
type Opener = Rc<dyn Fn(&Path)>;

#[derive(Clone)]
pub struct CalendarView {
    pub root: gtk::Box,
    title: gtk::Label,
    today: gtk::Button,
    legend: gtk::Box,
    grid: gtk::Grid,
    month: Rc<Cell<(i32, u32)>>,
    load: Loader,
    open: Opener,
}

impl CalendarView {
    pub fn new(load: Loader, open: Opener) -> Self {
        let root = gtk::Box::new(gtk::Orientation::Vertical, 10);
        root.set_margin_top(16);
        root.set_margin_bottom(16);
        root.set_margin_start(16);
        root.set_margin_end(16);

        let bar = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        let previous = gtk::Button::from_icon_name("go-previous-symbolic");
        previous.set_tooltip_text(Some("Previous month"));
        let next = gtk::Button::from_icon_name("go-next-symbolic");
        next.set_tooltip_text(Some("Next month"));
        let arrows = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        arrows.add_css_class("linked");
        arrows.append(&previous);
        arrows.append(&next);
        bar.append(&arrows);
        let today = gtk::Button::with_label("Today");
        today.set_tooltip_text(Some("Show this month"));
        bar.append(&today);
        let title = gtk::Label::new(None);
        title.add_css_class("title-2");
        title.set_margin_start(8);
        bar.append(&title);
        // A row of its own, so the statuses never hold the page wider than the grid.
        let legend = gtk::Box::new(gtk::Orientation::Horizontal, 6);
        // Keeps the grid still when a month without sessions empties it.
        legend.set_size_request(-1, 20);
        root.append(&bar);
        root.append(&legend);

        let grid = gtk::Grid::builder()
            .column_homogeneous(true)
            .vexpand(true)
            .build();
        grid.add_css_class("calendar-grid");
        let scroll = gtk::ScrolledWindow::builder()
            .vexpand(true)
            .hscrollbar_policy(gtk::PolicyType::Never)
            .child(&grid)
            .build();
        root.append(&scroll);

        let view = Self {
            root,
            title,
            today,
            legend,
            grid,
            month: Rc::new(Cell::new(this_month())),
            load,
            open,
        };
        for (button, step) in [(&previous, -1), (&next, 1)] {
            let view = view.clone();
            button.connect_clicked(move |_| {
                view.month.set(shift_month(view.month.get(), step));
                view.refresh();
            });
        }
        let view_for_today = view.clone();
        view.today.connect_clicked(move |_| {
            view_for_today.month.set(this_month());
            view_for_today.refresh();
        });
        view
    }

    pub fn refresh(&self) {
        let (year, month) = self.month.get();
        self.title
            .set_label(&format!("{} {year}", MONTHS[month as usize - 1]));
        self.today.set_sensitive((year, month) != this_month());
        let days = weeks::month_days(year, month);
        let sessions = (self.load)();
        let by_day = sessions_by_day(&sessions, days.start..days.end);

        while let Some(child) = self.legend.first_child() {
            self.legend.remove(&child);
        }
        for (status, class) in legend(by_day.values().flatten().copied()) {
            self.legend.append(&status_pill(status, class));
        }

        while let Some(child) = self.grid.first_child() {
            self.grid.remove(&child);
        }
        for (column, name) in WEEKDAYS.iter().enumerate() {
            let label = gtk::Label::new(Some(name));
            label.add_css_class("calendar-weekday");
            self.grid.attach(&label, column as i32, 0, 1, 1);
        }
        let today = super::local_today();
        for day in days.clone() {
            let offset = (day - days.start) as i32;
            let cell = self.day_cell(day, month, today, by_day.get(&day));
            self.grid.attach(&cell, offset % 7, offset / 7 + 1, 1, 1);
        }
    }

    fn day_cell(
        &self,
        day: i64,
        shown_month: u32,
        today: i64,
        sessions: Option<&Vec<&SessionSummary>>,
    ) -> gtk::Box {
        let (_, month, date) = weeks::civil_from_days(day);
        let cell = gtk::Box::new(gtk::Orientation::Vertical, 3);
        cell.add_css_class("calendar-day");
        cell.set_vexpand(true);
        if month != shown_month {
            cell.add_css_class("calendar-day-outside");
        }
        if day == today {
            cell.add_css_class("calendar-today");
        }
        let number = gtk::Label::new(Some(&if date == 1 {
            format!("{date} {}", &MONTHS[month as usize - 1][..3])
        } else {
            date.to_string()
        }));
        number.add_css_class("calendar-date");
        number.set_halign(gtk::Align::Start);
        cell.append(&number);
        for summary in sessions.into_iter().flatten() {
            cell.append(&self.session_chip(summary));
        }
        cell
    }

    fn session_chip(&self, summary: &SessionSummary) -> gtk::Button {
        let content = gtk::Box::new(gtk::Orientation::Horizontal, 5);
        let time = gtk::Label::new(Some(summary.started.get(11..16).unwrap_or_default()));
        time.add_css_class("calendar-session-time");
        content.append(&time);
        let title = gtk::Label::new(Some(&summary.title));
        title.set_xalign(0.0);
        title.set_hexpand(true);
        title.set_ellipsize(gtk::pango::EllipsizeMode::End);
        content.append(&title);
        let chip = gtk::Button::builder().child(&content).build();
        chip.add_css_class("calendar-session");
        chip.add_css_class(&format!("pill-{}", summary.status_class));
        chip.set_tooltip_text(Some(&format!(
            "{}\n{} · {} · {}",
            summary.title,
            time.label(),
            format_duration(summary.duration_ms),
            summary.status
        )));
        chip.update_property(&[gtk::accessible::Property::Label(&format!(
            "{}, {}, {}",
            summary.title,
            time.label(),
            summary.status
        ))]);
        let path: PathBuf = summary.path.clone();
        let open = self.open.clone();
        chip.connect_clicked(move |_| open(&path));
        chip
    }
}

/// The listed sessions recorded on each shown day, earliest first.
fn sessions_by_day(
    sessions: &[SessionSummary],
    shown: std::ops::Range<i64>,
) -> BTreeMap<i64, Vec<&SessionSummary>> {
    let mut days = BTreeMap::<i64, Vec<&SessionSummary>>::new();
    for summary in sessions {
        if let Some(day) = weeks::local_day(&summary.started)
            && shown.contains(&day)
        {
            days.entry(day).or_default().push(summary);
        }
    }
    for list in days.values_mut() {
        list.sort_by(|a, b| a.started.cmp(&b.started));
    }
    days
}

/// The statuses the shown sessions have, in the order a session goes through them.
fn legend<'a>(
    sessions: impl IntoIterator<Item = &'a SessionSummary>,
) -> Vec<(&'static str, &'static str)> {
    let mut statuses = sessions
        .into_iter()
        .map(|summary| (summary.status, summary.status_class))
        .collect::<Vec<_>>();
    statuses.sort_by_key(|(status, _)| {
        STATUS_ORDER
            .iter()
            .position(|known| known == status)
            .unwrap_or(STATUS_ORDER.len())
    });
    statuses.dedup();
    statuses
}

fn this_month() -> (i32, u32) {
    let (year, month, _) = weeks::civil_from_days(super::local_today());
    (year as i32, month)
}

fn shift_month((year, month): (i32, u32), step: i32) -> (i32, u32) {
    let index = year * 12 + month as i32 - 1 + step;
    (index.div_euclid(12), index.rem_euclid(12) as u32 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(started: &str, status: &'static str) -> SessionSummary {
        SessionSummary {
            path: PathBuf::from(started),
            archived: false,
            title: started.into(),
            started: started.into(),
            duration_ms: 0,
            status,
            status_class: "ok",
        }
    }

    #[test]
    fn months_step_across_years() {
        assert_eq!(shift_month((2026, 12), 1), (2027, 1));
        assert_eq!(shift_month((2026, 1), -1), (2025, 12));
        assert_eq!(shift_month((2026, 11), -1), (2026, 10));
    }

    #[test]
    fn sessions_land_on_their_local_day_earliest_first() {
        let sessions = [
            summary("2026-11-11T14:00:00+01:00", "Done"),
            summary("2026-11-11T09:00:00+01:00", "Recorded"),
            summary("2026-10-26T23:30:00-05:00", "Processed"),
            summary("2026-10-25T10:00:00+01:00", "Processed"),
            summary("session-20261111", "Recorded"),
        ];
        let days = sessions_by_day(&sessions, weeks::month_days(2026, 11));
        let day = |value: &str| weeks::local_day(value).unwrap();
        let starts = |value: &str| {
            days[&day(value)]
                .iter()
                .map(|summary| &summary.started[11..16])
                .collect::<Vec<_>>()
        };
        assert_eq!(starts("2026-11-11"), ["09:00", "14:00"]);
        assert_eq!(starts("2026-10-26"), ["23:30"]);
        assert_eq!(days.len(), 2);
        assert_eq!(
            legend(days.values().flatten().copied()),
            [("Recorded", "ok"), ("Processed", "ok"), ("Done", "ok")]
        );
    }
}
