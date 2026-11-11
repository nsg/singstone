//! Sessions grouped by the week they were recorded in, weeks starting Monday.

use crate::meeting::days_from_civil;
use std::ops::Range;

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
/// Weeks further back than this are named by their dates alone.
const RELATIVE_WEEKS: i64 = 8;

/// Days since 1970-01-01 of the local date a `YYYY-MM-DD…` wall clock starts with.
pub fn local_day(wallclock: &str) -> Option<i64> {
    let date = wallclock.get(..10)?;
    let mut parts = date.split('-');
    let year = parts.next()?.parse::<i32>().ok()?;
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    ((1..=12).contains(&month) && (1..=31).contains(&day))
        .then(|| days_from_civil(year, month, day))
}

pub fn civil_day(year: i32, month: u32, day: u32) -> i64 {
    days_from_civil(year, month, day)
}

/// The week a day falls in; 1970-01-01 was a Thursday.
pub fn week_of(day: i64) -> i64 {
    (day + 3).div_euclid(7)
}

/// What a group of sessions is called, and the dates it covers when the name
/// does not say them.
pub fn heading(week: i64, today: i64) -> (String, Option<String>) {
    let dates = week_dates(week, today);
    match week_of(today) - week {
        ..=0 => ("This week".into(), Some(dates)),
        1 => ("Last week".into(), Some(dates)),
        ago @ 2..=RELATIVE_WEEKS => (format!("{ago} weeks ago"), Some(dates)),
        _ => (dates, None),
    }
}

fn week_dates(week: i64, today: i64) -> String {
    let monday = week * 7 - 3;
    let (first_year, first_month, first_day) = civil_from_days(monday);
    let (last_year, last_month, last_day) = civil_from_days(monday + 6);
    let (this_year, _, _) = civil_from_days(today);
    let first_month_name = MONTHS[first_month as usize - 1];
    let last_month_name = MONTHS[last_month as usize - 1];
    if first_year != last_year {
        format!(
            "{first_day} {first_month_name} {first_year} – {last_day} {last_month_name} {last_year}"
        )
    } else {
        let year = if last_year == this_year {
            String::new()
        } else {
            format!(" {last_year}")
        };
        if first_month == last_month {
            format!("{first_day}–{last_day} {last_month_name}{year}")
        } else {
            format!("{first_day} {first_month_name} – {last_day} {last_month_name}{year}")
        }
    }
}

/// The days a month's calendar shows: whole weeks, from the Monday on or
/// before the 1st to the Sunday on or after its last day.
pub fn month_days(year: i32, month: u32) -> Range<i64> {
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    let first = week_of(days_from_civil(year, month, 1));
    let last = week_of(days_from_civil(next_year, next_month, 1) - 1);
    first * 7 - 3..(last + 1) * 7 - 3
}

pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(value: &str) -> i64 {
        local_day(value).expect("date")
    }

    #[test]
    fn weeks_start_on_monday_in_local_time() {
        assert_eq!(
            week_of(day("2026-11-09T00:00:00+01:00")),
            week_of(day("2026-11-15T23:59:00+01:00"))
        );
        assert_eq!(
            week_of(day("2026-11-08T23:59:00+01:00")) + 1,
            week_of(day("2026-11-09T08:00:00+01:00"))
        );
        assert_eq!(local_day("session-x"), None);
        assert_eq!(local_day("2026-13-01T00:00:00Z"), None);
    }

    #[test]
    fn dates_round_trip() {
        for value in ["1970-01-01", "2000-02-29", "2026-11-11", "1969-12-31"] {
            let parsed = day(value);
            let (year, month, date) = civil_from_days(parsed);
            assert_eq!(format!("{year:04}-{month:02}-{date:02}"), value);
        }
    }

    #[test]
    fn month_calendars_pad_to_whole_weeks() {
        let shown = |year, month| {
            let days = month_days(year, month);
            let first = civil_from_days(days.start);
            let last = civil_from_days(days.end - 1);
            (first, last, (days.end - days.start) / 7)
        };
        assert_eq!(shown(2026, 11), ((2026, 10, 26), (2026, 12, 6), 6));
        assert_eq!(shown(2026, 12), ((2026, 11, 30), (2027, 1, 3), 5));
        assert_eq!(shown(2027, 2), ((2027, 2, 1), (2027, 2, 28), 4));
        assert_eq!(shown(2026, 6), ((2026, 6, 1), (2026, 7, 5), 5));
    }

    #[test]
    fn headings_name_recent_weeks_relative_to_today() {
        let today = day("2026-11-11");
        let heading_of = |value: &str| heading(week_of(day(value)), today);
        assert_eq!(
            heading_of("2026-11-09"),
            ("This week".into(), Some("9–15 Nov".into()))
        );
        assert_eq!(
            heading_of("2026-11-08"),
            ("Last week".into(), Some("2–8 Nov".into()))
        );
        assert_eq!(
            heading_of("2026-10-28"),
            ("2 weeks ago".into(), Some("26 Oct – 1 Nov".into()))
        );
        assert_eq!(heading_of("2026-09-16").0, "8 weeks ago");
        assert_eq!(heading_of("2026-09-13"), ("7–13 Sep".into(), None));
        assert_eq!(heading_of("2025-11-12"), ("10–16 Nov 2025".into(), None));
        assert_eq!(
            heading_of("2025-12-31"),
            ("29 Dec 2025 – 4 Jan 2026".into(), None)
        );
        assert_eq!(heading_of("2026-11-20").0, "This week");
    }
}
