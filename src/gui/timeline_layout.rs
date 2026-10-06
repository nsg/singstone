use crate::types::{AudioSource, Utterance};
use gtk::subclass::prelude::*;
use gtk::{glib, prelude::*};
use gtk4 as gtk;
use std::cell::Cell;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct TimelineRow {
    pub start_ms: u64,
    pub mic: Vec<usize>,
    pub system: Vec<usize>,
}

pub(super) fn timeline_rows(utterances: &[Utterance]) -> Vec<TimelineRow> {
    let mut rows: Vec<TimelineRow> = Vec::new();

    for (index, utterance) in utterances.iter().enumerate() {
        let joins_last = rows.last().is_some_and(|row| {
            let other = match utterance.source {
                AudioSource::Mic => &row.system,
                AudioSource::System => &row.mic,
            };
            other
                .iter()
                .any(|other| substantially_overlaps(utterance, &utterances[*other]))
        });

        if !joins_last {
            rows.push(TimelineRow {
                start_ms: utterance.start_ms,
                mic: Vec::new(),
                system: Vec::new(),
            });
        }

        let row = rows.last_mut().expect("a timeline row was just created");
        row.start_ms = row.start_ms.min(utterance.start_ms);
        match utterance.source {
            AudioSource::Mic => row.mic.push(index),
            AudioSource::System => row.system.push(index),
        }
    }

    rows
}

fn substantially_overlaps(left: &Utterance, right: &Utterance) -> bool {
    let left_end = left.end_ms.max(left.start_ms.saturating_add(1));
    let right_end = right.end_ms.max(right.start_ms.saturating_add(1));
    let overlap = left_end
        .min(right_end)
        .saturating_sub(left.start_ms.max(right.start_ms));
    let left_duration = left_end.saturating_sub(left.start_ms);
    let right_duration = right_end.saturating_sub(right.start_ms);
    overlap > 0 && overlap.saturating_mul(2) >= left_duration.min(right_duration)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellAllocation {
    x: i32,
    width: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowAllocation {
    left: CellAllocation,
    marker: CellAllocation,
    right: CellAllocation,
}

fn row_allocation(width: i32, marker_natural_width: i32, one_sided: bool) -> RowAllocation {
    let width = width.max(0);
    let marker_width = marker_natural_width.clamp(0, width);
    let remaining = width - marker_width;
    if one_sided {
        let content = CellAllocation {
            x: marker_width,
            width: remaining,
        };
        return RowAllocation {
            left: content,
            marker: CellAllocation {
                x: 0,
                width: marker_width,
            },
            right: content,
        };
    }

    let left_width = remaining / 2;
    let marker_x = left_width;
    let right_x = marker_x + marker_width;
    RowAllocation {
        left: CellAllocation {
            x: 0,
            width: left_width,
        },
        marker: CellAllocation {
            x: marker_x,
            width: marker_width,
        },
        right: CellAllocation {
            x: right_x,
            width: width - right_x,
        },
    }
}

mod imp {
    use super::*;

    #[derive(Default)]
    pub struct TimelineLayout {
        pub one_sided: Cell<bool>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for TimelineLayout {
        const NAME: &'static str = "SingstoneTimelineLayout";
        type Type = super::TimelineLayout;
        type ParentType = gtk::LayoutManager;
    }

    impl ObjectImpl for TimelineLayout {}

    impl LayoutManagerImpl for TimelineLayout {
        fn request_mode(&self, _widget: &gtk::Widget) -> gtk::SizeRequestMode {
            gtk::SizeRequestMode::HeightForWidth
        }

        fn measure(
            &self,
            widget: &gtk::Widget,
            orientation: gtk::Orientation,
            for_size: i32,
        ) -> (i32, i32, i32, i32) {
            let Some([left, marker, right]) = layout_children(widget) else {
                return (0, 0, -1, -1);
            };

            if orientation == gtk::Orientation::Horizontal {
                let marker_width = marker.measure(orientation, -1).1;
                let left_minimum = visible_width(&left, false);
                let right_minimum = visible_width(&right, false);
                let content_minimum = left_minimum.max(right_minimum);
                let minimum = if self.one_sided.get() {
                    marker_width.saturating_add(content_minimum)
                } else {
                    marker_width.saturating_add(content_minimum.saturating_mul(2))
                };
                let left_width = visible_width(&left, true);
                let right_width = visible_width(&right, true);
                let natural = if self.one_sided.get() {
                    marker_width.saturating_add(left_width.max(right_width))
                } else {
                    marker_width.saturating_add(left_width.max(right_width).saturating_mul(2))
                };
                return (minimum, natural.max(minimum), -1, -1);
            }

            let width = for_size.max(0);
            let marker_width = marker.measure(gtk::Orientation::Horizontal, -1).1;
            let allocation = row_allocation(width, marker_width, self.one_sided.get());
            let height = [
                visible_height(&left, allocation.left.width),
                visible_height(&marker, allocation.marker.width),
                visible_height(&right, allocation.right.width),
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
            (height, height, -1, -1)
        }

        fn allocate(&self, widget: &gtk::Widget, width: i32, height: i32, _baseline: i32) {
            let Some([left, marker, right]) = layout_children(widget) else {
                return;
            };
            let marker_width = marker.measure(gtk::Orientation::Horizontal, -1).1;
            let allocation = row_allocation(width, marker_width, self.one_sided.get());
            allocate_visible(&left, allocation.left, height);
            allocate_visible(&marker, allocation.marker, height);
            allocate_visible(&right, allocation.right, height);
        }
    }

    fn layout_children(widget: &gtk::Widget) -> Option<[gtk::Widget; 3]> {
        let left = widget.first_child()?;
        let marker = left.next_sibling()?;
        let right = marker.next_sibling()?;
        Some([left, marker, right])
    }

    fn visible_width(child: &gtk::Widget, natural: bool) -> i32 {
        if child.should_layout() {
            let (minimum, natural_width, _, _) = child.measure(gtk::Orientation::Horizontal, -1);
            if natural { natural_width } else { minimum }
        } else {
            0
        }
    }

    fn visible_height(child: &gtk::Widget, width: i32) -> i32 {
        if child.should_layout() {
            child.measure(gtk::Orientation::Vertical, width).1
        } else {
            0
        }
    }

    fn allocate_visible(child: &gtk::Widget, allocation: CellAllocation, height: i32) {
        if child.should_layout() {
            child.size_allocate(
                &gtk::Allocation::new(allocation.x, 0, allocation.width, height),
                -1,
            );
        }
    }
}

glib::wrapper! {
    pub struct TimelineLayout(ObjectSubclass<imp::TimelineLayout>)
        @extends gtk::LayoutManager;
}

impl TimelineLayout {
    pub(super) fn new(one_sided: bool) -> Self {
        let layout: Self = glib::Object::new();
        layout.imp().one_sided.set(one_sided);
        layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utterance(source: AudioSource, start_ms: u64, end_ms: u64) -> Utterance {
        Utterance {
            start_ms,
            end_ms,
            source,
            speaker_id: String::new(),
            speaker: String::new(),
            text: String::new(),
            locked: false,
            echo: false,
        }
    }

    #[test]
    fn echo_shares_the_remote_rows_it_repeats() {
        let mut echo = utterance(AudioSource::Mic, 1_100, 1_900);
        echo.echo = true;
        let utterances = vec![utterance(AudioSource::System, 1_000, 2_000), echo];

        assert_eq!(
            timeline_rows(&utterances),
            vec![TimelineRow {
                start_ms: 1_000,
                mic: vec![1],
                system: vec![0],
            }]
        );
    }

    #[test]
    fn short_interjection_inside_a_long_line_shares_its_row() {
        let utterances = vec![
            utterance(AudioSource::System, 1_000, 10_000),
            utterance(AudioSource::Mic, 4_000, 4_300),
        ];

        assert_eq!(timeline_rows(&utterances).len(), 1);
    }

    #[test]
    fn touching_reply_starts_a_new_row() {
        let utterances = vec![
            utterance(AudioSource::Mic, 1_000, 2_000),
            utterance(AudioSource::System, 2_000, 3_000),
        ];

        assert_eq!(timeline_rows(&utterances).len(), 2);
    }

    #[test]
    fn same_side_lines_only_stack_when_the_other_side_bridges_them() {
        let separate = vec![
            utterance(AudioSource::Mic, 0, 1_000),
            utterance(AudioSource::Mic, 200, 800),
        ];
        assert_eq!(timeline_rows(&separate).len(), 2);

        let bridged = vec![
            utterance(AudioSource::System, 0, 2_000),
            utterance(AudioSource::Mic, 100, 700),
            utterance(AudioSource::Mic, 900, 1_500),
        ];
        assert_eq!(timeline_rows(&bridged)[0].mic, vec![1, 2]);
    }

    #[test]
    fn two_sided_allocation_centres_the_marker_and_uses_the_odd_pixel() {
        assert_eq!(
            row_allocation(101, 11, false),
            RowAllocation {
                left: CellAllocation { x: 0, width: 45 },
                marker: CellAllocation { x: 45, width: 11 },
                right: CellAllocation { x: 56, width: 45 },
            }
        );
        assert_eq!(row_allocation(100, 11, false).right.width, 45);
    }

    #[test]
    fn one_sided_allocation_puts_marker_first_and_uses_remaining_width() {
        let allocation = row_allocation(101, 11, true);
        assert_eq!(allocation.marker, CellAllocation { x: 0, width: 11 });
        assert_eq!(allocation.left, CellAllocation { x: 11, width: 90 });
        assert_eq!(allocation.right, allocation.left);
    }

    #[test]
    fn marker_is_clamped_when_the_row_is_too_narrow() {
        let allocation = row_allocation(7, 11, false);
        assert_eq!(allocation.marker, CellAllocation { x: 0, width: 7 });
        assert_eq!(allocation.left.width, 0);
        assert_eq!(allocation.right.width, 0);
    }
}
