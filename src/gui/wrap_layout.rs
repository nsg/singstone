use gtk::subclass::prelude::*;
use gtk::{glib, prelude::*};
use gtk4 as gtk;
use std::cell::Cell;

#[derive(Debug, PartialEq, Eq)]
struct FlowItem {
    index: usize,
    x: i32,
    width: i32,
}

#[derive(Debug, PartialEq, Eq)]
struct FlowLine {
    items: Vec<FlowItem>,
}

fn flow_lines(natural_widths: &[i32], container_width: i32, column_spacing: i32) -> Vec<FlowLine> {
    let container_width = container_width.max(0);
    let column_spacing = column_spacing.max(0);
    let mut lines = Vec::new();
    let mut items = Vec::new();
    let mut line_width = 0i32;

    for (index, natural_width) in natural_widths.iter().copied().enumerate() {
        let natural_width = natural_width.max(0);
        let next_x = line_width.saturating_add(column_spacing);
        if !items.is_empty() && next_x.saturating_add(natural_width) > container_width {
            lines.push(FlowLine { items });
            items = Vec::new();
            line_width = 0;
        }

        let x = if items.is_empty() {
            0
        } else {
            line_width.saturating_add(column_spacing)
        };
        let width = natural_width.min(container_width);
        line_width = x.saturating_add(natural_width);
        items.push(FlowItem { index, x, width });
    }

    if !items.is_empty() {
        lines.push(FlowLine { items });
    }
    lines
}

fn natural_flow_width(natural_widths: &[i32], column_spacing: i32) -> i32 {
    natural_widths
        .iter()
        .copied()
        .fold(0i32, i32::saturating_add)
        .saturating_add(
            column_spacing
                .max(0)
                .saturating_mul(natural_widths.len().saturating_sub(1) as i32),
        )
}

mod imp {
    use super::*;

    #[derive(Default)]
    pub struct WrapLayout {
        pub column_spacing: Cell<i32>,
        pub row_spacing: Cell<i32>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for WrapLayout {
        const NAME: &'static str = "SingstoneWrapLayout";
        type Type = super::WrapLayout;
        type ParentType = gtk::LayoutManager;
    }

    impl ObjectImpl for WrapLayout {}

    impl LayoutManagerImpl for WrapLayout {
        fn request_mode(&self, _widget: &gtk::Widget) -> gtk::SizeRequestMode {
            gtk::SizeRequestMode::HeightForWidth
        }

        fn measure(
            &self,
            widget: &gtk::Widget,
            orientation: gtk::Orientation,
            for_size: i32,
        ) -> (i32, i32, i32, i32) {
            let children = layout_children(widget);
            let widths = natural_widths(&children);

            if orientation == gtk::Orientation::Horizontal {
                let minimum = children
                    .iter()
                    .map(|child| child.measure(gtk::Orientation::Horizontal, -1).0)
                    .max()
                    .unwrap_or(0);
                let natural = natural_flow_width(&widths, self.column_spacing.get());
                return (minimum, natural, -1, -1);
            }

            let width = if for_size < 0 {
                natural_flow_width(&widths, self.column_spacing.get())
            } else {
                for_size
            };
            let lines = flow_lines(&widths, width, self.column_spacing.get());
            let natural = total_height(&children, &lines, self.row_spacing.get());
            (natural, natural, -1, -1)
        }

        fn allocate(&self, widget: &gtk::Widget, width: i32, _height: i32, _baseline: i32) {
            let children = layout_children(widget);
            let widths = natural_widths(&children);
            let lines = flow_lines(&widths, width, self.column_spacing.get());
            let mut y = 0i32;

            for line in &lines {
                let line_height = line_height(&children, line);
                for item in &line.items {
                    children[item.index].size_allocate(
                        &gtk::Allocation::new(item.x, y, item.width, line_height),
                        -1,
                    );
                }
                y = y
                    .saturating_add(line_height)
                    .saturating_add(self.row_spacing.get());
            }
        }
    }

    fn layout_children(widget: &gtk::Widget) -> Vec<gtk::Widget> {
        let mut children = Vec::new();
        let mut child = widget.first_child();
        while let Some(current) = child {
            child = current.next_sibling();
            if current.should_layout() {
                children.push(current);
            }
        }
        children
    }

    fn natural_widths(children: &[gtk::Widget]) -> Vec<i32> {
        children
            .iter()
            .map(|child| child.measure(gtk::Orientation::Horizontal, -1).1)
            .collect()
    }

    fn line_height(children: &[gtk::Widget], line: &FlowLine) -> i32 {
        line.items
            .iter()
            .map(|item| {
                children[item.index]
                    .measure(gtk::Orientation::Vertical, item.width)
                    .1
            })
            .max()
            .unwrap_or(0)
    }

    fn total_height(children: &[gtk::Widget], lines: &[FlowLine], row_spacing: i32) -> i32 {
        lines
            .iter()
            .map(|line| line_height(children, line))
            .fold(0i32, i32::saturating_add)
            .saturating_add(
                row_spacing
                    .max(0)
                    .saturating_mul(lines.len().saturating_sub(1) as i32),
            )
    }
}

glib::wrapper! {
    pub struct WrapLayout(ObjectSubclass<imp::WrapLayout>)
        @extends gtk::LayoutManager;
}

impl WrapLayout {
    pub fn new(column_spacing: i32, row_spacing: i32) -> Self {
        let layout: Self = glib::Object::new();
        layout.imp().column_spacing.set(column_spacing.max(0));
        layout.imp().row_spacing.set(row_spacing.max(0));
        layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_lines_break_at_the_available_width() {
        let lines = flow_lines(&[30, 20, 25, 40], 60, 5);
        let line_items = lines
            .iter()
            .map(|line| line.items.iter().map(|item| item.index).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        assert_eq!(lines.len(), 3);
        assert_eq!(line_items, vec![vec![0, 1], vec![2], vec![3]]);
    }
}
