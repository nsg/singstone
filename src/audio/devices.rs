use pipewire as pw;
use pw::properties::properties;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceInfo {
    pub id: u32,
    pub media_class: String,
    pub name: String,
    pub description: String,
}

impl DeviceInfo {
    /// Human-friendly label for UI lists.
    pub fn label(&self) -> String {
        if self.description.is_empty() || self.description == self.name {
            self.name.clone()
        } else {
            format!("{} ({})", self.description, self.name)
        }
    }
}

/// Enumerate usable PipeWire sources and sinks. Blocks briefly while the
/// registry round-trip completes; callers on a UI thread should run this on a
/// worker.
pub fn enumerate() -> Result<Vec<DeviceInfo>, Box<dyn std::error::Error>> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(Some(properties! {
        *pw::keys::REMOTE_NAME => "pipewire-0"
    }))?;
    let devices = Rc::new(RefCell::new(Vec::new()));
    let pending = Rc::new(Cell::new(None));
    let done = Rc::new(Cell::new(false));
    let callback_done = Rc::clone(&done);
    let callback_pending = Rc::clone(&pending);
    let loop_to_quit = mainloop.clone();
    let _core_listener = core
        .add_listener_local()
        .done(move |id, sequence| {
            if id == pw::core::PW_ID_CORE && callback_pending.get() == Some(sequence) {
                callback_done.set(true);
                loop_to_quit.quit();
            }
        })
        .register();
    let callback_pending = Rc::clone(&pending);
    let callback_core = core.clone();
    let callback_devices = Rc::clone(&devices);
    let registry_slot = Rc::new(RefCell::new(None));
    let listener_slot = Rc::new(RefCell::new(None));
    let callback_registry_slot = Rc::clone(&registry_slot);
    let callback_listener_slot = Rc::clone(&listener_slot);
    let start = mainloop.loop_().add_timer(move |_| {
        if callback_pending.get().is_some() {
            return;
        }
        if let Ok(registry) = callback_core.get_registry_rc() {
            let devices = Rc::clone(&callback_devices);
            let listener = registry
                .add_listener_local()
                .global(move |global| collect_device(global, &devices))
                .register();
            *callback_registry_slot.borrow_mut() = Some(registry);
            *callback_listener_slot.borrow_mut() = Some(listener);
            if let Ok(sequence) = callback_core.sync(0) {
                callback_pending.set(Some(sequence));
            }
        }
    });
    let _ = start.update_timer(Some(std::time::Duration::from_millis(1)), None);
    let timeout_loop = mainloop.downgrade();
    let timeout = mainloop.loop_().add_timer(move |_| {
        if let Some(mainloop) = timeout_loop.upgrade() {
            mainloop.quit();
        }
    });
    let _ = timeout.update_timer(Some(std::time::Duration::from_secs(2)), None);
    while !done.get() {
        mainloop.run();
        if !done.get() {
            break;
        }
    }

    let mut result = devices.borrow().clone();
    result.sort();
    Ok(result)
}

pub fn list() -> Result<(), Box<dyn std::error::Error>> {
    for device in enumerate()? {
        println!(
            "{}  {}  {}  \"{}\"",
            device.id, device.media_class, device.name, device.description
        );
    }
    println!("Use `default` for WirePlumber auto-connect or `none` to disable a stream.");
    Ok(())
}

fn collect_device(
    global: &pw::registry::GlobalObject<&pw::spa::utils::dict::DictRef>,
    devices: &RefCell<Vec<DeviceInfo>>,
) {
    if global.type_ != pw::types::ObjectType::Node {
        return;
    }
    let Some(props) = global.props else {
        return;
    };
    let Some(media_class) = props.get(*pw::keys::MEDIA_CLASS) else {
        return;
    };
    if !matches!(
        media_class,
        "Audio/Source" | "Audio/Source/Virtual" | "Audio/Sink" | "Audio/Duplex"
    ) {
        return;
    }
    let name = props.get(*pw::keys::NODE_NAME).unwrap_or("unknown");
    let description = props
        .get(*pw::keys::NODE_DESCRIPTION)
        .or_else(|| props.get(*pw::keys::NODE_NICK))
        .unwrap_or(name);
    devices.borrow_mut().push(DeviceInfo {
        id: global.id,
        media_class: media_class.to_owned(),
        name: name.to_owned(),
        description: description.to_owned(),
    });
}
