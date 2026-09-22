use glib::prelude::ToVariant;
use gtk4::{gio, glib};
use std::rc::Rc;

pub const XML: &str = r#"
<node>
  <interface name="io.github.nsg.Singstone.Recorder">
    <method name="StartRecording"/>
    <method name="StopRecording"/>
    <method name="GetStatus">
      <arg type="a{sv}" name="status" direction="out"/>
    </method>
    <signal name="StatusChanged">
      <arg type="a{sv}" name="status"/>
    </signal>
  </interface>
</node>
"#;
pub const OBJECT_PATH: &str = "/io/github/nsg/Singstone/Recorder";
pub const INTERFACE: &str = "io.github.nsg.Singstone.Recorder";

#[derive(Clone, Copy, Default)]
pub struct RecorderStatus {
    pub recording: bool,
    pub stopping: bool,
    pub mic: bool,
    pub system: bool,
    pub mic_level: f64,
    pub system_level: f64,
    pub elapsed: u32,
    pub screenshots: u32,
}

impl RecorderStatus {
    pub fn to_variant(self) -> glib::Variant {
        let dict = glib::VariantDict::new(None);
        dict.insert_value("recording", &self.recording.to_variant());
        dict.insert_value("stopping", &self.stopping.to_variant());
        dict.insert_value("mic", &self.mic.to_variant());
        dict.insert_value("system", &self.system.to_variant());
        dict.insert_value("mic_level", &self.mic_level.to_variant());
        dict.insert_value("system_level", &self.system_level.to_variant());
        dict.insert_value("elapsed", &self.elapsed.to_variant());
        dict.insert_value("screenshots", &self.screenshots.to_variant());
        dict.end()
    }
}

pub struct RecorderHandlers {
    pub start: Rc<dyn Fn() -> bool>,
    pub stop: Rc<dyn Fn()>,
    pub status: Rc<dyn Fn() -> RecorderStatus>,
}

pub fn register(
    connection: &gio::DBusConnection,
    handlers: RecorderHandlers,
) -> Result<gio::RegistrationId, glib::Error> {
    let node = gio::DBusNodeInfo::for_xml(XML)?;
    let interface = node
        .lookup_interface(INTERFACE)
        .expect("recorder D-Bus interface");
    connection
        .register_object(OBJECT_PATH, &interface)
        .method_call(
            move |_connection, _sender, _path, _interface, method, _params, invocation| match method
            {
                "StartRecording" => {
                    if (handlers.start)() {
                        invocation.return_value(None);
                    } else {
                        invocation.return_error(
                            gio::DBusError::Failed,
                            "Recording did not start; open Singstone for details",
                        );
                    }
                }
                "StopRecording" => {
                    (handlers.stop)();
                    invocation.return_value(None);
                }
                "GetStatus" => {
                    let status = (handlers.status)().to_variant();
                    let result = glib::Variant::tuple_from_iter([status]);
                    invocation.return_value(Some(&result));
                }
                _ => invocation.return_error(
                    gio::DBusError::UnknownMethod,
                    &format!("Unknown method {method}"),
                ),
            },
        )
        .build()
}

pub fn emit_status(connection: &gio::DBusConnection, status: RecorderStatus) {
    let status = status.to_variant();
    let parameters = glib::Variant::tuple_from_iter([status]);
    let _ = connection.emit_signal(
        None,
        OBJECT_PATH,
        INTERFACE,
        "StatusChanged",
        Some(&parameters),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_variant_contains_all_fields() {
        let variant = RecorderStatus {
            recording: true,
            stopping: true,
            mic: true,
            system: false,
            mic_level: 0.25,
            system_level: 0.75,
            elapsed: 42,
            screenshots: 3,
        }
        .to_variant();

        assert_eq!(variant.type_().as_str(), "a{sv}");
        let dict = glib::VariantDict::new(Some(&variant));
        assert_eq!(dict.lookup::<bool>("recording").unwrap(), Some(true));
        assert_eq!(dict.lookup::<bool>("stopping").unwrap(), Some(true));
        assert_eq!(dict.lookup::<bool>("mic").unwrap(), Some(true));
        assert_eq!(dict.lookup::<bool>("system").unwrap(), Some(false));
        assert_eq!(dict.lookup::<f64>("mic_level").unwrap(), Some(0.25));
        assert_eq!(dict.lookup::<f64>("system_level").unwrap(), Some(0.75));
        assert_eq!(dict.lookup::<u32>("elapsed").unwrap(), Some(42));
        assert_eq!(dict.lookup::<u32>("screenshots").unwrap(), Some(3));
    }
}
