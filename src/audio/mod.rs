//! PipeWire capture: real-time callback -> bounded queue -> writer thread.

pub mod capture;
pub mod devices;
pub mod pipewire;
pub mod record;
pub mod timing;
pub mod writer;
