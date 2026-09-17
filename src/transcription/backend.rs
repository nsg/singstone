#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WhisperBackend {
    pub badge: &'static str,
    pub description: String,
    pub accelerated: bool,
}

pub fn current() -> WhisperBackend {
    match std::env::var("SINGSTONE_WHISPER_BACKEND").as_deref() {
        Ok("intel-sycl") => intel_sycl_backend(),
        Ok("cpu") => cpu_backend(),
        _ if cfg!(feature = "intel-sycl") => intel_sycl_backend(),
        _ if cfg!(feature = "vulkan") => WhisperBackend {
            badge: "GPU · Vulkan",
            description: "GPU offload — Vulkan".into(),
            accelerated: true,
        },
        _ if cfg!(feature = "openblas") => WhisperBackend {
            badge: "CPU · OpenBLAS",
            description: "CPU — OpenBLAS".into(),
            accelerated: false,
        },
        _ => cpu_backend(),
    }
}

fn intel_sycl_backend() -> WhisperBackend {
    let device = std::env::var("SINGSTONE_WHISPER_DEVICE")
        .ok()
        .filter(|device| !device.trim().is_empty())
        .unwrap_or_else(|| "Intel GPU".into());
    WhisperBackend {
        badge: "Intel GPU · SYCL",
        description: format!("{device} — SYCL / Level Zero (FP16)"),
        accelerated: true,
    }
}

fn cpu_backend() -> WhisperBackend {
    WhisperBackend {
        badge: "CPU",
        description: "CPU transcription".into(),
        accelerated: false,
    }
}
