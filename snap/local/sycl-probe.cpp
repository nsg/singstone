#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <string>

namespace {

bool supported_by_oneapi(const std::string &name) {
  if (const char *override = std::getenv("SINGSTONE_ALLOW_UNSUPPORTED_GPU");
      override != nullptr && std::string(override) == "1") {
    return true;
  }

  // oneAPI 2026 supports integrated graphics from 11th-generation Intel Core
  // onward. Older Gen9 devices can enumerate and run a trivial kernel, then
  // abort in the substantially larger ggml kernels. Device names do not expose
  // the CPU generation directly, so accept current product families and the
  // UHD names used by supported 11th-generation and newer processors.
  for (const char *marker : {"Xe Graphics", "Arc", "Data Center GPU",
                             "Server GPU"}) {
    if (name.find(marker) != std::string::npos) {
      return true;
    }
  }
  if (name == "Intel(R) Graphics" || name == "Intel(R) UHD Graphics") {
    return true;
  }
  for (const char *model : {"UHD Graphics 710", "UHD Graphics 730",
                            "UHD Graphics 750", "UHD Graphics 770"}) {
    if (name.find(model) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() noexcept {
  try {
    std::cerr << "stage=enumerate\n";
    const auto platforms = sycl::platform::get_platforms();
    std::cerr << "platforms=" << platforms.size() << '\n';
    bool unsupported_device = false;
    for (const auto &platform : platforms) {
      const auto platform_name =
          platform.get_info<sycl::info::platform::name>();
      const auto devices = platform.get_devices();
      std::cerr << "platform=" << platform_name
                << " devices=" << devices.size() << '\n';
      for (const auto &device : devices) {
        const auto vendor_id =
            device.get_info<sycl::info::device::vendor_id>();
        const auto name = device.get_info<sycl::info::device::name>();
        const bool supported = supported_by_oneapi(name);
        std::cerr << "device=" << name << " gpu=" << device.is_gpu()
                  << " vendor=0x" << std::hex << vendor_id << std::dec
                  << " fp16=" << device.has(sycl::aspect::fp16)
                  << " supported=" << supported << '\n';
        if (device.is_gpu() && vendor_id == UINT32_C(0x8086) &&
            device.has(sycl::aspect::fp16) && supported) {
          std::cerr << "stage=queue device=" << name << '\n';
          // Submit a kernel so the probe validates device access and execution,
          // rather than merely finding an entry in the device list.
          sycl::queue queue(device);
          std::cerr << "stage=kernel\n";
          queue.single_task([=]() {}).wait_and_throw();
          std::cerr << "stage=success\n";
          std::cout << name << '\n' << std::flush;

          // Some Level Zero/Unified Runtime combinations crash while global
          // objects are torn down, after successful work has completed. This
          // disposable probe owns no resources that need to survive, so avoid
          // runtime shutdown and report the verified device immediately.
          std::_Exit(0);
        }
        if (device.is_gpu() && vendor_id == UINT32_C(0x8086) && !supported) {
          unsupported_device = true;
          std::cerr << "stage=unsupported-device device=" << name << '\n';
        }
      }
    }
    std::cerr << "stage=no-compatible-device\n";
    return unsupported_device ? 2 : 1;
  } catch (const std::exception &error) {
    std::cerr << "stage=exception error=" << error.what() << '\n';
  } catch (...) {
    std::cerr << "stage=exception error=unknown\n";
  }
  return 1;
}
