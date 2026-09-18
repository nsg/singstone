#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>

int main() noexcept {
  try {
    std::cerr << "stage=enumerate\n";
    const auto platforms = sycl::platform::get_platforms();
    std::cerr << "platforms=" << platforms.size() << '\n';
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
        std::cerr << "device=" << name << " gpu=" << device.is_gpu()
                  << " vendor=0x" << std::hex << vendor_id << std::dec
                  << " fp16=" << device.has(sycl::aspect::fp16) << '\n';
        if (device.is_gpu() && vendor_id == UINT32_C(0x8086) &&
            device.has(sycl::aspect::fp16)) {
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
      }
    }
    std::cerr << "stage=no-compatible-device\n";
  } catch (const std::exception &error) {
    std::cerr << "stage=exception error=" << error.what() << '\n';
  } catch (...) {
    std::cerr << "stage=exception error=unknown\n";
  }
  return 1;
}
