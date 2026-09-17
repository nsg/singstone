#include <sycl/sycl.hpp>

#include <cstdint>

int main() noexcept {
  try {
    for (const auto &platform : sycl::platform::get_platforms()) {
      for (const auto &device : platform.get_devices()) {
        const auto vendor_id =
            device.get_info<sycl::info::device::vendor_id>();
        if (device.is_gpu() && vendor_id == UINT32_C(0x8086) &&
            device.has(sycl::aspect::fp16)) {
          // Submit a kernel so the probe validates device access and execution,
          // rather than merely finding an entry in the device list.
          sycl::queue queue(device);
          queue.single_task([=]() {}).wait_and_throw();
          return 0;
        }
      }
    }
  } catch (...) {
  }
  return 1;
}
