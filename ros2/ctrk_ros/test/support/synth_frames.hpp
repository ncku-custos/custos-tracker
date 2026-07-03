#pragma once

#include <cstdint>
#include <sensor_msgs/msg/image.hpp>

namespace ctrk_ros::test {

// Deterministic hash noise — no RNG state, same value for (x,y,c) everywhere.
inline uint8_t hash8(uint32_t x, uint32_t y, uint32_t c) {
  uint32_t h = x * 374761393u + y * 668265263u + c * 2246822519u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return static_cast<uint8_t>(h >> 16);
}

// Model-free tracking scene (the same idea as tests/support/synth.hpp in the
// core, without OpenCV): a bright textured square translating over dark
// static noise. The texture is anchored to the target so it moves rigidly —
// exactly what a correlation filter locks onto.
struct SynthScene {
  int width = 320;
  int height = 240;
  int target_size = 48;
  float x0 = 40.f, y0 = 100.f;  // target top-left at frame 0
  float dx = 2.f, dy = 0.f;     // per-frame translation

  float target_x(int idx) const { return x0 + dx * static_cast<float>(idx); }
  float target_y(int idx) const { return y0 + dy * static_cast<float>(idx); }

  sensor_msgs::msg::Image frame(int idx) const {
    sensor_msgs::msg::Image img;
    img.encoding = "bgr8";
    img.width = static_cast<uint32_t>(width);
    img.height = static_cast<uint32_t>(height);
    img.step = static_cast<uint32_t>(width * 3);
    img.data.resize(static_cast<size_t>(img.step) * img.height);
    const int tx = static_cast<int>(target_x(idx));
    const int ty = static_cast<int>(target_y(idx));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const bool in_target = x >= tx && x < tx + target_size && y >= ty && y < ty + target_size;
        for (uint32_t c = 0; c < 3; ++c) {
          const uint8_t v =
              in_target ? static_cast<uint8_t>(160 + hash8(static_cast<uint32_t>(x - tx),
                                                           static_cast<uint32_t>(y - ty), c) %
                                                         96)
                        : static_cast<uint8_t>(
                              hash8(static_cast<uint32_t>(x), static_cast<uint32_t>(y), c) % 96);
          img.data[static_cast<size_t>(y) * img.step + static_cast<size_t>(x) * 3 + c] = v;
        }
      }
    }
    const auto t_ns = static_cast<int64_t>(static_cast<double>(idx) / 30.0 * 1e9);
    img.header.stamp.sec = static_cast<int32_t>(t_ns / 1000000000LL);
    img.header.stamp.nanosec = static_cast<uint32_t>(t_ns % 1000000000LL);
    return img;
  }
};

}  // namespace ctrk_ros::test
