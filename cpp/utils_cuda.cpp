#include <torch/extension.h>

#include <cassert>
#include <cstdint>

#include <vector>

uint64_t pack(const std::vector<uint16_t> &integer_list) {
  assert(integer_list.size() <= 4);

  uint64_t packed_integer = 0;

  packed_integer |= static_cast<uint64_t>(integer_list[0]) << 0;
  packed_integer |= static_cast<uint64_t>(integer_list[1]) << 16;
  packed_integer |= static_cast<uint64_t>(integer_list[2]) << 32;
  packed_integer |= static_cast<uint64_t>(integer_list[3]) << 48;

  return packed_integer;
}

std::array<uint16_t, 4> unpack_as_tensor(uint64_t packed_integer) {
  uint16_t first = static_cast<uint16_t>(packed_integer >> 0);
  uint16_t second = static_cast<uint16_t>(packed_integer >> 16);
  uint16_t third = static_cast<uint16_t>(packed_integer >> 32);
  uint16_t fourth = static_cast<uint16_t>(packed_integer >> 48);
  
  return {first, second, third, fourth};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack, "Pack utils");
  m.def("unpack_fast", &unpack_as_tensor, "Unpack utils");
}
