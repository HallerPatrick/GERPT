#include <torch/extension.h>

#include <cassert>
#include <cstdint>

#include <vector>

uint64_t pack(const std::vector<uint16_t> &integer_list) {
  assert(integer_list.size() <= 4);

  uint64_t packed_integer = 0;
  for (std::size_t i = 0; i < integer_list.size(); ++i) {
    packed_integer |= static_cast<uint64_t>(integer_list[i]) << (16 * i);
  }
  return packed_integer;
}

std::vector<uint16_t> unpack(uint64_t packed_integer) {
  std::vector<uint16_t> integer_list;
  for (std::size_t i = 0; i < 4; ++i) {
    integer_list.push_back(static_cast<uint16_t>(packed_integer >> (16 * i)));
  }
  return integer_list;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack, "Pack utils");
  m.def("unpack", &unpack, "Unpack utils");
}
