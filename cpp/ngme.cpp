#include <torch/extension.h>


/**
 * Pack a vector of exactly 4 integers into one
 *
 *
 * @param 16 bit integer vector
 * @returns packed 64 bit integer
 */
uint64_t pack_4_gram(const std::vector<uint16_t> &integer_list) {
  TORCH_CHECK(integer_list.size() == 4, "Four integers required");

  uint64_t packed_integer = 0;

  packed_integer |= static_cast<uint64_t>(integer_list[0]) << 0;
  packed_integer |= static_cast<uint64_t>(integer_list[1]) << 16;
  packed_integer |= static_cast<uint64_t>(integer_list[2]) << 32;
  packed_integer |= static_cast<uint64_t>(integer_list[3]) << 48;

  return packed_integer;
}

uint64_t pack_4_gram(const torch::Tensor &integer_list) {
  TORCH_CHECK(integer_list.size(0) == 4, "Four integers required");

  int64_t *data = integer_list.data_ptr<int64_t>();

  uint64_t packed_integer = 0;

  packed_integer |= static_cast<uint64_t>(data[0]) << 0;

  packed_integer |= static_cast<uint64_t>(data[1]) << 16;
  packed_integer |= static_cast<uint64_t>(data[2]) << 32;
  packed_integer |= static_cast<uint64_t>(data[3]) << 48;

  return packed_integer;
}

/**
 * Unpack integer into exactly 4 integers
 *
 * @param 64 bit packed integer
 * @returns array of 4 16bit integers
 */
std::array<uint16_t, 4> unpack_4_gram(uint64_t packed_integer) {
  uint16_t first = static_cast<uint16_t>(packed_integer >> 0);
  uint16_t second = static_cast<uint16_t>(packed_integer >> 16);
  uint16_t third = static_cast<uint16_t>(packed_integer >> 32);
  uint16_t fourth = static_cast<uint16_t>(packed_integer >> 48);
  
  return {first, second, third, fourth};
}


/**
 * Pack a vector of up to 4 integers into one
 *
 *
 * @param 16 bit integer vector
 * @returns packed 64 bit integer
 */
uint64_t pack(const std::vector<uint16_t> &integer_list) {
  auto ngram = integer_list.size();
  assert(ngram <= 4);

  if (ngram == 4)
    return pack_4_gram(integer_list);

  uint64_t packed_integer = 0;
  for (std::size_t i = 0; i < integer_list.size(); ++i) {
    packed_integer |= static_cast<uint64_t>(integer_list[i]) << (16 * i);
  }
  return packed_integer;
}

uint64_t pack_t(const torch::Tensor &integer_list) {
  auto ngram = integer_list.size(0);

  TORCH_CHECK(ngram <= 4, "Only support for 4 integers");

  if (ngram == 4) {
    return pack_4_gram(integer_list);
  }

  int64_t *data = integer_list.data_ptr<int64_t>();

  uint64_t packed_integer = 0;
  for (std::size_t i = 0; i < (std::size_t)integer_list.size(0); ++i) {
    packed_integer |= static_cast<uint64_t>(data[i]) << (16 * i);
  }
  return packed_integer;

}

/**
 * Unpack integer into up to 4 integers
 *
 * @param 64 bit packed integer
 * @returns array of 16bit integers
 */
std::vector<uint16_t> unpack(uint64_t packed_integer) {

  std::vector<uint16_t> integer_list;
  for (std::size_t i = 0; i < 4; ++i) {
    integer_list.push_back(static_cast<uint16_t>(packed_integer >> (16 * i)));
  }
  return integer_list;
}

/**
 * Pack a pytorch tensor with 2 dimensions into single dimension tensor
 *
 *
 * @param Tensor with 2 dimensions
 * @returns Packed tensor with 1 dimension
 */
torch::Tensor pack_tensor(const torch::Tensor &self) {

  // Expect 2 dimensions [ngram, sequence]
  TORCH_CHECK(self.dim() == 2, "Tensor is not 2 dimensional");

  auto seq_len = self.size(1);

  std::vector<int64_t> packed_vec;

  for(int i = 0; i < seq_len; ++i) {
    // Index column to get n-grams of timestep
    torch::Tensor col = self.index({torch::indexing::Slice(), i}).contiguous();
  
    uint64_t packed_int = pack_t(col);
  
    // Implicit cast to iint64_t, no unsigned types allows in torch :(
    packed_vec.push_back(packed_int);
  }

  return torch::tensor(packed_vec).to(torch::kInt64);
}

/**
 * Unpack a 1D tensor 
 *
 *
 * @param packed tensor of 1D
 * @returns tensor with dim [ngram, seq]
 */
torch::Tensor unpack_tensor(const torch::Tensor &self) {

  TORCH_CHECK(self.dim() == 1, "Tensor is not 1 dimensional");

  
  std::vector<torch::Tensor> unpacked_vecs;

  int64_t *data_self = self.data_ptr<int64_t>();

  for (int i = 0; i < self.size(0); ++i) {

    uint64_t packed_i = data_self[i];
    
    auto unpacked_vec = unpack(packed_i);
    
    // TODO: Do we need to "cast" vector to double?
    std::vector<double> unpacked_int_vec(unpacked_vec.begin(), unpacked_vec.end());
    auto t = torch::tensor(unpacked_int_vec).unsqueeze(0);
    unpacked_vecs.push_back(t);
  }

  return torch::cat(unpacked_vecs).t().to(torch::kI64);
}

/**
 * Unpack a tensor that includes a batch dimension. Expected batch dimension second.
 *
 *
 * @params packed tensor with batch dimension
 * @returns unpacked tensor with dimension [ngram, seq, batch]
 */
torch::Tensor unpack_batched_tensor(const torch::Tensor &self) {

  // Expect 2 dimensions [sequence, batch]
  TORCH_CHECK(self.dim() == 2, "Tensor is not 2 dimensional");

  auto batch = self.size(1);

  std::vector<torch::Tensor> unpacked_ts;

  for(int i = 0; i < batch; ++i) {
    // Index column to get n-grams of timestep
    torch::Tensor col = self.index({torch::indexing::Slice(), i}).contiguous();

    unpacked_ts.push_back(unpack_tensor(col).unsqueeze(-1));
  }

  // Concat all sequence tensors on batch dimension
  return torch::cat(unpacked_ts, -1);
}


/**
 * Multihot encoding function, can either recieve a packed or unpacked sequence.
 *
 * @params Index tensor
 * @params Number of classes
 * @params Wether index tensor is backed or not
 *
 * @returns Tensor with input dimensions if packed
 */
torch::Tensor n_hot(const torch::Tensor &p_self, int64_t num_classes, bool packed = false) {

  TORCH_CHECK(p_self.dtype() == torch::kLong,
              "ngme is only applicable to index tensor.");
  
  torch::Tensor self;
  if (packed) {
    self = unpack_batched_tensor(p_self);
  } else {
    self = p_self;
  }

  auto shape = self.sizes().vec();

  // First dimension is ngram
  shape.erase(shape.begin());

  // empty tensor could be converted to one hot representation,
  // but shape inference is not possible.
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      shape.push_back(num_classes);
      return torch::empty(shape, self.options());
    }
  }

  TORCH_CHECK(self.is_contiguous(), "Tensor has to be contiguous");

  // non-empty tensor
  if (self.device().type() != torch::kCUDA || self.device().type() != torch::kCUDA) {
    // for cuda, rely on device assert thrown by scatter
    TORCH_CHECK(self.min().item().toLong() >= 0,
                "Class values must be non-negative.");
  } else {
    if (self.device().type() != torch::kCUDA) {
      // rely on device asserts from scatter to avoid sync here
      TORCH_CHECK(num_classes > self.max().item().toLong(),
                  "Class values must be smaller than num_classes.");
    } else {
      // for cuda, assert that num_classes is at least 1
      TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
    }
  }

  shape.push_back(num_classes);

  torch::Tensor ret = torch::zeros(shape, self.options());
  
  for(int i = 0; i < self.size(0); ++i) {
    ret.scatter_(-1, self[i].unsqueeze(-1), 1);
  }

  return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("n_hot", &n_hot, "N-Gram Multihot Encoding");
  m.def("pack", &pack, "Pack list");
  m.def("pack_tensor", &pack_tensor, "Pack tensor");
  m.def("unpack", &unpack, "Unpack list");
  m.def("unpack_tensor", &unpack_tensor, "Unpack tensor");
  m.def("unpack_4_gram", &unpack_4_gram, "Unpack 4 grams");
  m.def("unpack_batched_tensor", &unpack_batched_tensor, "Unpack batched tensor");
}
