#include "ATen/ATen.h"

#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"
#include "ATen/core/Error.h"
#include "ATen/Config.h"

#include <tuple>

namespace at { namespace native {

static inline std::vector<int64_t> pooling_expand_param_if_needed(
  IntList list_param, const char *param_name, int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t) list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the pooling "
       << "dimensions, but got " << param_name << "=" << list_param;
     AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

static void check1d(
    const char* function_name,
    const char* argument_name,
    IntList x) {
  AT_CHECK(
      x.size() == 1,
      function_name, "() argument '", argument_name,
      "' should contain one int (got ", x.size(), ")");
}

Tensor adaptive_avg_pool1d(const Tensor & self, IntList output_size) {
  checkDim("adaptive_avg_pool1d", TensorArg(self, "self", 1), 3);
  check1d("adaptive_avg_pool1d", "output_size", output_size);

  auto output = at::adaptive_avg_pool2d(
      self.unsqueeze(2),
      {1, output_size[0]});

  return output.squeeze(2);
}

std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntList output_size) {
  checkDim("adaptive_max_pool1d", TensorArg(self, "self", 1), 3);
  check1d("adaptive_max_pool1d", "output_size", output_size);

  Tensor output, indices;
  std::tie(output, indices) = at::adaptive_max_pool2d(
      self.unsqueeze(2),
      {1, output_size[0]});

  return std::make_tuple(output.squeeze(2), indices.squeeze(2));
}

std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& self,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    IntList dilation,
    bool ceil_mode) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDim("max_pool1d", TensorArg(self, "self", 1), 3);
  check1d("max_pool1d", "kernel_size", kernel_size);
  check1d("max_pool1d", "stride", stride);
  check1d("max_pool1d", "padding", padding);
  check1d("max_pool1d", "dilation", dilation);

  Tensor output, indices;
  std::tie(output, indices) = at::max_pool2d_with_indices(
      self.unsqueeze(2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      {1, dilation[0]},
      ceil_mode);

  return std::make_tuple(output.squeeze(2), indices.squeeze(2));
}

Tensor avg_pool1d(
    const Tensor& self,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    bool ceil_mode,
    bool count_include_pad) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDim("avg_pool1d", TensorArg(self, "self", 1), 3);
  check1d("avg_pool1d", "kernel_size", kernel_size);
  check1d("avg_pool1d", "stride", stride);
  check1d("avg_pool1d", "padding", padding);

  auto output = at::thnn_avg_pool2d(
      self.unsqueeze(2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      ceil_mode,
      count_include_pad);

  return output.squeeze(2);
}

Tensor avg_pool2d(
    const Tensor& self,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    bool ceil_mode,
    bool count_include_pad) {
 bool use_mkldnn = false;
#if AT_MKLDNN_ENABLED()
  use_mkldnn = (self.type().backend() == at::Backend::CPU
               && self.type().scalarType() == at::kFloat // only on CPU Float Tensors
               && (self.ndimension() == 4)
               );
#endif
  if (use_mkldnn) {
#if AT_MKLDNN_ENABLED()
    auto k = self.ndimension();
    auto dim = k - 2;
    auto kernel_size_ = pooling_expand_param_if_needed(kernel_size, "kernel_size", dim);
    std::vector<int64_t> stride_(dim);
    if (stride.empty()) {
      stride_ = kernel_size_;
    } else {
      stride_ = pooling_expand_param_if_needed(stride, "stride", dim);
    }
    auto padding_ = pooling_expand_param_if_needed(padding, "padding", dim);
    auto output_and_indices = at::mkldnn_pooling(self.contiguous(), kernel_size_, stride_, padding_, ceil_mode, count_include_pad, true);
    return std::get<0>(output_and_indices);
#endif
  } else {
    return at::thnn_avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  }
}

Tensor avg_pool3d(
    const Tensor& self,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    bool ceil_mode,
    bool count_include_pad) {
 bool use_mkldnn = false;
#if AT_MKLDNN_ENABLED()
  use_mkldnn = (self.type().backend() == at::Backend::CPU
               && self.type().scalarType() == at::kFloat // only on CPU Float Tensors
               && (self.ndimension() == 5)
               );
#endif
  if (use_mkldnn) {
#if AT_MKLDNN_ENABLED()
    auto k = self.ndimension();
    auto dim = k - 2;
    auto kernel_size_ = pooling_expand_param_if_needed(kernel_size, "kernel_size", dim);
    std::vector<int64_t> stride_(dim);
    if (stride.empty()) {
      stride_ = kernel_size_;
    } else {
      stride_ = pooling_expand_param_if_needed(stride, "stride", dim);
    }
    auto padding_ = pooling_expand_param_if_needed(padding, "padding", dim);
    auto output_and_indices = at::mkldnn_pooling(self.contiguous(), kernel_size_, stride_, padding_, ceil_mode, count_include_pad, true);
    return std::get<0>(output_and_indices);
#endif
  } else {
    return at::thnn_avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  }
}

Tensor max_pool1d(
    const Tensor& self,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    IntList dilation,
    bool ceil_mode) {
  auto output_and_indices = at::max_pool1d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(output_and_indices);
}

Tensor max_pool2d(
    const Tensor& self,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    IntList dilation,
    bool ceil_mode) {
  bool is_dilated = false;
  for (auto d : dilation) {
    is_dilated |= (d != 1);
  }
 bool use_mkldnn = false;
#if AT_MKLDNN_ENABLED()
  use_mkldnn = (self.type().backend() == at::Backend::CPU
               && self.type().scalarType() == at::kFloat // only on CPU Float Tensors
               && (self.ndimension() == 4)
               && !is_dilated // not support dilation
               );
#endif
  if (use_mkldnn) {
#if AT_MKLDNN_ENABLED()
    auto k = self.ndimension();
    auto dim = k - 2;
    auto kernel_size_ = pooling_expand_param_if_needed(kernel_size, "kernel_size", dim);
    std::vector<int64_t> stride_(dim);
    if (stride.empty()) {
      stride_ = kernel_size_;
    } else {
      stride_ = pooling_expand_param_if_needed(stride, "stride", dim);
    }
    auto padding_ = pooling_expand_param_if_needed(padding, "padding", dim);
    auto output_and_indices = at::mkldnn_pooling(self.contiguous(), kernel_size_, stride_, padding_, ceil_mode, false, false);
    return std::get<0>(output_and_indices);
#endif
  } else {
    auto output_and_indices = at::max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode);
    return std::get<0>(output_and_indices);
  }
}

Tensor max_pool3d(
    const Tensor& self,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    IntList dilation,
    bool ceil_mode) {
 bool is_dilated = false;
 for (auto d : dilation) {
   is_dilated |= (d != 1);
 }
 bool use_mkldnn = false;
#if AT_MKLDNN_ENABLED()
  use_mkldnn = (self.type().backend() == at::Backend::CPU
               && self.type().scalarType() == at::kFloat // only on CPU Float Tensors
               && (self.ndimension() == 5)
               && !is_dilated // not support dilation
               );
#endif
  if (use_mkldnn) {
#if AT_MKLDNN_ENABLED()
    auto k = self.ndimension();
    auto dim = k - 2;
    auto kernel_size_ = pooling_expand_param_if_needed(kernel_size, "kernel_size", dim);
    std::vector<int64_t> stride_(dim);
    if (stride.empty()) {
      stride_ = kernel_size_;
    } else {
      stride_ = pooling_expand_param_if_needed(stride, "stride", dim);
    }
    auto padding_ = pooling_expand_param_if_needed(padding, "padding", dim);
    auto output_and_indices = at::mkldnn_pooling(self.contiguous(), kernel_size_, stride_, padding_, ceil_mode, false, false);
    return std::get<0>(output_and_indices);
#endif
  } else {
    auto output_and_indices = at::max_pool3d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode);
    return std::get<0>(output_and_indices);
  }
}
} // namespace native
} // namespace at
