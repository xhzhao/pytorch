#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<at::Tensor, at::Tensor> mkldnn_pooling(const Tensor& input, IntList kernel_size,
    IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, bool avg) {
  AT_ERROR("mkldnn_pooling: ATen not compiled with MKLDNN support");
}

at::Tensor mkldnn_pooling_backward(const Tensor& input, const Tensor& grad_output_t, const Tensor& indice,
    IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, bool avg) {
  AT_ERROR("mkldnn_pooling_backward: ATen not compiled with MKLDNN support");
}
}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>

using namespace mkldnn;

namespace at { namespace native {

std::vector<int64_t> pooling_output_size(IntList input_size, IntList kernel_size, IntList stride,
     IntList padding, bool ceil_mode, bool& force_exclude_padding_flag_) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = input_size[1];
  for (size_t d = 2; d < dim; ++d) {
    if (ceil_mode) {
      output_size[d] = std::ceil((float)(input_size[d] + (2 * padding[d - 2])
        - kernel_size[d-2]) / (float)stride[d - 2]) + 1;
    } else {
      output_size[d] = std::floor((float)(input_size[d] + (2 * padding[d - 2])
        - kernel_size[d-2]) / (float)stride[d - 2] + 1);
    }
  }

  if (dim == 5) {
     if (padding[0] || padding[1] || padding[2] || kernel_size[0] == 1 || kernel_size[1] == 1 || kernel_size[2] == 1) {
       if ((output_size[2] - 1) * stride[0] >= input_size[2] + padding[0]) {
         --output_size[2];
       }
       if ((output_size[3] - 1) * stride[1] >= input_size[3] +padding[1]) {
         --output_size[3];
       }
       if ((output_size[4] - 1) * stride[2] >= input_size[4] + padding[2]) {
         --output_size[4];
       }
     } else {
       force_exclude_padding_flag_ = true;
     }
  } else {
    if (padding[0] || padding[1] || kernel_size[0] == 1 || kernel_size[1] == 1) {
      if ((output_size[2] - 1) * stride[0] >= input_size[2] + padding[0]) {
        --output_size[2];
      }
      if ((output_size[3] - 1) * stride[1] >= input_size[3] + padding[1]) {
        --output_size[3];
      }
    } else {
      force_exclude_padding_flag_ = true;
    }
  }
  return output_size;
}

std::vector<int64_t> pooling_pad_size(IntList input_size, IntList output_size, IntList kernel_size, IntList stride, IntList padding) {
  auto dim = input_size.size();
  std::vector<int64_t> mkldnn_pad((dim-2)*2);
  mkldnn_pad[0] = padding[0];
  mkldnn_pad[1] = padding[0];
  mkldnn_pad[2] = padding[1];
  mkldnn_pad[3] = padding[1];

  auto h = input_size[2] + mkldnn_pad[0];
  while (h + mkldnn_pad[1] < stride[0] * (output_size[2] - 1) + kernel_size[0]) mkldnn_pad[1]++;

  auto w = input_size[3] + mkldnn_pad[2];
  while (w + mkldnn_pad[3] < stride[1] * (output_size[3] - 1) + kernel_size[1]) mkldnn_pad[3]++;

  if (dim == 5) {
    mkldnn_pad[4] = padding[2];
    mkldnn_pad[5] = padding[2];
    auto d = input_size[4] + mkldnn_pad[4];
    while (d + mkldnn_pad[5] < stride[2] * (output_size[4] - 1) + kernel_size[2]) mkldnn_pad[5]++;
  }
  return mkldnn_pad;
}

std::tuple<at::Tensor, at::Tensor> mkldnn_pooling(const Tensor& input, IntList kernel_size,
    IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, bool avg) {
  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto data_t = memory::data_type::f32;
  bool force_exclude_padding_flag_ = false;
  IntList input_size = input.sizes();
  auto dim = input_size.size();
  auto format_input = memory::format::nchw;
  auto output_size = pooling_output_size(input_size, kernel_size, stride, padding, ceil_mode, force_exclude_padding_flag_);
  auto mkldnn_pad = pooling_pad_size(input_size, output_size, kernel_size, stride, padding);

  auto output = at::empty(output_size, input.options());
  auto option = input.options().dtype(at::kByte);
  if (kernel_size[0] >= 256 || kernel_size[1] >= 256)
    option = input.options().dtype(at::kInt);

  memory::dims input_tz(dim);
  memory::dims output_tz(dim);
  memory::dims kernel_tz(dim-2);
  memory::dims stride_tz(dim-2);
  memory::dims padding1_tz(dim-2);
  memory::dims padding2_tz(dim-2);

  input_tz[0] = input_size[0];
  input_tz[1] = input_size[1];
  input_tz[2] = input_size[2];
  input_tz[3] = input_size[3];
  output_tz[0] = output_size[0];
  output_tz[1] = output_size[1];
  output_tz[2] = output_size[2];
  output_tz[3] = output_size[3];

  kernel_tz[0] = kernel_size[0];
  kernel_tz[1] = kernel_size[1];
  stride_tz[0] = stride[0];
  stride_tz[1] = stride[1];

  padding1_tz[0] = mkldnn_pad[0];
  padding1_tz[1] = mkldnn_pad[2];
  padding2_tz[0] = mkldnn_pad[1];
  padding2_tz[1] = mkldnn_pad[3];

  if (dim == 5) {
    format_input = memory::format::ncdhw;
    if (kernel_size[2] >= 256) option = input.options().dtype(at::kInt);
    input_tz[4] = input_size[4];
    output_tz[4] = output_size[4];
    kernel_tz[2] = kernel_size[2];
    stride_tz[2] = stride[2];
    padding1_tz[2] = mkldnn_pad[4];
    padding2_tz[2] = mkldnn_pad[5];
  }

  auto indice = at::empty(output_size, option);

  algorithm pooling_algorithm = algorithm::pooling_max;
  if (avg) {
    if (count_include_pad) {
       pooling_algorithm = algorithm::pooling_avg_include_padding;
    } else {
      pooling_algorithm = algorithm::pooling_avg_exclude_padding;
    }
    if (force_exclude_padding_flag_ == true) {
      pooling_algorithm = algorithm::pooling_avg_exclude_padding;
    }
  }

  auto input_md = memory::desc({ input_tz }, data_t, format_input);
  auto output_md = memory::desc({ output_tz }, data_t, format_input);
  auto pool_fwd_desc = pooling_forward::desc(prop_kind::forward, pooling_algorithm,
    input_md, output_md, stride_tz, kernel_tz, padding1_tz, padding2_tz, padding_kind::zero);
  auto pool_fwd_pd = pooling_forward::primitive_desc(pool_fwd_desc, cpu_engine);

  auto input_memory = memory({input_md, cpu_engine}, input.data_ptr());
  auto output_usr_memory = memory({output_md, cpu_engine}, output.data_ptr());

  std::vector<primitive> net;
  auto output_pd = pool_fwd_pd.dst_primitive_desc();
  auto output_memory = output_usr_memory;
  if (output_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd)) {
    output_memory = memory(output_pd);
  }
  std::shared_ptr<pooling_forward> pool_fwd;
  std::shared_ptr<memory> pool_workspace_memory;
  if (avg) {
    pool_fwd.reset(new pooling_forward(pool_fwd_pd, input_memory, output_memory));
  } else {
    pool_workspace_memory.reset(new memory(pool_fwd_pd.workspace_primitive_desc(), indice.data_ptr()));
    pool_fwd.reset(new pooling_forward(pool_fwd_pd, input_memory, output_memory, *pool_workspace_memory));
  }
  net.push_back(*pool_fwd);

  if (output_memory != output_usr_memory) {
    net.push_back(reorder(output_memory, output_usr_memory));
  }
  Stream::Instance().get_stream().submit(net);
  return std::make_tuple(output, indice);
}

at::Tensor mkldnn_pooling_backward(const Tensor& input, const Tensor& grad_output_t, const Tensor& indice,
    IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, bool avg) {
  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto data_t = memory::data_type::f32;
  IntList input_size = input.sizes();
  auto dim = input_size.size();
  auto format_input = memory::format::nchw;
  Tensor grad_output = grad_output_t.contiguous();
  auto grad_input = at::empty(input_size, grad_output.options());
  auto output_size = grad_output_t.sizes();
  auto mkldnn_pad = pooling_pad_size(input_size, output_size, kernel_size, stride, padding);

  memory::dims input_tz(dim);
  memory::dims output_tz(dim);
  memory::dims kernel_tz(dim-2);
  memory::dims stride_tz(dim-2);
  memory::dims padding1_tz(dim-2);
  memory::dims padding2_tz(dim-2);

  input_tz[0] = input_size[0];
  input_tz[1] = input_size[1];
  input_tz[2] = input_size[2];
  input_tz[3] = input_size[3];
  output_tz[0] = output_size[0];
  output_tz[1] = output_size[1];
  output_tz[2] = output_size[2];
  output_tz[3] = output_size[3];

  kernel_tz[0] = kernel_size[0];
  kernel_tz[1] = kernel_size[1];
  stride_tz[0] = stride[0];
  stride_tz[1] = stride[1];
  padding1_tz[0] = mkldnn_pad[0];
  padding1_tz[1] = mkldnn_pad[2];
  padding2_tz[0] = mkldnn_pad[1];
  padding2_tz[1] = mkldnn_pad[3];

  bool force_exclude_padding_flag_ = true;
  if (padding[0] || padding[1] || kernel_size[0] == 1 || kernel_size[1] == 1) {
    force_exclude_padding_flag_ = false;
  }

  if (dim == 5) {
    format_input = memory::format::ncdhw;
    input_tz[4] = input_size[4];
    output_tz[4] = output_size[4];
    kernel_tz[2] = kernel_size[2];
    stride_tz[2] = stride[2];
    padding1_tz[2] = mkldnn_pad[4];
    padding2_tz[2] = mkldnn_pad[5];
    if (padding[0] || padding[1] || padding[2] || kernel_size[0] == 1 || kernel_size[1] == 1 || kernel_size[2] == 1) {
      force_exclude_padding_flag_ = false;
    }
  }

  algorithm pooling_algorithm = algorithm::pooling_max;
  if (avg) {
    if (count_include_pad) {
       pooling_algorithm = algorithm::pooling_avg_include_padding;
    } else {
      pooling_algorithm = algorithm::pooling_avg_exclude_padding;
    }
    if (force_exclude_padding_flag_ == true) {
      pooling_algorithm = algorithm::pooling_avg_exclude_padding;
    }
  }

  auto pool_src_md = memory::desc({ input_tz }, data_t, format_input);
  auto pool_dst_md = memory::desc({ output_tz }, data_t, format_input);

  auto pool_user_src_memory = memory({ pool_src_md, cpu_engine });
  auto pool_user_dst_memory = memory({ pool_dst_md, cpu_engine });

  auto pool_fwd_desc = pooling_forward::desc(prop_kind::forward, pooling_algorithm,
    pool_src_md, pool_dst_md, stride_tz, kernel_tz, padding1_tz, padding2_tz, padding_kind::zero);
  auto pool_fwd_pd = pooling_forward::primitive_desc(pool_fwd_desc, cpu_engine);

  auto pool_diff_dst_memory = memory({{{ output_tz }, data_t, format_input}, cpu_engine}, grad_output.data_ptr());

  auto pool_bwd_desc = pooling_backward::desc(pooling_algorithm, pool_src_md, pool_dst_md,
    stride_tz, kernel_tz, padding1_tz, padding2_tz, padding_kind::zero);

  auto pool_bwd_pd = pooling_backward::primitive_desc(pool_bwd_desc, cpu_engine, pool_fwd_pd);

  auto grad_input_usr_memory = memory(pool_bwd_pd.diff_src_primitive_desc(),grad_input.data_ptr());

  auto grad_input_pd = pool_bwd_pd.diff_src_primitive_desc();
  auto pool_diff_src_memory = grad_input_usr_memory;
  if (pool_diff_src_memory.get_primitive_desc() != memory::primitive_desc(grad_input_pd))
    pool_diff_src_memory = memory(grad_input_pd);
  std::vector<primitive> net;
  std::shared_ptr<pooling_backward> pool_bwd ;
  std::shared_ptr<memory> pool_workspace_memory;
  if (avg) {
    pool_bwd.reset(new pooling_backward(pool_bwd_pd, pool_diff_dst_memory, pool_diff_src_memory));
  } else {
    pool_workspace_memory.reset(new memory(pool_fwd_pd.workspace_primitive_desc(), indice.data_ptr()));
    pool_bwd.reset(new pooling_backward(pool_bwd_pd, pool_diff_dst_memory, *pool_workspace_memory, pool_diff_src_memory));
  }
  net.push_back(*pool_bwd);

  if (pool_diff_src_memory!= grad_input_usr_memory)
    net.push_back(reorder(pool_diff_src_memory, grad_input_usr_memory));
  Stream::Instance().get_stream().submit(net);
  return grad_input;
}

}} // namespace at::native
#endif
