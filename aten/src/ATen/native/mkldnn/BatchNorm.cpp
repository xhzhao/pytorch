#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool training, double exponential_average_factor, double epsilon){
  AT_ERROR("mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var, double epsilon) {
  AT_ERROR("mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>

using namespace mkldnn;

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool training, double exponential_average_factor, double epsilon)
{
  unsigned flags = 0;
  auto propagation = training ? prop_kind::forward_training : prop_kind::forward_inference;
  bool use_weight_bias_ = (weight.defined() && bias.defined());
  bool use_running_stat = (running_mean.defined() && running_var.defined());
  if (use_weight_bias_) flags |= use_scale_shift;
  if (use_running_stat && (!training)) flags |= use_global_stats;

  IntList input_size = input.sizes();
  auto dim = input_size.size();
  memory::dims input_tz(dim);
  auto format_input = (dim == 5) ? memory::format::ncdhw : memory::format::nchw;
  for (size_t i = 0; i < dim; i++) {
    input_tz[i] = input_size[i];
  }
  int32_t ic = input_size[1];

  auto cpu_engine = CpuEngine::Instance().get_engine();
  memory::data_type data_t = memory::data_type::f32;
  auto output = at::empty(input_size, input.options());
  auto input_md = memory::desc({input_tz}, data_t, format_input);

  auto input_usr_memory = memory({input_md, cpu_engine}, input.data_ptr());
  auto output_usr_memory = memory({input_md, cpu_engine}, output.data_ptr());

  auto bn_forward_desc = batch_normalization_forward::desc(propagation, input_md, epsilon, flags);
  auto bn_forward_pd = batch_normalization_forward::primitive_desc(bn_forward_desc, cpu_engine);

  memory scaleshift_memory = null_memory(cpu_engine);
  if (use_weight_bias_) {
    scaleshift_memory = memory(bn_forward_pd.weights_primitive_desc());
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory.get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = ((float*)weight.data_ptr())[i];
      scaleshift_buf[ic + i] = ((float*)bias.data_ptr())[i];
    }
  }

  auto output_pd = bn_forward_pd.dst_primitive_desc();
  auto output_memory = output_usr_memory;
  if (output_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd)) {
    output_memory = memory(output_pd);
  }

  std::shared_ptr<batch_normalization_forward> bn_forward;
  std::shared_ptr<memory> mean_memory, variance_memory;
  Tensor save_mean = at::empty({ic}, input.options());
  Tensor save_var= at::empty({ic}, input.options());
  if (!training) {
    if (use_running_stat) {
      mean_memory.reset(new memory(bn_forward_pd.mean_primitive_desc(), running_mean.data_ptr()));
      variance_memory.reset(new memory(bn_forward_pd.variance_primitive_desc(), running_var.data_ptr()));
      if (use_weight_bias_) {
        bn_forward.reset(new batch_normalization_forward(bn_forward_pd, input_usr_memory,
          mkldnn::primitive::at(*mean_memory), mkldnn::primitive::at(*variance_memory),
          scaleshift_memory, output_memory));
      } else {
        bn_forward.reset(new batch_normalization_forward(bn_forward_pd, input_usr_memory,
          mkldnn::primitive::at(*mean_memory), mkldnn::primitive::at(*variance_memory), output_memory));
      }
    } else {
      if (use_weight_bias_) {
        bn_forward.reset(new batch_normalization_forward(bn_forward_pd, input_usr_memory,
          scaleshift_memory, output_memory));
      } else {
        bn_forward.reset(new batch_normalization_forward(bn_forward_pd, input_usr_memory, output_memory));
      }
    }
  } else {
    mean_memory.reset(new memory(bn_forward_pd.mean_primitive_desc(), save_mean.data_ptr()));
    variance_memory.reset(new memory(bn_forward_pd.variance_primitive_desc(), save_var.data_ptr()));
    if (use_weight_bias_) {
      bn_forward.reset(new batch_normalization_forward(bn_forward_pd, input_usr_memory,
        scaleshift_memory, output_memory, *mean_memory, *variance_memory));
    } else {
      bn_forward.reset(new batch_normalization_forward(bn_forward_pd, input_usr_memory,
        output_memory, *mean_memory, *variance_memory));
    }
  }
  std::vector<primitive> net;
  net.push_back(*bn_forward);

  if (output_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd)) {
    net.push_back(reorder(output_memory, output_usr_memory));
  }

  Stream::Instance().get_stream().submit(net);
  if (training && use_running_stat) {
    float len = (float)(input_size[0] * input_size[2] * input_size[3]);
    len =(dim == 5) ? (len * input_size[4]) : len;

    float* mean_buf = reinterpret_cast<float *>(mean_memory->get_data_handle());
    float* var_buf = reinterpret_cast<float *>(variance_memory->get_data_handle());
    float* running_mean_buf = reinterpret_cast<float *>(running_mean.data_ptr());
    float* running_var_buf = reinterpret_cast<float *>(running_var.data_ptr());
    const float reborn = 1.0f - exponential_average_factor;
    const float adjust = exponential_average_factor * len / (len - 1);
    for (int32_t i=0; i<ic; ++i) {
      running_mean_buf[i] = running_mean_buf[i] * reborn + mean_buf[i] * exponential_average_factor;
      running_var_buf[i]  = running_var_buf[i] * reborn + var_buf[i] * adjust;
    }
  }
  return std::tuple<Tensor, Tensor, Tensor>{output, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var, double epsilon)
{
  unsigned flags = 0;
  bool use_weight_bias_ = (weight.defined() && bias.defined());
  bool use_running_stat = (running_mean.defined() && running_var.defined());
  if (use_weight_bias_) flags |= use_scale_shift;

  IntList input_size = input.sizes();
  auto dim = input_size.size();
  memory::dims input_tz(dim);
  auto format_input = (dim == 5) ? memory::format::ncdhw : memory::format::nchw;
  for (size_t i = 0; i < dim; i++) {
    input_tz[i] = input_size[i];
  }
  int32_t ic = input_size[1];

  auto cpu_engine = CpuEngine::Instance().get_engine();
  memory::data_type data_t = memory::data_type::f32;

  auto grad_input = at::empty(input.sizes(), input.options());
  auto grad_weight = at::empty({}, input.options());
  auto grad_bias = at::empty({}, input.options());

  auto input_md = memory::desc({input_tz}, data_t, format_input);

  auto input_usr_memory = memory({input_md, cpu_engine}, input.data_ptr());
  auto grad_output_usr_memory = memory({input_md, cpu_engine}, grad_output.data_ptr());
  auto grad_input_usr_memory =  memory({input_md, cpu_engine}, grad_input.data_ptr());

  auto bn_forward_desc = batch_normalization_forward::desc(prop_kind::forward_training, input_md, epsilon, flags);
  auto bn_forward_pd = batch_normalization_forward::primitive_desc(bn_forward_desc, cpu_engine);

  std::shared_ptr<batch_normalization_backward::desc> bn_backward_desc;
  if (use_weight_bias_) {
    bn_backward_desc.reset(new batch_normalization_backward::desc(prop_kind::backward, input_md, input_md, epsilon, flags));
  } else {
    bn_backward_desc.reset(new batch_normalization_backward::desc(prop_kind::backward_data, input_md, input_md, epsilon, flags));
  }
  auto bn_backward_pd = batch_normalization_backward::primitive_desc(*bn_backward_desc, cpu_engine, bn_forward_pd);
  std::vector<primitive> net;
  auto grad_output_memory = grad_output_usr_memory;
  auto grad_input_memory = grad_input_usr_memory;

  auto grad_scaleshift_memory = memory(bn_backward_pd.diff_weights_primitive_desc());

  auto scaleshift_memory = memory(bn_forward_pd.weights_primitive_desc());
  auto mean_memory = memory(bn_backward_pd.mean_primitive_desc(), save_mean.data_ptr());
  auto variance_memory = memory(bn_backward_pd.variance_primitive_desc(), save_var.data_ptr());
  std::shared_ptr<batch_normalization_backward> bn_backward;
  if (use_weight_bias_) {
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory.get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = ((float*)weight.data_ptr())[i];
      scaleshift_buf[ic + i] = ((float*)bias.data_ptr())[i];
    }
    bn_backward.reset(new batch_normalization_backward(bn_backward_pd, input_usr_memory,
      mkldnn::primitive::at(mean_memory), mkldnn::primitive::at(variance_memory),
      grad_output_memory, scaleshift_memory, grad_input_memory, grad_scaleshift_memory));
  } else {
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory.get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = 0.0;
      scaleshift_buf[ic + i] = 1.0;
    }
    bn_backward.reset(new batch_normalization_backward(bn_backward_pd, input_usr_memory,
      mkldnn::primitive::at(mean_memory), mkldnn::primitive::at(variance_memory),
      grad_output_memory, grad_input_memory));
  }
  net.push_back(*bn_backward);
  Stream::Instance().get_stream().submit(net);
  if (use_weight_bias_) {
    grad_weight.resize_(weight.sizes());
    grad_bias.resize_(weight.sizes());
    float* grad_scaleshift_buf = reinterpret_cast<float *>(grad_scaleshift_memory.get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
     ((float*)grad_weight.data_ptr())[i] = grad_scaleshift_buf[i];
     ((float*)grad_bias.data_ptr())[i] = grad_scaleshift_buf[ic + i];
    }
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native
#endif
