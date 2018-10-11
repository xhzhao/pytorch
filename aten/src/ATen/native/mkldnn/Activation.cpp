#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

at::Tensor mkldnn_relu(const at::Tensor& input, double negative_slope) {
  AT_ERROR("mkldnn_relu: ATen not compiled with MKLDNN support");
}

at::Tensor & mkldnn_relu_(at::Tensor& input, double negative_slope) {
  AT_ERROR("mkldnn_relu_: ATen not compiled with MKLDNN support");
}
at::Tensor mkldnn_relu_backward(const at::Tensor& input, const at::Tensor& grad_output_t, double negative_slope) {
  AT_ERROR("mkldnn_relu_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>

using namespace mkldnn;

namespace at { namespace native {

at::Tensor mkldnn_relu(const at::Tensor& input, const double negative_slope = 0.0f ) {

  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto data_t = memory::data_type::f32;

  IntList input_size = input.sizes();
  auto dim = input_size.size();
  memory::dims input_tz(dim);
  auto format_input = (dim == 5) ? memory::format::ncdhw : memory::format::nchw;
  for (size_t i = 0; i < dim; i++)
    input_tz[i] = input_size[i];

  //create a output tensor with same size with input
  auto output = at::empty(input_size, input.options());
  auto input_md = memory::desc({input_tz}, data_t, format_input);
  auto output_md = memory::desc({input_tz}, data_t, format_input);

  std::shared_ptr<eltwise_forward::desc> eltwise_forward_desc;
  eltwise_forward_desc.reset(new eltwise_forward::desc(prop_kind::forward,
    algorithm::eltwise_relu, input_md, negative_slope));

  std::shared_ptr<eltwise_forward::primitive_desc> eltwise_forward_pd;
  eltwise_forward_pd.reset(new eltwise_forward::primitive_desc(*eltwise_forward_desc, cpu_engine));

  auto input_usr_memory = memory({input_md, cpu_engine}, input.data_ptr());
  auto output_usr_memory = memory({output_md, cpu_engine}, output.data_ptr());
  std::vector<primitive> net;

  auto output_pd = eltwise_forward_pd->dst_primitive_desc();
  auto output_memory = output_usr_memory;
  if (output_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd))
    output_memory = memory(output_pd);

  std::shared_ptr<eltwise_forward> elt_forward;
  elt_forward.reset(new eltwise_forward(*eltwise_forward_pd, input_usr_memory, output_memory));
  net.push_back(*elt_forward);

  if (output_memory != output_usr_memory)
    net.push_back(reorder(output_memory, output_usr_memory));

  Stream::Instance().get_stream().submit(net);

  return output;
}

// inplace
at::Tensor & mkldnn_relu_(at::Tensor& input, const double negative_slope = 0.0f ) {

  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto data_t = memory::data_type::f32;
  IntList input_size = input.sizes();
  auto dim = input_size.size();

  memory::dims input_tz(dim);
  auto format_input = (dim == 5) ? memory::format::ncdhw : memory::format::nchw;
  for (size_t i = 0; i < dim; i++)
     input_tz[i] = input_size[i];

  auto input_md = memory::desc({input_tz}, data_t, format_input);

  std::shared_ptr<eltwise_forward::desc> eltwise_forward_desc;
  eltwise_forward_desc.reset(new eltwise_forward::desc(prop_kind::forward,
    algorithm::eltwise_relu, input_md, negative_slope));

  std::shared_ptr<eltwise_forward::primitive_desc> eltwise_forward_pd;
  eltwise_forward_pd.reset(new eltwise_forward::primitive_desc(*eltwise_forward_desc, cpu_engine));

  auto input_usr_memory = memory({input_md, cpu_engine}, input.data_ptr());
  std::vector<primitive> net;

  auto output_pd = eltwise_forward_pd->dst_primitive_desc();
  auto output_memory = input_usr_memory;
  if (input_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd))
    output_memory = memory(output_pd);

  std::shared_ptr<eltwise_forward> elt_forward;
  elt_forward.reset(new eltwise_forward(*eltwise_forward_pd, input_usr_memory, output_memory));
  net.push_back(*elt_forward);

  if (output_memory != input_usr_memory)
    net.push_back(reorder(output_memory, input_usr_memory));

  Stream::Instance().get_stream().submit(net);
  return input;
}

at::Tensor mkldnn_relu_backward(const at::Tensor& input, const at::Tensor& grad_output_t, const double negative_slope = 0.0f ){

  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto data_t = memory::data_type::f32;

  IntList input_size = input.sizes();
  auto dim = input_size.size();
  Tensor grad_output = grad_output_t.contiguous();
  auto grad_input = at::empty(input_size, grad_output.options());

  memory::dims input_tz(dim);
  auto format_input = (dim == 5) ? memory::format::ncdhw : memory::format::nchw;
  for (size_t i = 0; i < dim; i++)
    input_tz[i] = input_size[i];

  // Backward relu
  auto diff_dst_md =  memory::desc({input_tz}, data_t, format_input);
  auto input_md =  memory::desc({input_tz}, data_t, format_input);

  // need to re-create relu_forward_pd to feed relu_backward_weight_pd
  std::shared_ptr<eltwise_forward::desc> eltwise_forward_desc;
  eltwise_forward_desc.reset(new eltwise_forward::desc(prop_kind::forward,
    algorithm::eltwise_relu, input_md, negative_slope));

  std::shared_ptr<eltwise_forward::primitive_desc> eltwise_forward_pd;
  eltwise_forward_pd.reset(new eltwise_forward::primitive_desc(*eltwise_forward_desc, cpu_engine));

  // create backward relu primitive_descriptor
  std::shared_ptr<eltwise_backward::desc> eltwise_backward_desc;
  eltwise_backward_desc.reset(new eltwise_backward::desc(algorithm::eltwise_relu, diff_dst_md, input_md, negative_slope));

  std::shared_ptr<eltwise_backward::primitive_desc> eltwise_backward_pd;
  eltwise_backward_pd.reset(new eltwise_backward::primitive_desc(*eltwise_backward_desc, cpu_engine, *eltwise_forward_pd));

  // create memory for relu diff src
  auto input_memory = memory({{{input_tz}, data_t, format_input}, cpu_engine}, input.data_ptr());
  auto diff_dst_memory  = memory({{{input_tz}, data_t, format_input}, cpu_engine}, grad_output.data_ptr());

  auto grad_input_usr_memory = memory({{{input_tz}, data_t, format_input}, cpu_engine}, grad_input.data_ptr());
  auto grad_input_pd = eltwise_backward_pd->diff_src_primitive_desc();
  auto grad_input_memory = grad_input_usr_memory;
  if (grad_input_memory.get_primitive_desc() != memory::primitive_desc(grad_input_pd))
    grad_input_memory = memory(grad_input_pd);

  std::vector<primitive> net;

  // finally create a backward relu primitive
  std::shared_ptr<eltwise_backward> elt_backward;

  elt_backward.reset(new eltwise_backward(*eltwise_backward_pd, input_memory, diff_dst_memory, grad_input_memory));
  net.push_back(*elt_backward);

  if (grad_input_memory != grad_input_usr_memory)
    net.push_back(reorder(grad_input_memory, grad_input_usr_memory));

  Stream::Instance().get_stream().submit(net);
  return  grad_input;
}

}} // namespace at::native
#endif


