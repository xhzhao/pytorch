#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  AT_ERROR("mkldnn_relu: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_relu_(Tensor& input) {
  AT_ERROR("mkldnn_relu_: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_relu_backward(const Tensor& input, const Tensor& grad_output_t) {
  AT_ERROR("mkldnn_relu_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>
#include <ATen/mkldnn/Memory.h>
#include <ATen/mkldnn/Utils.h>

using namespace mkldnn;

namespace at { namespace native {

namespace {

constexpr double negative_slope = 0.0;
constexpr int max_dim = 3;

struct ActivationParams {
  int64_t dim;
  int64_t input_size[2 + max_dim];
};

void setActivationParams(ActivationParams* params, const Tensor& input) {
  memset(params, 0, sizeof(ActivationParams));

  params->dim = input.dim();
  for (int64_t i = 0; i < params->dim; ++i) {
    params->input_size[i] = input.size(i);
  }
}

struct ActivationArgs {
  ActivationParams params;
  memory::dims input_tz;
  memory::format format_data;

  ActivationArgs(const Tensor& input) {
    setActivationParams(&params, input);

    for (int64_t i = 0; i < input.dim(); ++i) {
      input_tz.push_back(params.input_size[i]);
    }
    format_data = (input.dim() == 5) ? memory::format::ncdhw : memory::format::nchw;
  }

  memory::primitive_desc input_pd() { return _primitive_md(input_tz, format_data); }
  memory::primitive_desc output_pd() { return _primitive_md(input_tz, format_data); }
};

eltwise_forward::primitive_desc _elt_fwd_pd(const ActivationArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto elt_prop = prop_kind::forward;
  auto elt_algo = algorithm::eltwise_relu;
  auto input_md = _format_md(args.input_tz, args.format_data);

  auto _desc = eltwise_forward::desc(elt_prop, elt_algo, input_md, negative_slope);
  return eltwise_forward::primitive_desc(_desc, _engine);
}

eltwise_backward::primitive_desc _elt_bwd_pd(const ActivationArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto elt_algo = algorithm::eltwise_relu;
  auto input_md = _format_md(args.input_tz, args.format_data);
  auto output_md = _format_md(args.input_tz, args.format_data);

  auto _desc = eltwise_backward::desc(elt_algo, output_md, input_md, negative_slope);
  return eltwise_backward::primitive_desc(_desc, _engine, _elt_fwd_pd(args));
}

struct MKLDNNReluForward : MKLDNNPrimitive<eltwise_forward> {
  std::shared_ptr<memory> _input;
  std::shared_ptr<memory> _output;

  MKLDNNReluForward() : MKLDNNPrimitive<eltwise_forward>() {
    set_null_memory(_input);
    set_null_memory(_output);
  }

  void set(const eltwise_forward::primitive_desc& pd, const memory& input, const memory& output) {
    _input->set_data_handle(input.get_data_handle());
    _output->set_data_handle(output.get_data_handle());
    if (_prim == nullptr) {
      _prim.reset(new eltwise_forward(pd, *_input, *_output));
    }
  }
};

struct MKLDNNReluBackward : MKLDNNPrimitive<eltwise_backward> {
  std::shared_ptr<memory> _input;
  std::shared_ptr<memory> _grad_output;
  std::shared_ptr<memory> _grad_input;

  MKLDNNReluBackward() : MKLDNNPrimitive<eltwise_backward>() {
    set_null_memory(_input);
    set_null_memory(_grad_output);
    set_null_memory(_grad_input);
  }

  void set(const eltwise_backward::primitive_desc& pd, const memory& input,
      const memory& grad_output, const memory& grad_input) {

    _input->set_data_handle(input.get_data_handle());
    _grad_output->set_data_handle(grad_output.get_data_handle());
    _grad_input->set_data_handle(grad_input.get_data_handle());

    if (_prim == nullptr) {
      _prim.reset(new eltwise_backward(pd, *_input, *_grad_output, *_grad_input));
    }
  }
};

}  // namespace

Tensor mkldnn_relu(const Tensor& input) {
  auto output = at::empty_like(input);

  ActivationArgs args(input);
  auto _pd = _elt_fwd_pd(args);

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto output_usr = MKLDNNMemory(args.output_pd(), output);

  auto output_prv = output_usr.create(_pd.dst_primitive_desc());

  std::shared_ptr<MKLDNNReluForward> relu_fwd;
  static thread_local PrimitiveCache<ActivationParams, MKLDNNReluForward> cache;
  if (cache.find(args.params, relu_fwd)) {
    relu_fwd->set(_pd, input_usr._memory, output_prv);
  } else {
    relu_fwd.reset(new MKLDNNReluForward());
    relu_fwd->set(_pd, input_usr._memory, output_prv);
    cache.insert(args.params, relu_fwd);
  }

  MKLDNN_EXEC(relu_fwd->get_primitive());
  output_usr.reorder_from(output_prv);

  return output;
}

Tensor& mkldnn_relu_(Tensor& input) {
  ActivationArgs args(input);
  auto _pd = _elt_fwd_pd(args);

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto output_usr = input_usr;

  auto output_prv = output_usr.create(_pd.dst_primitive_desc());

  std::shared_ptr<MKLDNNReluForward> relu_fwd;
  static thread_local PrimitiveCache<ActivationParams, MKLDNNReluForward> cache;
  if (cache.find(args.params, relu_fwd)) {
    relu_fwd->set(_pd, input_usr._memory, output_prv);
  } else {
    relu_fwd.reset(new MKLDNNReluForward());
    relu_fwd->set(_pd, input_usr._memory, output_prv);
    cache.insert(args.params, relu_fwd);
  }
  MKLDNN_EXEC(relu_fwd->get_primitive());
  output_usr.reorder_from(output_prv);

  return input;
}

Tensor mkldnn_relu_backward(const Tensor& input, const Tensor& grad_output_t) {
  Tensor grad_output = grad_output_t.contiguous();
  auto grad_input= at::empty(input.sizes(), grad_output.options());

  ActivationArgs args(input);
  auto _pd = _elt_bwd_pd(args);

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto grad_output_usr = MKLDNNMemory(args.output_pd(), grad_output);
  auto grad_input_usr = MKLDNNMemory(args.input_pd(), grad_input);

  auto grad_input_prv = grad_input_usr.create(_pd.diff_src_primitive_desc());

  std::shared_ptr<MKLDNNReluBackward> relu_bwd;
  static thread_local PrimitiveCache<ActivationParams, MKLDNNReluBackward> cache;
  if (cache.find(args.params, relu_bwd)) {
    relu_bwd->set(_pd, input_usr._memory, grad_output_usr._memory, grad_input_prv);
  } else {
    relu_bwd.reset(new MKLDNNReluBackward());
    relu_bwd->set(_pd, input_usr._memory, grad_output_usr._memory, grad_input_prv);
    cache.insert(args.params, relu_bwd);
  }
  MKLDNN_EXEC(relu_bwd->get_primitive());
  grad_input_usr.reorder_from(grad_input_prv);

  return grad_input;
}

}}  // namespace at::native

#endif
