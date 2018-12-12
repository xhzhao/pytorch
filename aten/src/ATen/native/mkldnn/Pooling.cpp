#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor> mkldnn_pooling(const Tensor& input, IntList kernel_size,
    IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, bool avg) {
  AT_ERROR("mkldnn_pooling: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_pooling_backward(const Tensor& input, const Tensor& grad_output_t,
    const Tensor& indice, IntList kernel_size, IntList stride, IntList padding,
    bool ceil_mode, bool count_include_pad, bool avg) {
  AT_ERROR("mkldnn_pooling_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>
#include <ATen/mkldnn/Memory.h>
#include <ATen/mkldnn/Utils.h>

using namespace mkldnn;

namespace at { namespace native {

namespace {

constexpr int max_dim = 3;

static std::vector<int64_t> pooling_output_size(IntList input_size, IntList kernel_size,
    IntList stride, IntList padding, bool ceil_mode, bool& force_exclude_padding_flag) {

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
       force_exclude_padding_flag = true;
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
      force_exclude_padding_flag = true;
    }
  }

  return output_size;
}

static std::vector<int64_t> pooling_pad_size(IntList input_size, IntList output_size,
    IntList kernel_size, IntList stride, IntList padding) {

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

struct PoolingParams {
  int64_t dim;
  int64_t input_size[2 + max_dim];
  int64_t output_size[2 + max_dim];
  int64_t kernel[max_dim];
  int64_t stride[max_dim];
  int64_t paddingl[max_dim];
  int64_t paddingr[max_dim];
  bool ceil_mode;
  bool count_include_pad;
  bool force_exclude_padding_flag;
  bool avg;
};

void setPoolingParams(PoolingParams* params, const Tensor& input,
    const Tensor& output, IntList kernel, IntList stride, std::vector<int64_t> padding,
     bool ceil_mode, bool count_include_pad, bool force_exclude_padding_flag, bool avg) {

  memset(params, 0, sizeof(PoolingParams));

  params->dim = input.dim();
  for (int64_t i = 0; i < params->dim; ++i) {
    params->input_size[i] = input.size(i);
    params->output_size[i] = output.size(i);
  }
  for (size_t i = 0; i < kernel.size(); ++i) {
    params->kernel[i] = kernel[i];
    params->stride[i] = stride[i];
    params->paddingl[i] = padding[2*i];
    params->paddingr[i] = padding[2*i+1];
  }
  params->ceil_mode = ceil_mode;
  params->count_include_pad = count_include_pad;
  params->force_exclude_padding_flag = force_exclude_padding_flag;
  params->avg = avg;
}

struct PoolingArgs {
  PoolingParams params;
  memory::dims input_tz;
  memory::dims output_tz;
  memory::dims _kernel;
  memory::dims _stride;
  memory::dims _paddingl;
  memory::dims _paddingr;
  memory::format format_data;

  PoolingArgs(const Tensor& input, const Tensor& output, IntList kernel, IntList stride, std::vector<int64_t> padding,
      bool ceil_mode, bool count_include_pad, bool force_exclude_padding_flag, bool avg) {

    setPoolingParams(&params, input, output, kernel, stride,
      padding, ceil_mode, count_include_pad, force_exclude_padding_flag, avg);

    for (int64_t i = 0; i < input.dim(); ++i) {
      input_tz.push_back(params.input_size[i]);
      output_tz.push_back(params.output_size[i]);
    }

    for (size_t k = 0; k < kernel.size(); ++k) {
      _kernel.push_back(params.kernel[k]);
      _stride.push_back(params.stride[k]);
      _paddingl.push_back(params.paddingl[k]);
      _paddingr.push_back(params.paddingr[k]);
    }

    format_data = (params.dim == 5) ? memory::format::ncdhw : memory::format::nchw;

  }

  memory::primitive_desc input_pd() { return _primitive_md(input_tz, format_data); }
  memory::primitive_desc output_pd() { return _primitive_md(output_tz, format_data);}
};

pooling_forward::primitive_desc _pooling_fwd_pd(const PoolingArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto pooling_prop = prop_kind::forward;
  auto pooling_algo = algorithm::pooling_max;

  if (args.params.avg) {
    if (args.params.count_include_pad) {
      pooling_algo = algorithm::pooling_avg_include_padding;
    } else {
      pooling_algo = algorithm::pooling_avg_exclude_padding;
    }
    if (args.params.force_exclude_padding_flag) {
      pooling_algo = algorithm::pooling_avg_exclude_padding;
    }
  }

  auto input_md = _format_md(args.input_tz, args.format_data);
  auto output_md = _format_md(args.output_tz, args.format_data);

  auto pooling_fwd_desc = pooling_forward::desc(pooling_prop, pooling_algo, input_md, output_md,
    args._stride, args._kernel, args._paddingl, args._paddingr, padding_kind::zero);

  return pooling_forward::primitive_desc(pooling_fwd_desc, _engine);
}

pooling_backward::primitive_desc _pooling_bwd_pd(const PoolingArgs& args, const pooling_forward::primitive_desc& _fwd_pd) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto pooling_algo = algorithm::pooling_max;

  if (args.params.avg) {
    if (args.params.count_include_pad) {
      pooling_algo = algorithm::pooling_avg_include_padding;
    } else {
      pooling_algo = algorithm::pooling_avg_exclude_padding;
    }
    if (args.params.force_exclude_padding_flag) {
      pooling_algo = algorithm::pooling_avg_exclude_padding;
    }
  }

  auto input_md = _format_md(args.input_tz, args.format_data);
  auto output_md = _format_md(args.output_tz, args.format_data);

  auto pooling_bwd_desc = pooling_backward::desc(pooling_algo, input_md, output_md,
    args._stride, args._kernel, args._paddingl, args._paddingr, padding_kind::zero);

  return pooling_backward::primitive_desc(pooling_bwd_desc, _engine, _fwd_pd);
}

struct MKLDNNPoolingForward : MKLDNNPrimitive<pooling_forward> {
  std::shared_ptr<memory> _input;
  std::shared_ptr<memory> _output;
  std::shared_ptr<memory> _workspace;

  MKLDNNPoolingForward() : MKLDNNPrimitive<pooling_forward>() {
    set_null_memory(_input);
    set_null_memory(_output);
    set_null_memory(_workspace);
  }

  void set(const pooling_forward::primitive_desc& pd, const memory& input, const memory& output,
      const std::shared_ptr<memory>& workspace) {

    _input->set_data_handle(input.get_data_handle());
    _output->set_data_handle(output.get_data_handle());

    if (workspace != nullptr) {
      _workspace->set_data_handle(workspace->get_data_handle());
      if (_prim == nullptr) {
        _prim.reset(new pooling_forward(pd, *_input, *_output, *_workspace));
      }
    } else {
      if (_prim == nullptr) {
        _prim.reset(new pooling_forward(pd, *_input, *_output));
      }
    }
  }
};

struct MKLDNNPoolingBackward : MKLDNNPrimitive<pooling_backward> {
  std::shared_ptr<memory> _grad_input;
  std::shared_ptr<memory> _grad_output;
  std::shared_ptr<memory> _workspace;

  MKLDNNPoolingBackward() : MKLDNNPrimitive<pooling_backward>() {
    set_null_memory(_grad_input);
    set_null_memory(_grad_output);
    set_null_memory(_workspace);
  }

  void set(const pooling_backward::primitive_desc& pd, const memory& grad_input,
       const memory& grad_output, const std::shared_ptr<memory>& workspace) {

    _grad_input->set_data_handle(grad_input.get_data_handle());
    _grad_output->set_data_handle(grad_output.get_data_handle());

    if (workspace != nullptr) {
      _workspace->set_data_handle(workspace->get_data_handle());
      if (_prim == nullptr) {
        _prim.reset(new pooling_backward(pd, *_grad_output, *_workspace, *_grad_input));
      }
    } else {
      if (_prim == nullptr) {
        _prim.reset(new pooling_backward(pd, *_grad_output, *_grad_input));
      }
    }
  }
};

}  // namespace

std::tuple<Tensor, Tensor> mkldnn_pooling(const Tensor& input, IntList kernel_size,
    IntList stride, IntList padding, bool ceil_mode, bool count_include_pad, bool avg) {

  bool force_exclude_padding_flag = false;
  IntList input_size = input.sizes();
  auto output_size = pooling_output_size(input_size, kernel_size, stride, padding,
      ceil_mode, force_exclude_padding_flag);
  auto output = at::empty(output_size, input.options());
  auto indice = at::empty(output_size, input.options().dtype(at::kInt));

  auto mkldnn_pad = pooling_pad_size(input_size, output_size, kernel_size, stride, padding);

  PoolingArgs args(input, output, kernel_size, stride, mkldnn_pad, ceil_mode,
      count_include_pad, force_exclude_padding_flag, avg);
  auto _pd = _pooling_fwd_pd(args);

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto output_usr = MKLDNNMemory(args.output_pd(), output);

  auto output_prv = output_usr.create(_pd.dst_primitive_desc());

  std::shared_ptr<memory> workspace;
  if (!avg) {
    workspace.reset(new memory(_pd.workspace_primitive_desc(), indice.data_ptr()));
  }

  std::shared_ptr<MKLDNNPoolingForward> pooling_fwd;
  static thread_local PrimitiveCache<PoolingParams, MKLDNNPoolingForward> cache;
  if (cache.find(args.params, pooling_fwd)) {
    pooling_fwd->set(_pd, input_usr._memory, output_prv, workspace);
  } else {
    pooling_fwd.reset(new MKLDNNPoolingForward());
    pooling_fwd->set(_pd, input_usr._memory, output_prv, workspace);
    cache.insert(args.params, pooling_fwd);
  }

  MKLDNN_EXEC(pooling_fwd->get_primitive());

  output_usr.reorder_from(output_prv);

  return std::make_tuple(output, indice);
}

Tensor mkldnn_pooling_backward(const Tensor& input, const Tensor& grad_output_t,
    const Tensor& indice, IntList kernel_size, IntList stride, IntList padding,
    bool ceil_mode, bool count_include_pad, bool avg) {

  IntList input_size = input.sizes();
  auto dim = input_size.size();
  Tensor grad_output = grad_output_t.contiguous();
  auto grad_input = at::empty(input_size, grad_output.options());
  auto output_size = grad_output_t.sizes();
  auto mkldnn_pad = pooling_pad_size(input_size, output_size, kernel_size, stride, padding);

  bool force_exclude_padding_flag = true;

  if (dim == 5) {
    if (padding[0] || padding[1] || padding[2] || kernel_size[0] == 1 || kernel_size[1] == 1 || kernel_size[2] == 1) {
      force_exclude_padding_flag = false;
    }
  } else {
    if (padding[0] || padding[1] || kernel_size[0] == 1 || kernel_size[1] == 1) {
      force_exclude_padding_flag = false;
    }
  }

  PoolingArgs args(input, grad_output, kernel_size, stride, mkldnn_pad, ceil_mode,
      count_include_pad, force_exclude_padding_flag, avg);

  auto _fwd_pd = _pooling_fwd_pd(args);
  auto _bwd_pd = _pooling_bwd_pd(args, _fwd_pd);

  auto grad_input_usr = MKLDNNMemory(args.input_pd(), grad_input);
  auto grad_output_usr = MKLDNNMemory(args.output_pd(), grad_output);

  auto grad_input_prv = grad_input_usr.create(_bwd_pd.diff_src_primitive_desc());

  std::shared_ptr<memory> workspace;
  if (!avg) {
    workspace.reset(new memory(_fwd_pd.workspace_primitive_desc(), indice.data_ptr()));
  }

  std::shared_ptr<MKLDNNPoolingBackward> pooling_bwd;
  static thread_local PrimitiveCache<PoolingParams, MKLDNNPoolingBackward> cache;
  if (cache.find(args.params, pooling_bwd)) {
    pooling_bwd->set(_bwd_pd, grad_input_prv, grad_output_usr._memory, workspace);
  } else {
    pooling_bwd.reset(new MKLDNNPoolingBackward());
    pooling_bwd->set(_bwd_pd, grad_input_prv, grad_output_usr._memory, workspace);
    cache.insert(args.params, pooling_bwd);
  }
  MKLDNN_EXEC(pooling_bwd->get_primitive());

  grad_input_usr.reorder_from(grad_input_prv);

  return grad_input;
}

}}  // namespace at::native

#endif
