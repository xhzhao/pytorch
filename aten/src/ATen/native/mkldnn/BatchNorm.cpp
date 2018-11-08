#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& running_mean,
    const Tensor& running_var, bool training, double exponential_average_factor, double epsilon){
  AT_ERROR("mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var, double epsilon, bool training) {
  AT_ERROR("mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
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

struct BatchNormParams {
  int64_t dim;
  int64_t input_size[2 + max_dim];
  double epsilon;
  bool training;
  bool use_weight_bias;
  bool use_running_stat;
  unsigned flags;
};

void setBatchNormParams(BatchNormParams* params, const Tensor& input, double epsilon,
    bool training, bool use_weight_bias, bool use_running_stat, unsigned flags) {

  memset(params, 0, sizeof(BatchNormParams));

  params->dim = input.dim();
  for (int64_t i = 0; i < params->dim; ++i) {
    params->input_size[i] = input.size(i);
  }

  params->epsilon = epsilon;
  params->training = training;
  params->use_weight_bias= use_weight_bias;
  params->use_running_stat = use_running_stat;
  params->flags = flags;
}

struct BatchNormArgs {
  BatchNormParams params;
  memory::dims input_tz;
  memory::format format_data;

  BatchNormArgs(const Tensor& input, const Tensor& weight, const Tensor& bias,
       const Tensor& running_mean, const Tensor& running_var, bool training, double epsilon) {

    unsigned flags = 0;
    bool use_weight_bias = (weight.defined() && bias.defined());
    bool use_running_stat = (running_mean.defined() && running_var.defined());
    if (use_weight_bias) flags |= use_scale_shift;
    if (use_running_stat && (!training)) flags |= use_global_stats;

    setBatchNormParams(&params, input, epsilon, training, use_weight_bias, use_running_stat, flags);

    for (int64_t i = 0; i < params.dim; ++i) {
      input_tz.push_back(params.input_size[i]);
    }

    format_data = (params.dim == 5) ? memory::format::ncdhw : memory::format::nchw;
  }

  memory::primitive_desc input_pd() { return _primitive_md(input_tz, format_data); }
  memory::primitive_desc output_pd() { return _primitive_md(input_tz, format_data); }
};

batch_normalization_forward::primitive_desc _batchnorm_fwd_pd(const BatchNormArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto batchnorm_prop = args.params.training ? prop_kind::forward_training : prop_kind::forward_inference;

  auto input_md = _format_md(args.input_tz, args.format_data);
  auto bn_forward_desc = batch_normalization_forward::desc(batchnorm_prop,
      input_md, args.params.epsilon, args.params.flags);

  return batch_normalization_forward::primitive_desc(bn_forward_desc, _engine);
}

batch_normalization_backward::primitive_desc _batchnorm_bwd_pd(const BatchNormArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto input_md = _format_md(args.input_tz, args.format_data);
  auto batchnorm_prop = args.params.use_weight_bias ? prop_kind::backward : prop_kind::backward_data;

  auto batchnorm_bwd_desc = batch_normalization_backward::desc(batchnorm_prop,
      input_md, input_md, args.params.epsilon, args.params.flags);
  return batch_normalization_backward::primitive_desc(batchnorm_bwd_desc, _engine, _batchnorm_fwd_pd(args));
}

struct MKLDNNBatchNormForward : MKLDNNPrimitive<batch_normalization_forward> {
  std::shared_ptr<memory> _input;
  std::shared_ptr<memory> _output;
  std::shared_ptr<memory> _scaleshift_memory;
  std::shared_ptr<memory> _mean_memory;
  std::shared_ptr<memory> _variance_memory;

  MKLDNNBatchNormForward() : MKLDNNPrimitive<batch_normalization_forward>() {
    set_null_memory(_input);
    set_null_memory(_output);
    set_null_memory(_scaleshift_memory);
    set_null_memory(_mean_memory);
    set_null_memory(_variance_memory);
  }

  void set(const batch_normalization_forward::primitive_desc& pd, const memory& input,
      const std::shared_ptr<memory>& scaleshift_memory, const std::shared_ptr<memory>& mean_memory,
      const std::shared_ptr<memory>& variance_memory, const memory& output, bool training) {

    _input->set_data_handle(input.get_data_handle());
    _output->set_data_handle(output.get_data_handle());

    if (!training) {
      if (mean_memory != nullptr && variance_memory != nullptr) {
        _mean_memory->set_data_handle(mean_memory->get_data_handle());
        _variance_memory->set_data_handle(variance_memory->get_data_handle());
        if (scaleshift_memory != nullptr) {
          _scaleshift_memory->set_data_handle(scaleshift_memory->get_data_handle());
          if (_prim == nullptr) {
            _prim.reset(new batch_normalization_forward(pd, *_input, primitive::at(*_mean_memory),
                primitive::at(*_variance_memory), *_scaleshift_memory, *_output));
          }
        } else {
          if (_prim == nullptr) {
            _prim.reset(new batch_normalization_forward(pd, *_input, primitive::at(*_mean_memory),
                primitive::at(*_variance_memory), *_output));
          }
        }
      } else {
        if (scaleshift_memory != nullptr) {
          _scaleshift_memory->set_data_handle(scaleshift_memory->get_data_handle());
          if (_prim == nullptr) {
            _prim.reset(new batch_normalization_forward(pd, *_input, *_scaleshift_memory, *_output));
          }
        } else {
          if (_prim == nullptr) {
            _prim.reset(new batch_normalization_forward(pd, *_input, *_output));
          }
        }
      }
    } else {
      _mean_memory->set_data_handle(mean_memory->get_data_handle());
      _variance_memory->set_data_handle(variance_memory->get_data_handle());
      if (scaleshift_memory != nullptr) {
        _scaleshift_memory->set_data_handle(scaleshift_memory->get_data_handle());
        if (_prim == nullptr) {
          _prim.reset(new batch_normalization_forward(pd, *_input, *_scaleshift_memory, *_output,
              *_mean_memory, *_variance_memory));
        }
      } else {
        if (_prim == nullptr) {
          _prim.reset(new batch_normalization_forward(pd, *_input, *_output, *_mean_memory, *_variance_memory));
        }
      }
    }
  }
};

struct MKLDNNBatchNormBackward : MKLDNNPrimitive<batch_normalization_backward> {
  std::shared_ptr<memory> _input;
  std::shared_ptr<memory> _grad_output;
  std::shared_ptr<memory> _grad_input;
  std::shared_ptr<memory> _grad_scaleshift;
  std::shared_ptr<memory> _mean_memory;
  std::shared_ptr<memory> _variance_memory;
  std::shared_ptr<memory> _scaleshift;

  MKLDNNBatchNormBackward() : MKLDNNPrimitive<batch_normalization_backward>() {
    set_null_memory(_input);
    set_null_memory(_grad_output);
    set_null_memory(_grad_input);
    set_null_memory(_grad_scaleshift);
    set_null_memory(_mean_memory);
    set_null_memory(_variance_memory);
    set_null_memory(_scaleshift);
  }

  void set(const batch_normalization_backward::primitive_desc& pd, const memory& input,
      const memory& scaleshift_memory, const memory& mean_memory, const memory& variance_memory,
      const memory& grad_input, const memory& grad_output, const std::shared_ptr<memory>& grad_scaleshift) {

    _input->set_data_handle(input.get_data_handle());
    _grad_output->set_data_handle(grad_output.get_data_handle());
    _grad_input->set_data_handle(grad_input.get_data_handle());
    _mean_memory->set_data_handle(mean_memory.get_data_handle());
    _variance_memory->set_data_handle(variance_memory.get_data_handle());
    _scaleshift->set_data_handle(scaleshift_memory.get_data_handle());

     if (grad_scaleshift != nullptr ) {
       _grad_scaleshift->set_data_handle(grad_scaleshift->get_data_handle());
       if (_prim == nullptr) {
         _prim.reset(new batch_normalization_backward(pd, *_input, primitive::at(*_mean_memory),
             primitive::at(*_variance_memory), *_grad_output, *_scaleshift, *_grad_input, *_grad_scaleshift));
       }
     } else {
       if (_prim == nullptr) {
         _prim.reset(new batch_normalization_backward(pd, *_input, primitive::at(*_mean_memory),
             primitive::at(*_variance_memory), *_grad_output, *_grad_input));
       }
     }
  }
};

}  // namespace

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(const Tensor& input,
    const Tensor& weight, const Tensor& bias, const Tensor& running_mean,
    const Tensor& running_var, bool training, double exponential_average_factor, double epsilon) {

  auto output = at::empty_like(input);

  BatchNormArgs args(input, weight, bias, running_mean, running_var, training, epsilon);

  auto input_size = args.params.input_size;
  int32_t ic = input_size[1];
  auto save_mean = at::empty({ic}, input.options());
  auto save_var = at::empty({ic}, input.options());

  auto _pd = _batchnorm_fwd_pd(args);

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto output_usr = MKLDNNMemory(args.output_pd(), output);
  auto output_prv = output_usr.create(_pd.dst_primitive_desc());

  std::shared_ptr<memory> scaleshift_memory;
  if (args.params.use_weight_bias) {
    scaleshift_memory.reset(new memory(_pd.weights_primitive_desc()));
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory->get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = ((float*)weight.data_ptr())[i];
      scaleshift_buf[ic + i] = ((float*)bias.data_ptr())[i];
    }
  }

  std::shared_ptr<memory> mean_memory, variance_memory;
  if (!args.params.training) {
    if (args.params.use_running_stat) {
      mean_memory.reset(new memory(_pd.mean_primitive_desc(), running_mean.data_ptr()));
      variance_memory.reset(new memory(_pd.variance_primitive_desc(), running_var.data_ptr()));
    }
  } else {
    mean_memory.reset(new memory(_pd.mean_primitive_desc(), save_mean.data_ptr()));
    variance_memory.reset(new memory(_pd.variance_primitive_desc(), save_var.data_ptr()));
  }

  std::shared_ptr<MKLDNNBatchNormForward> batchnorm_fwd;
  static thread_local PrimitiveCache<BatchNormParams, MKLDNNBatchNormForward> cache;
  if (cache.find(args.params, batchnorm_fwd)) {
    batchnorm_fwd->set(_pd, input_usr._memory, scaleshift_memory, mean_memory,
        variance_memory, output_prv, args.params.training);
  } else {
    batchnorm_fwd.reset(new MKLDNNBatchNormForward());
    batchnorm_fwd->set(_pd, input_usr._memory, scaleshift_memory, mean_memory,
        variance_memory, output_prv, args.params.training);
    cache.insert(args.params, batchnorm_fwd);
  }

  MKLDNN_EXEC(batchnorm_fwd->get_primitive());
  output_usr.reorder_from(output_prv);

  if (args.params.training && args.params.use_running_stat) {
    float len = (float)(input_size[0] * input_size[2] * input_size[3]);
    len =(args.params.dim == 5) ? (len * input_size[4]) : len;

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

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& input,
    const Tensor& grad_output, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean,
    const Tensor& save_var, double epsilon, bool training) {

  auto grad_input = at::empty_like(input);
  auto grad_weight = at::empty({}, input.options());
  auto grad_bias = at::empty({}, input.options());

  BatchNormArgs args(input, weight, bias, running_mean, running_var, training, epsilon);

  auto _pd = _batchnorm_bwd_pd(args);

  auto input_size = args.params.input_size;
  int32_t ic = input_size[1];

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto grad_input_usr = MKLDNNMemory(args.input_pd(), grad_input);
  auto grad_output_usr = MKLDNNMemory(args.output_pd(), grad_output);

  auto mean_memory = MKLDNNMemory(_pd.mean_primitive_desc(), save_mean);
  auto variance_memory = MKLDNNMemory(_pd.variance_primitive_desc(), save_var);

  auto scaleshift_memory = memory(_pd.weights_primitive_desc());

  std::shared_ptr<memory> grad_scaleshift;
  if (args.params.use_weight_bias) {
    grad_scaleshift.reset(new memory(_pd.diff_weights_primitive_desc()));
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory.get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = ((float*)weight.data_ptr())[i];
      scaleshift_buf[ic + i] = ((float*)bias.data_ptr())[i];
    }
  } else {
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory.get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = 0.0;
      scaleshift_buf[ic + i] = 1.0;
    }
  }

  std::shared_ptr<MKLDNNBatchNormBackward> batchnorm_bwd;
  static thread_local PrimitiveCache<BatchNormParams, MKLDNNBatchNormBackward> cache;
  if (cache.find(args.params, batchnorm_bwd)) {
    batchnorm_bwd->set(_pd, input_usr._memory, scaleshift_memory, mean_memory._memory,
        variance_memory._memory, grad_input_usr._memory, grad_output_usr._memory, grad_scaleshift);
  } else {
    batchnorm_bwd.reset(new MKLDNNBatchNormBackward());
    batchnorm_bwd->set(_pd, input_usr._memory, scaleshift_memory, mean_memory._memory,
        variance_memory._memory, grad_input_usr._memory, grad_output_usr._memory, grad_scaleshift);
    cache.insert(args.params, batchnorm_bwd);
  }

  MKLDNN_EXEC(batchnorm_bwd->get_primitive());

  if (args.params.use_weight_bias) {
    grad_weight.resize_(weight.sizes());
    grad_bias.resize_(weight.sizes());
    float* grad_scaleshift_buf = reinterpret_cast<float *>(grad_scaleshift->get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
     ((float*)grad_weight.data_ptr())[i] = grad_scaleshift_buf[i];
     ((float*)grad_bias.data_ptr())[i] = grad_scaleshift_buf[ic + i];
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native
#endif
