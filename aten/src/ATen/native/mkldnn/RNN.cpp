#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn(
    const Tensor& input, const Tensor& batch_sizes, TensorList weight, const Tensor& hx,const Tensor& cx,
    int64_t celltype) {
  throw std::runtime_error("mkldnn_rnn: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn_backward(
    const Tensor& input, const Tensor& batch_sizes, TensorList weight, const Tensor& hx, const Tensor& cx,
    const Tensor& y, const Tensor& hy, const Tensor& cy, const Tensor& grad_y, const Tensor& grad_hy, const Tensor& grad_cy,
    const Tensor& workspace, int64_t celltype) {
  throw std::runtime_error("mkldnn_rnn_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_ENABLED()

#include <ATen/mkldnn/Runtime.h>

#define MKLDNN_RNN_TANH  0
#define MKLDNN_RNN_RELU  1
#define MKLDNN_GRU       2
#define MKLDNN_LSTM      3

using namespace mkldnn;

namespace at { namespace native {

constexpr int64_t ldgoi_shuffle_dim = 0;
constexpr int64_t ldgo_shffule_dim = 0;

namespace {

  memory::desc _format_md(const memory::dims _dims, const memory::format _format) {
    return memory::desc({_dims}, memory::data_type::f32, _format);
  }

  memory::desc _generic_md(const memory::dims _dims) {
    return _format_md(_dims, memory::format::any);
  }

  memory _reorder(const memory& usr_memory, const memory::primitive_desc& pd) {
    if (usr_memory.get_primitive_desc() != memory::primitive_desc(pd)) {
      auto _memory = memory(pd);
      std::vector<memory::primitive> net;
      net.push_back(reorder(usr_memory, _memory));
      Stream::Instance().get_stream().submit(net);
      return _memory;
    }
    return usr_memory;
  }

  memory _reorder_prepare(const memory& usr_memory, const memory::primitive_desc& pd) {
    if (usr_memory.get_primitive_desc() != memory::primitive_desc(pd)) {
      auto _memory = memory(pd);
      return _memory;
    }
    return usr_memory;
  }
  void memory_zeros(memory& t, int numel) {
    auto memory_pd = t.get_primitive_desc();
    //size_t numel = memory_pd.get_size();
    float * dataptr = (float*)t.get_data_handle();
    for(size_t i = 0; i < numel; i++) {
      dataptr[i] = 0;
    }
    return;
  }


  // NB: MKLDNN has several special requirements for RNN weight primitive
  // a) weight needs to be in ldgio format
  // b) for LSTM, mkldnn gate order is (forget, input, output, cell), different
  // from pytorch native (input, forget, cell, output)
  Tensor _shuffle_gates(const Tensor& weight, int64_t num_gates, bool forward=true) {
    return weight.contiguous();
  }

} // anonymous namespace


void print_tensor(memory t, std::string name) {
#if 0
  auto memory_pd = t.get_primitive_desc();
  size_t numel = memory_pd.get_size();
  auto ndims = memory_pd.desc().data.ndims;
  const auto &dims = memory_pd.desc().data.dims;
  std::cout << "Tensor name = "<< name<<", size = "<<numel << ", ndims = "<< ndims<<", shape = [";
  for(size_t i = 0; i < ndims; i++) {
    std::cout<<dims[i]<<",";
  }
  std::cout <<"] , data = ";
  float * dataptr = (float*)t.get_data_handle();
  for(size_t i = 0; i < numel; i++) {
    std::cout << dataptr[i]<<", ";
  }
  std::cout<<std::endl;
#endif
}

void print_tensor(Tensor t, std::string name) {
#if 1
  if(t.defined()){
    std::cout << "Tensor name = "<< name<<", size = "<<t.numel()<<" , data = ";

    float* dataptr = (float*)t.data_ptr();
    for(size_t i = 0; i < t.numel(); i++) {
      std::cout << dataptr[i]<<", ";
    }
    std::cout<<std::endl;
    std::cout << "  sizes = " <<t.sizes()<<std::endl;
    std::cout << "  strides = " <<t.strides()<<std::endl;
  }
#endif
}


std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn_call(
    const Tensor& input, const Tensor& batch_sizes, std::vector<Tensor> weight, const Tensor& hx, const Tensor& cx,
    int64_t celltype, bool has_biases, int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  //std::cout<<"mkldnn_rnn_lstm call start, celltype = "<< celltype<< std::endl;
  Tensor hidden_in, hidden_out, hy, cy;
  if (celltype == MKLDNN_LSTM) {
    hidden_in = at::cat({hx, cx}, 1);
    hidden_out = at::empty_like(hidden_in);
    std::vector<Tensor> hidden_arr = hidden_out.chunk(2, 1);
    hy = hidden_arr[0];
    cy = hidden_arr[1];
  } else {
    hidden_in = hx;
    hidden_out = at::empty_like(hx);
    hy = hidden_out;
    cy = at::empty({0}, hx.options()); // NB: Not allowed to return undefined tensors
  }

  auto workspace = at::empty({0}, input.options());
  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto null_memory_ = null_memory(cpu_engine);

  int32_t time_step = input.size(0);
  int32_t batch_size = input.size(1);
  int32_t input_size = input.size(2);
  int32_t hidden_size = hx.size(2);
  Tensor output = at::empty({time_step, batch_size, hidden_size});
  std::cout<<"forward, celltype = "<< celltype<<", L = "<<num_layers<<", T = "<<time_step<<", N = "<<batch_size<<", I = "<<input_size<<", H = "<<hidden_size<<std::endl;
  std::cout<<"weight.size() = "<<weight.size()<<std::endl;

  int32_t num_directions = bidirectional ? 2 : 1;
  auto format_tnc = memory::format::tnc;
  auto format_ldgoi = memory::format::ldgoi;
  auto format_ldgo = memory::format::ldgo;
  auto format_ldsnc = memory::format::ldsnc;

  auto rnn_prop = train ? prop_kind::forward_training : prop_kind::forward_inference;
  algorithm rnn_algo;
  algorithm rnn_activation = algorithm::eltwise_tanh;
  int32_t num_gates = 4;
  int32_t num_states = 2;
  int32_t additional_bias = 0;
  if (celltype == MKLDNN_RNN_TANH ) {
    rnn_algo = algorithm::vanilla_rnn;
    num_gates = 1;
    num_states = 1;
  } else if(celltype == MKLDNN_LSTM){
    rnn_algo = algorithm::vanilla_lstm;
    num_gates = 4;
    num_states = 2;
  } else if(celltype == MKLDNN_GRU){
    rnn_algo = algorithm::gru_linear_before_reset;
    num_gates = 3;
    num_states = 1;
    additional_bias = 1;
  }
  AT_ASSERTM(celltype >= MKLDNN_RNN_TANH && celltype <= MKLDNN_LSTM , "celltype invalid");

  auto rnn_dir = rnn_direction::unidirectional_left2right;
  //print_tensor(weight[0], "weight[0]");
  auto weight_ih = weight[0]; 
  auto weight_hh = weight[1]; 
  Tensor bias;
  if (has_biases) {
    if (celltype == MKLDNN_GRU){
      bias= at::empty({num_layers * num_directions, num_gates + additional_bias, hidden_size});
      //print_tensor(weight[2], "weight[2]");
      //print_tensor(weight[3], "weight[3]");
      for (int l = 0; l < num_layers * num_directions; l++) {
        auto bias_l = bias[l];
        auto weight2_l = weight[2][l];
        auto weight3_l = weight[3][l];
        auto bias_wx = weight2_l.chunk(3, 0); 
        auto bias_wh = weight3_l.chunk(3, 0); 
        bias_l[0] = bias_wx[0] + bias_wh[0];
        bias_l[1] = bias_wx[1] + bias_wh[1];
        bias_l[2] = bias_wx[2];
        bias_l[3] = bias_wh[2];
      }
      //print_tensor(bias, "bias");
    } else {
      bias = weight[2] + weight[3];
    }
  } else if(weight.size() == 2) {
    // fill zeros for non bias case
    bias = at::zeros({num_layers, num_directions, num_gates + additional_bias, hidden_size});
  }
  //print_tensor(input, "input "); 
  //print_tensor(hx, "hx "); 
  //print_tensor(weight_ih, "weight_ih");
  //print_tensor(weight_hh, "weight_hh");
  //print_tensor(bias, "bias");
  auto input_size_wx = weight_ih.size(2);
  auto hidden_size_wx = weight_hh.size(1) / num_gates;
  auto hidden_size_wh = weight_ih.size(1) / num_gates;
  AT_ASSERTM(input_size_wx == input_size, "input size mismatch");
  AT_ASSERTM(hidden_size_wx == hidden_size, "hidden size mismatch");
  AT_ASSERTM(hidden_size_wh == hidden_size, "hidden size mismatch");


try {
  auto rnn_cell_ = rnn_cell::desc(rnn_algo, rnn_activation);
  memory::dims input_tz = {time_step, batch_size, input_size};
  memory::dims weight_ih_tz = {num_layers, num_directions, input_size, num_gates, hidden_size};
  memory::dims weight_hh_tz = {num_layers, num_directions, hidden_size, num_gates, hidden_size};
  memory::dims bias_tz = {num_layers, num_directions, num_gates + additional_bias, hidden_size};
  memory::dims hidden_tz = {num_layers, num_directions, num_states, batch_size, hidden_size};
  memory::dims output_tz = {time_step, batch_size, hidden_size};

  auto input_md = _format_md(input_tz, format_tnc);
  auto hidden_md = _generic_md(hidden_tz);
  auto weight_ih_md = _generic_md(weight_ih_tz);
  auto weight_hh_md = _generic_md(weight_hh_tz);
  auto bias_md = _generic_md(bias_tz);
  auto output_md = _format_md(output_tz, format_tnc);



  auto rnn_forward_desc = rnn_forward::desc(rnn_prop, rnn_cell_, rnn_dir,
    input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md);

  auto rnn_forward_pd = rnn_forward::primitive_desc(rnn_forward_desc, cpu_engine);

  auto input_usr_mem = memory({input_md, cpu_engine}, input.data_ptr());
  auto hidden_in_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_in.data_ptr());
  auto weight_ih_usr_mem = memory({_format_md(weight_ih_tz, format_ldgoi), cpu_engine}, weight_ih.data_ptr());
  auto weight_hh_usr_mem = memory({_format_md(weight_hh_tz, format_ldgoi), cpu_engine}, weight_hh.data_ptr());
  auto bias_usr_mem = memory({_format_md(bias_tz, format_ldgo), cpu_engine}, bias.data_ptr());
  auto output_usr_mem = memory({output_md, cpu_engine}, output.data_ptr());
  auto hidden_out_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_out.data_ptr());

  auto workspace_mem = null_memory_;
  if (train) {
    auto workspace_pd = rnn_forward_pd.workspace_primitive_desc();
    auto workspace_size = workspace_pd.get_size();
    workspace.resize_(workspace_size);
    workspace_mem = memory(workspace_pd, workspace.data_ptr());
  }

  auto input_mem = _reorder(input_usr_mem, rnn_forward_pd.src_layer_primitive_desc());
  auto hidden_in_mem = _reorder(hidden_in_usr_mem, rnn_forward_pd.src_iter_primitive_desc());
  auto weight_ih_mem = _reorder(weight_ih_usr_mem, rnn_forward_pd.weights_layer_primitive_desc());
  auto weight_hh_mem = _reorder(weight_hh_usr_mem, rnn_forward_pd.weights_iter_primitive_desc());
  auto bias_mem = _reorder(bias_usr_mem, rnn_forward_pd.bias_primitive_desc());
  auto output_mem = _reorder(output_usr_mem, rnn_forward_pd.dst_layer_primitive_desc());
  auto hidden_out_mem = _reorder(hidden_out_usr_mem, rnn_forward_pd.dst_iter_primitive_desc());

  //print_tensor(weight_ih_usr_mem, "weight_ih_usr_mem");
  //print_tensor(weight_ih_mem, "weight_ih_mem");


  //print_tensor(weight_hh_usr_mem, "weight_hh_usr_mem");
  //print_tensor(weight_hh_mem, "weight_hh_mem");

  std::vector<primitive> net;
  net.push_back(rnn_forward(rnn_forward_pd, input_mem, hidden_in_mem, weight_ih_mem, weight_hh_mem,
    bias_mem, output_mem, hidden_out_mem, workspace_mem));

  if (hidden_out_mem != hidden_out_usr_mem) {
    net.push_back(reorder(hidden_out_mem, hidden_out_usr_mem));
  }

  Stream::Instance().get_stream().submit(net);

  //print_tensor(input, "input");
  //print_tensor(hx, "hx");
  //print_tensor(output, "output");
  //print_tensor(hy, "hy");

  //print_tensor(input_mem,"input_mem");
  //print_tensor(hidden_in_mem, "hidden_in_mem");
  //print_tensor(weight_ih_usr_mem, "weight_ih_usr_mem");
  //print_tensor(weight_ih_mem, "weight_ih_mem");
  //print_tensor(weight_hh_mem, "weight_hh_mem");
  //print_tensor(output_mem, "output_mem");
  //print_tensor(hidden_out_mem, "hidden_out_mem");

  } catch (error &e) {
    std::cerr << "message: " << e.message << std::endl;
  }
 
  return std::make_tuple(output, hy, cy, workspace);
}

void weigth_fit_mkldnn(std::vector<Tensor>& weight_dst, TensorList weight, int64_t celltype, bool has_biases, int64_t num_layers, bool bidirectional, int64_t input_size, int64_t hidden_size) {
  AT_ASSERTM(hidden_size == input_size, "could not flatten wight if hidden_size != input_size");
  int32_t num_gates = 4;
  if (celltype == MKLDNN_RNN_TANH ) {
    num_gates = 1;
  } else if(celltype == MKLDNN_LSTM){
    num_gates = 4;
  } else if(celltype == MKLDNN_GRU){
    num_gates = 3;
  }
  int num_directions = bidirectional ? 2 : 1;
  int bias_factor = has_biases ? 4 : 2;
  int weight_len = num_directions * num_layers * bias_factor;
  std::cout << "weight.size() = "<< weight.size()<<", weight_len = "<<weight_len<<std::endl;
  AT_ASSERTM(weight.size() == weight_len, "weight.size() mismatch with weight_len");

  Tensor weight_ih = at::empty({num_layers * num_directions, num_gates * hidden_size, hidden_size});
  Tensor weight_hh = at::empty({num_layers * num_directions, num_gates * hidden_size, hidden_size});
  Tensor bias_ih = at::empty({num_layers * num_directions, num_gates * hidden_size}); 
  Tensor bias_hh = at::empty({num_layers * num_directions, num_gates * hidden_size});
  int index = 0;
  for (int i = 0; i < weight_len; ) {
    std::cout <<"index = "<<index<<", i = "<<i<<std::endl;
    //print_tensor(weight_ih[index], "weight_ih[index]");
    //print_tensor(weight[i] , "weight[i]");
    weight_ih[index].copy_(weight[i++]);
    weight_hh[index].copy_(weight[i++]);
    if (has_biases) {
      bias_ih[index].copy_(weight[i++]);
      bias_hh[index].copy_(weight[i++]);
    }
    index++;
  } 
  weight_dst.emplace_back(weight_ih);
  weight_dst.emplace_back(weight_hh);
  if (has_biases) {
    weight_dst.emplace_back(bias_ih);
    weight_dst.emplace_back(bias_hh);
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn(
    const Tensor& input, const Tensor& batch_sizes, TensorList weight, const Tensor& hx, const Tensor& cx,
    int64_t celltype, bool has_biases, int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {

  int32_t input_size = input.size(2);
  int32_t hidden_size = hx.size(2);

  if ((input_size != hidden_size) || (bidirectional && (num_layers > 1))) {
    //call mkldnn rnn api layer by layer if layer > 1 and direction > 1
  } else {
    // flatten weight
    std::vector<Tensor> weight_dst;
    weigth_fit_mkldnn(weight_dst, weight, celltype, has_biases, num_layers, bidirectional, input_size, hidden_size);
    //call mkldnn rnn api directly
    auto result = mkldnn_rnn_call(input, batch_sizes, weight_dst, hx, cx, celltype, has_biases, num_layers, dropout_p, train, bidirectional, batch_first);
    return result;
  }

}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn_backward(
    const Tensor& input, const Tensor& batch_sizes, TensorList weight, const Tensor& hx, const Tensor& cx,
    const Tensor& y, const Tensor& hy, const Tensor& cy, const Tensor& grad_y, const Tensor& grad_hy, const Tensor& grad_cy,
    const Tensor& workspace, int64_t celltype, bool has_biases, int64_t num_layers, double dropout_p, bool train,
    bool bidirectional, bool batch_first) {
  //std::cout << "hy: " << hy.defined() << std::endl;
  //std::cout << "cy: " << cy.defined() << std::endl;
  //std::cout << "hx: " << hx.defined() << std::endl;
  //std::cout << "cx: " << cx.defined() << std::endl;
  //std::cout << "grad_y: " << grad_y.defined() << std::endl;
  //std::cout << "grad_hy: " << grad_hy.defined() << std::endl;
  //std::cout << "grad_cy: " << grad_cy.defined() << std::endl;

  //auto train = true;

  //std::cout<<"mkldnn_rnn_lstm_backward , celltype = "<<celltype<<std::endl;
  AT_CHECK(train, "mkldnn_rnn backward can only be called in training mode");

  // TODO: cache hidden_in, hidden_out from forward?
  // NB: MKLDNN requires to concat hx and cx for lstm
  Tensor hidden_in, hidden_out, grad_hidden_out, grad_hidden_in, grad_hx, grad_cx;
  if (celltype == MKLDNN_LSTM) {
    hidden_in = at::cat({hx, cx}, 0);
    hidden_out = at::cat({hy, cy}, 0);
    if(grad_hy.defined() && grad_cy.defined()) {
      grad_hidden_out = at::cat({grad_hy, grad_cy}, 0);
    } else if(grad_hy.defined()) {
      auto grad_cy_zeros = at::zeros_like(cy);
      grad_hidden_out = at::cat({grad_hy, grad_cy_zeros}, 0);
    } else if(grad_cy.defined()) {
      auto grad_hy_zeros = at::zeros_like(hy);
      grad_hidden_out = at::cat({grad_hy_zeros, grad_cy}, 0);
    } else {
      auto grad_hy_zeros = at::zeros_like(hy);
      auto grad_cy_zeros = at::zeros_like(cy);
      grad_hidden_out = at::cat({grad_hy_zeros, grad_cy_zeros}, 0);
    }
    grad_hidden_in = at::empty_like(grad_hidden_out);
    std::vector<Tensor> hidden_arr = grad_hidden_in.chunk(2, 0);
    grad_hx = hidden_arr[0];
    grad_cx = hidden_arr[1];
  } else {
    hidden_in = hx;
    hidden_out = hy;
    if(grad_hy.defined()) {
      grad_hidden_out = grad_hy;
    } else {
      grad_hidden_out = at::zeros_like(hy);
    }
    grad_hidden_in = at::empty_like(grad_hidden_out);
    grad_hx = grad_hidden_in;
    grad_cx = at::empty({0}, hx.options()); // NB: Not allowed to return undefined tensors
  }


  // TODO: check if we need to clone this?
  auto output = y;
  auto grad_output = grad_y;
  if(!grad_y.defined()) {
    grad_output = at::zeros_like(y);
  }
  //print_tensor(grad_y, "grad_y");
  //print_tensor(grad_hy, "grad_hy");
  //print_tensor(grad_output, "grad_y");
  //print_tensor(grad_hidden_out, "grad_hy");

  auto grad_input = at::empty_like(input);

  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto null_memory_ = null_memory(cpu_engine);

  //int32_t num_layers = 1;
  int32_t num_directions = bidirectional ? 2 : 1;

  int32_t time_step = input.size(0);
  int32_t batch_size = input.size(1);
  int32_t input_size = input.size(2);
  int32_t hidden_size = hx.size(2);
  
  //std::cout<<"backward, T = "<<time_step<<", N = "<<batch_size<<", I = "<<input_size<<", H = "<<hidden_size<<std::endl;

  auto format_tnc = memory::format::tnc;
  auto format_ldgoi = memory::format::ldgoi;
  auto format_ldgo = memory::format::ldgo;
  auto format_ldsnc = memory::format::ldsnc;

  auto rnn_prop = prop_kind::backward;
  auto rnn_dir = rnn_direction::unidirectional_left2right;

  algorithm rnn_algo;
  algorithm rnn_activation = algorithm::eltwise_tanh;
  int32_t num_gates = 4;
  int32_t num_states = 2;
  int32_t additional_bias = 0;
  if (celltype == MKLDNN_RNN_TANH ) {
    rnn_algo = algorithm::vanilla_rnn;
    num_gates = 1;
    num_states = 1;
  } else if(celltype == MKLDNN_LSTM){
    rnn_algo = algorithm::vanilla_lstm;
    num_gates = 4;
    num_states = 2;
  } else if(celltype == MKLDNN_GRU){
    rnn_algo = algorithm::gru_linear_before_reset;
    num_gates = 3;
    num_states = 1;
    additional_bias = 1;
  } 


  auto weight_ih = weight[0];
  auto weight_hh = weight[1];
  Tensor bias = at::zeros({(num_gates + additional_bias) * hidden_size}, weight[0].options());
  auto grad_weight_ih = at::zeros_like(weight_ih);
  auto grad_weight_hh = at::zeros_like(weight_hh);
  auto grad_bias = at::zeros_like(bias);

  memory::dims input_tz = {time_step, batch_size, input_size};
  memory::dims weight_ih_tz = {num_layers, num_directions, input_size, num_gates, hidden_size};
  memory::dims weight_hh_tz = {num_layers, num_directions, hidden_size, num_gates, hidden_size};
  memory::dims bias_tz = {num_layers, num_directions, num_gates + additional_bias, hidden_size};
  memory::dims hidden_tz = {num_layers, num_directions, num_states, batch_size, hidden_size};
  memory::dims output_tz = {time_step, batch_size, hidden_size};

  auto input_md = _format_md(input_tz, format_tnc);
  auto hidden_md = _generic_md(hidden_tz);
  auto weight_ih_md = _generic_md(weight_ih_tz);
  auto weight_hh_md = _generic_md(weight_hh_tz);
  auto bias_md = _generic_md(bias_tz);
  auto output_md = _format_md(output_tz, format_tnc);

  auto rnn_cell_ = rnn_cell::desc(rnn_algo, rnn_activation);
try{
  auto rnn_forward_desc = rnn_forward::desc(rnn_prop, rnn_cell_, rnn_dir,
    input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md);
  auto rnn_forward_pd = rnn_forward::primitive_desc(rnn_forward_desc, cpu_engine);

  auto rnn_backward_desc = rnn_backward::desc(rnn_prop, rnn_cell_, rnn_dir,
    input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md,
    input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md);

  auto rnn_backward_pd = rnn_backward::primitive_desc(rnn_backward_desc, cpu_engine, rnn_forward_pd);

  auto input_usr_mem = memory({input_md, cpu_engine}, input.data_ptr());
  auto hidden_in_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_in.data_ptr());
  auto weight_ih_usr_mem = memory({_format_md(weight_ih_tz, format_ldgoi), cpu_engine}, weight_ih.data_ptr());
  auto weight_hh_usr_mem = memory({_format_md(weight_hh_tz, format_ldgoi), cpu_engine}, weight_hh.data_ptr());
  auto bias_usr_mem = memory({_format_md(bias_tz, format_ldgo), cpu_engine}, bias.data_ptr());
  auto output_usr_mem = memory({output_md, cpu_engine}, output.data_ptr());
  auto hidden_out_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_out.data_ptr());
  auto grad_input_usr_mem = memory({input_md, cpu_engine}, grad_input.data_ptr());
  auto grad_hidden_in_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, grad_hidden_in.data_ptr());
  auto grad_weight_ih_usr_mem = memory({_format_md(weight_ih_tz, format_ldgoi), cpu_engine}, grad_weight_ih.data_ptr());
  auto grad_weight_hh_usr_mem = memory({_format_md(weight_hh_tz, format_ldgoi), cpu_engine}, grad_weight_hh.data_ptr());
  auto grad_bias_usr_mem = memory({_format_md(bias_tz, format_ldgo), cpu_engine}, grad_bias.data_ptr());
  auto grad_output_usr_mem = memory({output_md, cpu_engine}, grad_output.data_ptr());
  auto grad_hidden_out_use_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, grad_hidden_out.data_ptr());



  // print_tensor(grad_output_usr_mem, "grad_output_usr_mem");
  //print_tensor(grad_hidden_out_use_mem, "grad_hidden_out_use_mem");
  
  auto input_mem = _reorder(input_usr_mem, rnn_backward_pd.src_layer_primitive_desc());
  auto hidden_in_mem = _reorder(hidden_in_usr_mem, rnn_backward_pd.src_iter_primitive_desc());
  auto weight_ih_mem = _reorder(weight_ih_usr_mem, rnn_backward_pd.weights_layer_primitive_desc());
  auto weight_hh_mem = _reorder(weight_hh_usr_mem, rnn_backward_pd.weights_iter_primitive_desc());
  auto bias_mem = _reorder(bias_usr_mem, rnn_backward_pd.bias_primitive_desc());
  auto output_mem = _reorder(output_usr_mem, rnn_backward_pd.dst_layer_primitive_desc());
  auto hidden_out_mem = _reorder(hidden_out_usr_mem, rnn_backward_pd.dst_iter_primitive_desc());
  //auto grad_input_mem = _reorder(grad_input_usr_mem, rnn_backward_pd.diff_src_layer_primitive_desc());
  //auto grad_hidden_in_mem = _reorder(grad_hidden_in_usr_mem, rnn_backward_pd.diff_src_iter_primitive_desc());
  auto grad_weight_ih_mem = _reorder_prepare(grad_weight_ih_usr_mem, rnn_backward_pd.diff_weights_layer_primitive_desc());
  auto grad_weight_hh_mem = _reorder_prepare(grad_weight_hh_usr_mem, rnn_backward_pd.diff_weights_iter_primitive_desc());
  //auto grad_bias_mem = _reorder(grad_bias_usr_mem, rnn_backward_pd.diff_bias_primitive_desc());
  auto grad_output_mem = _reorder(grad_output_usr_mem, rnn_backward_pd.diff_dst_layer_primitive_desc());
  auto grad_hidden_out_mem = _reorder(grad_hidden_out_use_mem, rnn_backward_pd.diff_dst_iter_primitive_desc());
  auto workspace_mem = memory(rnn_backward_pd.workspace_primitive_desc(), workspace.data_ptr());


  memory_zeros(grad_weight_ih_mem, grad_weight_ih.numel());
  memory_zeros(grad_weight_hh_mem, grad_weight_hh.numel());

  //print_tensor(input_mem,"input_mem");
  //print_tensor(hidden_in_mem, "hidden_in_mem");
  //print_tensor(weight_ih_usr_mem, "weight_ih_usr_mem");
  //print_tensor(weight_ih_mem, "weight_ih_mem");
  //print_tensor(weight_hh_mem, "weight_hh_mem");
  //print_tensor(output_mem, "output_mem");
  //print_tensor(hidden_out_mem, "hidden_out_mem");
  //print_tensor(grad_input_usr_mem, "grad_input_usr_mem");
  //print_tensor(grad_hidden_in_usr_mem, "grad_hidden_in_usr_mem");
  //print_tensor(grad_output_mem, "grad_output_mem");
  //print_tensor(grad_hidden_out_mem, "grad_hidden_out_mem");

  std::vector<primitive> net;
  net.push_back(rnn_backward(rnn_backward_pd, input_mem, hidden_in_mem, weight_ih_mem, weight_hh_mem, bias_mem,
    output_mem, hidden_out_mem, grad_input_usr_mem, grad_hidden_in_usr_mem, grad_weight_ih_mem, grad_weight_hh_mem,
    grad_bias_usr_mem, grad_output_mem, grad_hidden_out_mem, workspace_mem));

  if (grad_weight_ih_mem != grad_weight_ih_usr_mem) {
    net.push_back(reorder(grad_weight_ih_mem, grad_weight_ih_usr_mem));
  }
  if (grad_weight_hh_mem != grad_weight_hh_usr_mem) {
    net.push_back(reorder(grad_weight_hh_mem, grad_weight_hh_usr_mem));
  }


/*
  if (grad_input_mem != grad_input_usr_mem) {
    net.push_back(reorder(grad_input_mem, grad_input_usr_mem));
  }
  if (grad_hidden_in_mem != grad_hidden_in_usr_mem) {
    net.push_back(reorder(grad_hidden_in_mem, grad_hidden_in_usr_mem));
  }
  if (grad_weight_ih_mem != grad_weight_ih_usr_mem) {
    net.push_back(reorder(grad_weight_ih_mem, grad_weight_ih_usr_mem));
  }
  if (grad_weight_hh_mem != grad_weight_hh_usr_mem) {
    net.push_back(reorder(grad_weight_hh_mem, grad_weight_hh_usr_mem));
  }
  if (grad_bias_mem != grad_bias_usr_mem) {
    net.push_back(reorder(grad_bias_mem, grad_bias_usr_mem));
  }
*/
  Stream::Instance().get_stream().submit(net);


  //print_tensor(grad_weight_ih_mem, "grad_weight_ih_mem");
  //print_tensor(grad_weight_ih_usr_mem, "grad_weight_ih_usr_mem");
} catch (error &e) {
    std::cerr << "message: " << e.message << std::endl;
}
  std::vector<Tensor> grad_weights;
  grad_weights.emplace_back(grad_weight_ih);
  grad_weights.emplace_back(grad_weight_hh);
  if (weight.size() == 4) {
#if 1
    if (celltype == MKLDNN_GRU) {
      //print_tensor(weight[2], "weight[2]");
      auto grad_bx = at::zeros_like(weight[2]);
      //auto grad_bx = at::zeros({3 * hidden_size}, grad_bias.options());
      auto grad_bh = at::zeros({3 * hidden_size}, grad_bias.options());
      std::vector<Tensor> grad_bx_chunk = grad_bx.chunk(3, 0);
      std::vector<Tensor> grad_bh_chunk = grad_bh.chunk(3, 0);
      std::vector<Tensor> chunks = grad_bias.chunk(4, 0);
      //print_tensor(grad_bx_chunk[0], "grad_bx_chunk[0]");
      //print_tensor(chunks[0], "chunks[0]");
      grad_bx_chunk[0].copy_(chunks[0]);
      grad_bh_chunk[0].copy_(chunks[0]);
      grad_bx_chunk[1].copy_(chunks[1]);
      grad_bh_chunk[1].copy_(chunks[1]);
      grad_bx_chunk[2].copy_(chunks[2]);
      grad_bh_chunk[2].copy_(chunks[3]);

      grad_weights.emplace_back(grad_bx);
      grad_weights.emplace_back(grad_bh);
      //print_tensor(grad_bias, "grad_bias");
      //print_tensor(grad_bx, "grad_bx");
      //print_tensor(grad_bh, "grad_bh");
    } else {
      grad_weights.emplace_back(grad_bias);
      grad_weights.emplace_back(grad_bias);
    }
#else

      grad_weights.emplace_back(grad_bias);
      grad_weights.emplace_back(grad_bias);
#endif
  }


  return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>{grad_input, grad_hx, grad_cx, grad_weights};

}

Tensor mkldnn_rnn_flatten_weight(
    TensorList weight_arr, int64_t weight_stride0,
    int64_t input_size,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_bidirectional
    ) {
}

}} // namespace at::native

#endif
