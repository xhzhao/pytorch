#ifndef THNN_OMP_OVERHEAD_THRESHOLD
#define THNN_OMP_OVERHEAD_THRESHOLD 5000
#endif
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LeakyReLU.c"
#else

void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal negval_,
          bool inplace)
{
  scalar_t negval = TH_CONVERT_ACCREAL_TO_REAL(negval_);
  if (inplace)
  {
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP)
    {
      serial_path = 1;
    }
    else
    {
      TH_TENSOR_APPLY_OMP(scalar_t, input,
        if ((*input_data) <= 0)
          *input_data *= negval;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path)
    {
      TH_TENSOR_APPLY(scalar_t, input,
        if (*input_data <= 0)
          *input_data *= negval;
      );
    }
    THTensor_(set)(output, input);
  }
  else
  {
    THTensor_(resizeAs)(output, input);
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP)
    {
      serial_path = 1;
    }
    else
    {
      int64_t output_size = THTensor_(nElement)(output);
      int output_contig = THTensor_(isContiguous)(output);
      int input_contig = THTensor_(isContiguous)(input);
      TH_TENSOR_APPLY2_OMP(output_size, output_contig, input_contig, scalar_t, output, scalar_t, input,
        const scalar_t r = (*input_data > 0) ? 1 : negval;
        *output_data = *input_data * r;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) {
      TH_TENSOR_APPLY2(scalar_t, output, scalar_t, input,
        const scalar_t r = (*input_data > 0) ? 1 : negval;
        *output_data = *input_data * r;
      );
    }
  }
}

void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal negval_,
          bool inplace)
{
  scalar_t negval = TH_CONVERT_ACCREAL_TO_REAL(negval_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  if (inplace)
  {
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP)
    {
      serial_path = 1;
    }
    else
    {
      int64_t gradOutput_size = THTensor_(nElement)(gradOutput);
      int gradOutput_contig = THTensor_(isContiguous)(gradOutput);
      int input_contig = THTensor_(isContiguous)(input);
      TH_TENSOR_APPLY2_OMP(gradOutput_size, gradOutput_contig, input_contig, scalar_t, gradOutput, scalar_t, input,
        if ((*input_data) <= 0)
          *gradOutput_data *= negval;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path) 
    {
      TH_TENSOR_APPLY2(scalar_t, gradOutput, scalar_t, input,
        if (*input_data <= 0)
          *gradOutput_data *= negval;
      );
    }
    THTensor_(set)(gradInput, gradOutput);
  }
  else
  {
    THTensor_(resizeAs)(gradInput, input);
    int serial_path = 0;
#ifdef _OPENMP
    int inOMP = omp_in_parallel();
    if (inOMP)
    {
      serial_path = 1;
    }
    else
    {
      int64_t gradInput_size = THTensor_(nElement)(gradInput);
      int gradInput_contig = THTensor_(isContiguous)(gradInput);
      int gradOutput_contig = THTensor_(isContiguous)(gradOutput);
      int input_contig = THTensor_(isContiguous)(input);
      TH_TENSOR_APPLY3_OMP(gradInput_size, gradInput_contig, gradOutput_contig, input_contig, scalar_t, gradInput,scalar_t, gradOutput, scalar_t, input,
        *gradInput_data = *input_data > 0 ? *gradOutput_data : *gradOutput_data * negval;,
        THNN_OMP_OVERHEAD_THRESHOLD
      );
    }
#else
    serial_path = 1;
#endif
    if (serial_path)
    {
      TH_TENSOR_APPLY3(scalar_t, gradInput, scalar_t, gradOutput, scalar_t, input,
        *gradInput_data = *input_data > 0 ? *gradOutput_data : *gradOutput_data * negval;
      );
    }
  }
}

#undef THNN_OMP_OVERHEAD_THRESHOLD
#endif
