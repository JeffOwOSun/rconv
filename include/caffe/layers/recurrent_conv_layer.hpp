#ifndef CAFFE_RECURRENT_CONV_LAYER_HPP_
#define CAFFE_RECURRENT_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class RecurrentConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - dilation (\b optional, default 1). The filter
   *  dilation, given by dilation_size for equal dimensions for different
   *  dilation. By default the convolution has dilation 1.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit RecurrentConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Recurrent Convolution"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//       const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

private:
  vector<shared_ptr<Blob<Dtype> > > rconv_blobs_;
  // output of previous frames
  vector<shared_ptr<Blob<Dtype> > > a_ts_; // history of output
  vector<shared_ptr<Blob<Dtype> > > z_ts_; // history of pre-relu output
  vector<shared_ptr<Blob<Dtype> > > a_lm1_ts_; // history of input

  bool rconv_filter_is_1x1_;
  vector<int> frame_shape_;


  void rconv_backward_cpu(const Blob<Dtype>* top_diff,
			 Dtype* Wa_diff, Dtype* Wh_diff);
};

}  // namespace caffe

#endif  // CAFFE_RECURRENT_CONV_LAYER_HPP_
