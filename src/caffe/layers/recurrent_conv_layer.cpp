#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/recurrent_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  // Currently only support 1*1 conv for recurrent part
  // some assumptions to simplify first impl
  CHECK(this->group_ == 1) << "Only support group = 1";

  // init the rconv_blobs_, with bias
  LOG(INFO) << "Initializing recurrent conv weight & bias";
  vector<int> rconv_bias_shape(1, this->num_output_);
  vector<int> rconv_weight_shape(2, this->num_output_);
  CHECK(this->num_spatial_axes_ == 2) << "Just for fun";
  rconv_filter_is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    rconv_weight_shape.push_back(1);
  }
  rconv_blobs_.resize(2);
  rconv_blobs_[0].reset(new Blob<Dtype>(rconv_weight_shape));
  rconv_blobs_[1].reset(new Blob<Dtype>(rconv_bias_shape));
  // fill the weight similarly to iRNN but in conv manner
  rconv_identity_fill_weight(rconv_blobs_[0].get());
  set_blob_zero(rconv_blobs_[1].get());

  // copy rconv_blobs out to this->blobs_ for testing
//   CHECK(this->blobs_.size() == 2) << "Use bias term while testing";
//   this->blobs_.resize(4);
//   this->blobs_[2].reset(new Blob<Dtype>(rconv_weight_shape));
//   this->blobs_[3].reset(new Blob<Dtype>(rconv_bias_shape));
//   this->blobs_[2]->CopyFrom(*rconv_blobs_[0].get());
//   this->blobs_[3]->CopyFrom(*rconv_blobs_[1].get());
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // reshape is called in both SetUp and Forward phases
  // assumption check
  CHECK(bottom.size() == 2) << "Bottom should contain one frame and frame_info";
  CHECK(top.size() == 1) << "One frame out at a time";

  LOG(INFO) << "Reshaping";

  int frame_idx = get_frame_index(bottom[1]);
  if (frame_idx == 0) {
    BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
    frame_shape_ = bottom[0]->shape();

    // reset history storage
    a_tm1_.reset(new Blob<Dtype>(this->output_shape_));
    set_blob_zero(a_tm1_.get());

  } else {
    CHECK(frame_shape_ == bottom[0]->shape())
      << "All frames in the same video must have the same shape.";
  }
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::compute_output_shape() {
  // this is identical to normal conv layer
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();

  CHECK(bottom.size() == 2) << "rconv layer takes 2 blobs (data, frame_info) as input";

  // z_t = conv(W_x, x) + conv(W_t * a_tm1)
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  CHECK(this->num_ == 1) << "Only support 1 image per batch";
  for (int n = 0; n < this->num_; ++n) {
    this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
			   top_data + n * this->top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
  // now top_data = conv(W_x, x), NEXT: add conv(W_t * a_tm1) to it
  // filter is 1x1, no im2col required
  CHECK(rconv_filter_is_1x1_ == true) << "Only support 1x1 conv for t";
  CHECK(this->group_ == 1) << "Only support group = 1";
  const int rconv_kernel_dim = 1;
  const Dtype* rconv_weight = rconv_blobs_[0]->cpu_data();
  const Dtype* rconv_bias = rconv_blobs_[1]->cpu_data();
  const Dtype* rconv_input = a_tm1_->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			this->num_output_, this->out_spatial_dim_, rconv_kernel_dim,
			(Dtype)1., rconv_weight, rconv_input, (Dtype)1., top_data);
  this->forward_cpu_bias(top_data, rconv_bias);
  // now top_data = conv(W_x, x) + conv(W_t, a_tm1), NEXT: apply ReLU and update a_t-1

  LOG(INFO) << "frame index: " << get_frame_index(bottom[1]);
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::rconv_identity_fill_weight(Blob<Dtype>* blob) {
  Dtype* data = blob->mutable_cpu_data();
  int n_filter = blob->shape(0);
  int n_channel = blob->shape(1);
  CHECK(n_filter == n_channel) << "Input channel should be equal to output channel";

  for (int i = 0; i < blob->count(); ++i) {
    data[i] = Dtype(0);
  }
  // TODO: make sure this is identity
  for (int i = 0; i < n_filter; ++i) {
    int offset = blob->offset(i, i, 0, 0);
    data[offset] = Dtype(1);
  }
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::set_blob_zero(Blob<Dtype>* blob) {
  Dtype* data = blob->mutable_cpu_data();
  for (int i = 0; i < blob->count(); ++i) {
    data[i] = Dtype(0);
  }
}

// #ifdef CPU_ONLY
// STUB_GPU(RecurrentConvolutionLayer);
// #endif

INSTANTIATE_CLASS(RecurrentConvolutionLayer);
REGISTER_LAYER_CLASS(RecurrentConvolution);

}  // namespace caffe
