#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/recurrent_conv_layer.hpp"

// for testing
#include <iostream>
#include <iomanip>

namespace caffe {

template <typename Dtype>
inline int get_frame_index(const Blob<Dtype>* blob) {
  return int(blob->cpu_data()[0]);
}

template <typename Dtype>
void log_shape(const Blob<Dtype>* blob) {
  LOG(INFO) << blob->shape(0) << ", " << blob->shape(1) << ", "
	    << blob->shape(2) << ", " << blob->shape(3);
}

template <typename Dtype>
void log_first_channel_of_blob(const Blob<Dtype>* blob) {
  const int h = blob -> shape(2);
  const int w = blob -> shape(3);
  const Dtype* data = blob->cpu_data();

  for (int r = 0; r < std::min(10, h); ++r) {
    for (int c = 0; c < std::min(10, w); ++c) {
      std::cout << std::fixed << std::setw(9)// << std::setprecision(2)
		<< data[r * w + c] << ", ";
    }
    std::cout << std::endl;
  }
}

template <typename Dtype>
void log_first_channel_diff_of_blob(const Blob<Dtype>* blob) {
  const int h = blob -> shape(2);
  const int w = blob -> shape(3);
  const Dtype* data = blob->cpu_diff();

  for (int r = 0; r < std::min(10, h); ++r) {
    for (int c = 0; c < std::min(10, w); ++c) {
      std::cout << std::fixed << std::setw(9) //<< std::setprecision(2)
		<< data[r * w + c] << ", ";
    }
    std::cout << std::endl;
  }
}

template <typename Dtype>
static void rconv_identity_fill_weight(Blob<Dtype>* blob) {
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
static void set_blob_constant(Blob<Dtype>* blob, Dtype val) {
  Dtype* data = blob->mutable_cpu_data();
  for (int i = 0; i < blob->count(); ++i) {
    data[i] = val;
  }
}

template <typename Dtype>
static Blob<Dtype>* rconv_backward_relu(const Blob<Dtype>* top,
					const Blob<Dtype>* bottom)
{
  CHECK(top->shape() == bottom->shape()) << "Shape mismatch";
  const Dtype* bottom_data = bottom->cpu_data();
  const Dtype* top_diff = top->cpu_diff();
  Blob<Dtype>* back_relu_top(new Blob<Dtype>(top->shape()));
  Dtype* back_relu_top_diff = back_relu_top->mutable_cpu_diff();

  for (int i = 0; i < top->count(); ++i) {
    back_relu_top_diff[i] = top_diff[i] * Dtype(bottom_data[i] > 0);
  }
  return back_relu_top;
}

template <typename Dtype>
static Blob<Dtype>* rconv_relu_mask(const Blob<Dtype>* bottom)
{
  Blob<Dtype>* relu_mask(new Blob<Dtype>(bottom->shape()));
  Dtype* relu_mask_diff = relu_mask->mutable_cpu_diff();
  const Dtype* bottom_data = bottom->cpu_data();

  for (int i = 0; i < bottom->count(); ++i) {
    relu_mask_diff[i] = Dtype(bottom_data[i] > 0);
  }
  return relu_mask;
}


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
  CHECK(this->num_spatial_axes_ == 2) << "Should be 2d conv";
  rconv_filter_is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    rconv_weight_shape.push_back(1);
  }
  rconv_blobs_.resize(1);
  rconv_blobs_[0].reset(new Blob<Dtype>(rconv_weight_shape));
  // fill the weight similarly to iRNN but in conv manner
  rconv_identity_fill_weight(rconv_blobs_[0].get());

  //TODO: de-comment this
  this->blobs_.push_back(rconv_blobs_[0]);
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // reshape is called in both SetUp and Forward phases
  // assumption check
  CHECK(bottom.size() == 2) << "Bottom should contain one frame and frame_info";
  CHECK(top.size() == 1) << "One frame out at a time";

  LOG(INFO) << "Reshaping";

  const int frame_idx = get_frame_index(bottom[1]);
  if (frame_idx == 0) {
    BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
    frame_shape_ = bottom[0]->shape();

    // reset history storage
    z_ts_.resize(1);
    a_ts_.resize(1);
    a_lm1_ts_.resize(1);
    z_ts_[0].reset(new Blob<Dtype>(top[0]->shape()));
    a_ts_[0].reset(new Blob<Dtype>(top[0]->shape()));
    a_lm1_ts_[0].reset(new Blob<Dtype>(bottom[0]->shape()));
    set_blob_constant(z_ts_[0].get(), Dtype(0));
    set_blob_constant(a_ts_[0].get(), Dtype(0));
    set_blob_constant(a_lm1_ts_[0].get(), Dtype(0));
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

// TODO: put this in some namespace
template <typename Dtype>
static void push_back_history(const Blob<Dtype>* new_val,
			      vector<shared_ptr<Blob<Dtype> > >& history)
{
  CHECK(history.size() > 0) << "Length of history cannot be zero";
  shared_ptr<Blob<Dtype> > prev = history.back();
  CHECK(prev->shape() == new_val->shape()) << 
    "Shape mismatch between new_val and history";
  shared_ptr<Blob<Dtype> > curr(new Blob<Dtype>(new_val->shape()));
  curr->CopyFrom(*new_val);
//   LOG(INFO) << "pushing back new val:";
//   log_first_channel_of_blob(new_val);
  history.push_back(curr);
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == 2) << "rconv layer takes 2 blobs (data, frame_info) as input";
  LOG(INFO) << "frame index: " << get_frame_index(bottom[1]);

  // z_t = conv(W_x, x) + conv(W_t * a_tm1)
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  push_back_history(bottom[0], a_lm1_ts_);

  CHECK(this->num_ == 1) << "Only support 1 image per batch";
  this->forward_cpu_gemm(bottom_data, weight, top_data);
  if (this->bias_term_) {
    const Dtype* bias = this->blobs_[1]->cpu_data();
    this->forward_cpu_bias(top_data, bias);
  }
  // now top_data = conv(W_x, x) + bias, NEXT: add conv(W_t * a_tm1) to it
  // filter is 1x1, no im2col required
  CHECK(rconv_filter_is_1x1_ == true) << "Only support 1x1 conv for t";
  CHECK(this->group_ == 1) << "Only support group = 1";
  const shared_ptr<Blob<Dtype> > a_tm1 = a_ts_.back();
  const int rconv_kernel_dim = 1;
  const Dtype* rconv_weight = rconv_blobs_[0]->cpu_data();
  const Dtype* rconv_input = a_tm1->cpu_data();
  { // for debug
    LOG(INFO) << "normal conv";
    log_first_channel_of_blob(top[0]);
    LOG(INFO) << "a_t-1";
    LOG(INFO) << a_ts_.size();
    log_first_channel_of_blob(a_tm1.get());
  }
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			this->num_output_, this->out_spatial_dim_, rconv_kernel_dim,
			(Dtype)1., rconv_weight, rconv_input, (Dtype)1., top_data);
  push_back_history(top[0], z_ts_);
  // top_data = z_t = conv(W_x, x) + conv(W_t, a_tm1), NEXT: append z_t, apply ReLU

  const int relu_count = top[0]->count();
  for (int i = 0; i < relu_count; ++i) {
    top_data[i] = std::max(top_data[i], Dtype(0));
  }
  push_back_history(top[0], a_ts_);
  // now top_data = a_t = relu(z_t)
  LOG(INFO) << "a_t";
  log_first_channel_of_blob(a_ts_.back().get());

  // check length of history
  CHECK(a_ts_.size() == z_ts_.size()) << "history mismatch";
  CHECK(a_ts_.size() == a_lm1_ts_.size()) << "history mismatch";
//   CHECK(a_ts_.size() == get_frame_index(bottom[1])+2)
//     << "hisroty-frame_index mismatch; "
//     << a_ts_.size() << " vs " << get_frame_index(bottom[1]);
}

template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(top.size() == 1) << "Only one output blob";
  CHECK(bottom.size() == 2) << "Image Blob and Frame Info";
  CHECK(a_ts_.size() == z_ts_.size()) << "history mismatch";
  CHECK(this->num_ == 1) << "One image per batch";

  Dtype* Wa_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* Wh_diff = rconv_blobs_[0]->mutable_cpu_diff();

  LOG(INFO) << "top_diff, pre_relu";
  log_first_channel_diff_of_blob(top[0]);

  // should delete this!!!!!
  const Blob<Dtype>* const back_relu_top = rconv_backward_relu(top[0],z_ts_.back().get());

  LOG(INFO) << "top_diff, post_relu";
  log_first_channel_diff_of_blob(back_relu_top);

  // gradient w.r.t. weight. Note that we will accumulate diffs.
  CHECK(this->param_propagate_down_[0] == true) << "should propagate down";
  rconv_backward_cpu(back_relu_top, Wa_diff, Wh_diff);

  LOG(INFO) << "Wa_diff";
  log_shape(this->blobs_[0].get());
  log_first_channel_diff_of_blob(this->blobs_[0].get());
  LOG(INFO) << "Wh_diff";
  log_first_channel_diff_of_blob(rconv_blobs_[0].get());


  CHECK(this->bias_term_ == false) << "no bias term during test";
//   // Bias gradient, if necessary.
//   if (this->bias_term_ && this->param_propagate_down_[1]) {
//     Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
//     this->backward_cpu_bias(bias_diff, top_diff);
//   }

  // gradient w.r.t. bottom data, if necessary.
  // this is the same as in normal conv layer, RIGHT?
  const Dtype* top_diff = back_relu_top->cpu_diff(); // d(J)/d(z_t)
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    this->backward_cpu_gemm(top_diff, weight, bottom_diff);
  }

  delete back_relu_top;
}


template <typename Dtype>
static void rconv_update_dt_acc(const Blob<Dtype>* W_h,
				const Blob<Dtype>* z_t,
				Blob<Dtype>*& dt_acc) {
  LOG(INFO) << "into rconv_update";
  const Blob<Dtype>* const relu_mask = rconv_relu_mask(z_t);
  Blob<Dtype>* new_dt_acc(new Blob<Dtype>(dt_acc->shape()));

//   log_shape(W_h);
//   log_shape(dt_acc);

  const int output_channel = W_h->count(1);
  const int kernel_dim = W_h->shape(0); // W_h will be transposed
  const int output_spatial = dt_acc->count(2);
  LOG(INFO) << "in update rconv_update_dt_acc: M, N, K: " << output_channel << ", "
	    << output_spatial << ", " << kernel_dim;
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			output_channel, output_spatial, kernel_dim,
			(Dtype)1., W_h->cpu_data(), dt_acc->cpu_diff(),
			(Dtype)0., new_dt_acc->mutable_cpu_diff());

  LOG(INFO) << "new_dt_acc";
  log_first_channel_diff_of_blob(new_dt_acc);
//   CHECK(relu_mask->count() == new_dt_acc->count()) << "Shape mismatch";
//   LOG(INFO) << "mask_shape";
//   log_shape(relu_mask);
//   LOG(INFO) << "conv(W_h_trans, dt_acc) shape";
//   log_shape(new_dt_acc);

  caffe_mul(new_dt_acc->count(), relu_mask->cpu_diff(), new_dt_acc->cpu_diff(),
 	    new_dt_acc->mutable_cpu_diff());
  std::swap(new_dt_acc, dt_acc);
  delete new_dt_acc;
  delete relu_mask;
}


template <typename Dtype>
void RecurrentConvolutionLayer<Dtype>::rconv_backward_cpu(const Blob<Dtype>* top,
							  Dtype* const Wa_diff,
							  Dtype* const Wh_diff) {
  Blob<Dtype>* dt_acc(new Blob<Dtype>(top->shape())); // !!!delete this!!!
  dt_acc->CopyFrom(*top, true);

  LOG(INFO) << "top diff";
  log_first_channel_diff_of_blob(top);

  for (int t = a_ts_.size()-1; t > 0; --t) {
    LOG(INFO) << "t = " << t << ", dt_acc_diff";
    log_first_channel_diff_of_blob(dt_acc);

    const Blob<Dtype>* const a_lm1_t = a_lm1_ts_[t].get(); // don't delete this
    // !!! this may not be correct
    this->weight_cpu_gemm(a_lm1_t->cpu_data(), dt_acc->cpu_diff(), Wa_diff);

    { // dW_h
      CHECK(rconv_filter_is_1x1_) << "Only support 1x1 fitler for W_h now";
      const Blob<Dtype>* const a_tm1 = a_ts_[t-1].get(); // don't delete this
      const int output_channel = dt_acc->shape(1);
      const int kernel_dim = dt_acc->count(2);
      const int output_spatial = a_tm1->shape(1); // a_tm1 will be transpose

//       log_shape(dt_acc);
//       log_shape(a_tm1);
      LOG(INFO) << "in backward: M, N, K: " << output_channel << ", "
		<< output_spatial << ", " << kernel_dim;
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			    output_channel, output_spatial, kernel_dim,
			    (Dtype)1., dt_acc->cpu_diff(), a_tm1->cpu_data(),
			    (Dtype)1., Wh_diff);
    }
    // update dt_acc
    const Blob<Dtype>* const z_tm1 = z_ts_[t-1].get(); // don't delete this
    rconv_update_dt_acc(rconv_blobs_[0].get(), z_tm1, dt_acc);
  }
  delete dt_acc;
}

// #ifdef CPU_ONLY
// STUB_GPU(RecurrentConvolutionLayer);
// #endif

INSTANTIATE_CLASS(RecurrentConvolutionLayer);
REGISTER_LAYER_CLASS(RecurrentConvolution);

}  // namespace caffe
