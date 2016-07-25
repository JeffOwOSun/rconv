#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/recurrent_conv_layer.hpp"

#ifdef USE_CUDNN
#include "this_file_will_not_exist.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RecurrentConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RecurrentConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 3, 6, 4)),
	frame_idx_(new Blob<Dtype>(1, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    // set frame index
    frame_idx_->mutable_cpu_data()[0] = 0;
    blob_bottom_vec_.push_back(frame_idx_);
    // push top as place holder
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~RecurrentConvolutionLayerTest() {
    delete blob_bottom_;
    delete frame_idx_;
    delete blob_top_;
  }

//   virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
//     this->ref_blob_top_.reset(new Blob<Dtype>());
//     this->ref_blob_top_->ReshapeLike(*top);
//     return this->ref_blob_top_.get();
//   }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const frame_idx_;
  Blob<Dtype>* const blob_top_;

  // shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RecurrentConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(RecurrentConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(4);

  shared_ptr<Layer<Dtype> > layer(new RecurrentConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);

  // setting group should not change the shape
  convolution_param->set_num_output(3);
  layer.reset(new RecurrentConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(RecurrentConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();

  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->set_bias_term(false);
  convolution_param->mutable_weight_filler()->set_type("gaussian");

  RecurrentConvolutionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  //  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  //  this->frame_idx_->mutable_cpu_data()[0] = 1;
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				  this->blob_top_vec_, 0);

//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
//   this->frame_idx_->mutable_cpu_data()[0] = 2;
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
//   this->frame_idx_->mutable_cpu_data()[0] = 3;
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);

}

}  // namespace caffe
