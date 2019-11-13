// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

inline std::vector<int> get_new_shape(
    const std::vector<const lite::Tensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  lite::Tensor cpu_starts_tensor;
  vec_new_data =
      std::vector<T>(new_data, new_data + new_data_tensor->dims().production());
  return vec_new_data;
}

template <typename dtype>
void resize_nearest_align(std::vector<const lite::Tensor*> inputs,
                          lite::Tensor* output,
                          bool with_align) {
  int hin = inputs[0]->dims()[2];
  int win = inputs[0]->dims()[3];
  int channels = inputs[0]->dims()[1];
  int num = inputs[0]->dims()[0];
  int hout = output->dims()[2];
  int wout = output->dims()[3];
  dtype scale_w = (with_align) ? (static_cast<float>(win - 1) / (wout - 1))
                               : (static_cast<float>(win) / (wout));
  dtype scale_h = (with_align) ? (static_cast<float>(hin - 1) / (hout - 1))
                               : (static_cast<float>(hin) / (hout));
  const dtype* src = inputs[0]->data<dtype>();
  dtype* dst = output->mutable_data<dtype>();
  int dst_stride_w = 1;
  int dst_stride_h = wout;
  int dst_stride_c = wout * hout;
  int dst_stride_batch = wout * hout * channels;
  int src_stride_w = 1;
  int src_stride_h = win;
  int src_stride_c = win * hin;
  int src_stride_batch = win * hin * channels;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int src_index = n * src_stride_batch + c * src_stride_c;
      for (int h = 0; h < hout; ++h) {
        for (int w = 0; w < wout; ++w) {
          int fw = (with_align) ? static_cast<int>(scale_w * w + 0.5)
                                : static_cast<int>(scale_w * w);
          fw = (fw < 0) ? 0 : fw;
          int fh = (with_align) ? static_cast<int>(scale_h * h + 0.5)
                                : static_cast<int>(scale_h * h);
          fh = (fh < 0) ? 0 : fh;
          int w_start = static_cast<int>(fw);
          int h_start = static_cast<int>(fh);
          int dst_index = n * dst_stride_batch + c * dst_stride_c +
                          h * dst_stride_h + w * dst_stride_w;
          dst[dst_index] =
              src[src_index + w_start * src_stride_w + h_start * src_stride_h];
        }
      }
    }
  }
}

class NearestInterpComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "X";
  std::string sizetensor0_ = "SizeTensor0";
  std::string sizetensor1_ = "SizeTensor1";
  std::string input_scale_ = "Scale";
  std::string input1_ = "OutSize";
  std::string output_ = "Out";

  float height_scale_ = 0.0f;
  float width_scale_ = 0.0f;
  int out_height_ = -1;
  int out_width_ = -1;
  bool align_corners_ = true;
  std::string interp_method_ = "Nearest";
  DDim dims_{{2, 3}};
  DDim _dims0_{{2, 3, 3, 2}};
  DDim _dims1_{{2}};
  DDim sizetensor_dims_{{1}};
  DDim scale_dims_{{1}};

 public:
  NearestInterpComputeTester(const Place& place,
                             const std::string& alias,
                             float height_scale,
                             float width_scale,
                             int out_height,
                             int out_width,
                             bool align_corners,
                             std::string interp_method)
      : TestCase(place, alias),
        height_scale_(height_scale),
        width_scale_(width_scale),
        out_height_(out_height),
        out_width_(out_width),
        align_corners_(align_corners),
        interp_method_(interp_method) {}

  void RunBaseline(Scope* scope) override {
    width_scale_ = height_scale_;
    auto* outputs = scope->NewTensor(output_);
    CHECK(outputs);
    outputs->Resize(dims_);
    std::vector<const lite::Tensor*> inputs;
    inputs.emplace_back(scope->FindTensor(input0_));
    inputs.emplace_back(scope->FindTensor(input1_));

    std::vector<const lite::Tensor*> SizeTensor(2);
    SizeTensor[0] = scope->FindTensor(sizetensor0_);
    SizeTensor[1] = scope->FindTensor(sizetensor1_);
    const lite::Tensor* input_scale = scope->FindTensor(input_scale_);

    float scale = height_scale_;
    int in_h = inputs[0]->dims()[2];
    int in_w = inputs[0]->dims()[3];
    if (SizeTensor.size() > 0) {
      auto new_size = get_new_shape(SizeTensor);
      out_height_ = new_size[0];
      out_width_ = new_size[1];
    } else {
      auto scale_tensor = input_scale;
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        scale = scale_data[0];
      }
      if (scale > 0) {
        out_height_ = static_cast<int>(in_h * scale);
        out_width_ = static_cast<int>(in_w * scale);
      }
      auto out_size = inputs[1];
      if (out_size != nullptr) {
        auto out_size_data = get_new_data_from_tensor<int>(out_size);
        out_height_ = out_size_data[0];
        out_width_ = out_size_data[1];
      }
    }
    height_scale_ = scale;
    width_scale_ = scale;

    if (out_width_ != -1 && out_height_ != -1) {
      height_scale_ = static_cast<float>(out_height_ / inputs[0]->dims()[2]);
      width_scale_ = static_cast<float>(out_width_ / inputs[0]->dims()[3]);
    }
    int num_cout = inputs[0]->dims()[0];
    int c_cout = inputs[0]->dims()[1];
    outputs->Resize({num_cout, c_cout, out_height_, out_width_});

    resize_nearest_align<float>(inputs, outputs, align_corners_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("nearest_interp");
    op_desc->SetInput("X", {input0_});
    op_desc->SetInput("SizeTensor", {sizetensor0_, sizetensor1_});
    op_desc->SetInput("Scale", {input_scale_});
    op_desc->SetInput("OutSize", {input1_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("scale", height_scale_);
    op_desc->SetAttr("out_h", out_height_);
    op_desc->SetAttr("out_w", out_width_);
    op_desc->SetAttr("align_corners", align_corners_);
    op_desc->SetAttr("interp_method", interp_method_);
  }

  void PrepareData() override {
    std::vector<float> data0(_dims0_.production());
    for (int i = 0; i < _dims0_.production(); i++) {
      data0[i] = i * 1.1;
    }

    std::vector<int> data1(_dims1_.production());
    for (int i = 0; i < _dims1_.production(); i++) {
      data1[i] = (i + 1) * 2;
    }

    SetCommonTensor(input0_, _dims0_, data0.data());
    SetCommonTensor(input1_, _dims1_, data1.data());

    std::vector<int> sizetensor_data(1);
    sizetensor_data[0] = out_height_;
    SetCommonTensor(sizetensor0_, sizetensor_dims_, sizetensor_data.data());

    sizetensor_data[0] = out_width_;
    SetCommonTensor(sizetensor1_, sizetensor_dims_, sizetensor_data.data());

    std::vector<float> scale_data(1);
    scale_data[0] = height_scale_;
    SetCommonTensor(input_scale_, scale_dims_, scale_data.data());
  }
};

void test_nearest_interp(Place place) {
  std::string interp_method = "Nearest";
  for (float scale : {0.123, 2., 1.2}) {
    for (int out_height : {2, 1, 6}) {
      for (int out_width : {2, 3, 5}) {
        for (bool align_corners : {true, false}) {
          std::unique_ptr<arena::TestCase> tester(
              new NearestInterpComputeTester(place,
                                             "def",
                                             scale,
                                             scale,
                                             out_height,
                                             out_width,
                                             align_corners,
                                             interp_method));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(NearestInterp, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_nearest_interp(place);
#endif
}

}  // namespace lite
}  // namespace paddle
