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

#include "lite/api/compute_api.h"  // NOLINT
#include <iostream>
#include <memory>
#include <vector>

using namespace paddle::lite;  // NOLINT

int main(int argc, char** argv) {
  printf("--------pooling test--------\n");
  // TEST pooling
  ComputeEngine<TARGET(kARM)> ce;
  // Create Operator
  OpHandle op_h = ce.CreateOperator("pool2d");
  operators::PoolParam pool_param;
  Tensor x, y;
  x.Resize({1, 2, 5, 5});
  float* indata = x.mutable_data<float>();
  for (int i = 0; i < x.numel(); ++i) {
    indata[i] = 1.f;
  }
  pool_param.x = &x;
  pool_param.output = &y;
  pool_param.global_pooling = true;
  pool_param.ksize = {2, 2};
  pool_param.paddings = std::make_shared<std::vector<int>>(4, 0);
  pool_param.strides = {2, 2};
  pool_param.pooling_type = "avg";
  // SetParam
  ce.SetParam(op_h, &pool_param);
  // Launch
  ce.Launch(op_h);
  const float* outdata = y.data<float>();
  DDim out_dim = y.dims();
  std::cout << "outdim: {" << out_dim[0] << ", " << out_dim[1] << ", "
            << out_dim[2] << ", " << out_dim[3] << "}\n";
  printf("pooling out: \n");
  for (int i = 0; i < y.numel(); ++i) {
    printf("%.3f ", outdata[i]);
  }
  printf("\n");
  // Release OpHandle
  ce.ReleaseOpHandle(op_h);

  printf("--------conv 1x1s1p0 sgemm test--------\n");
  // TEST  1x1s1p0 sgemm
  op_h = ce.CreateOperator("conv2d");
  operators::ConvParam conv_param;
  x.Resize({1, 3, 5, 5});
  indata = x.mutable_data<float>();
  for (int i = 0; i < x.numel(); ++i) {
    indata[i] = 1.f;
  }
  Tensor filter;
  filter.Resize({2, 3, 1, 1});
  float* filter_data = filter.mutable_data<float>();
  for (int i = 0; i < filter.numel(); ++i) {
    filter_data[i] = 1.f;
  }
  conv_param.filter = &filter;
  conv_param.x = &x;
  conv_param.output = &y;
  conv_param.paddings = std::make_shared<std::vector<int>>(4, 0);  // pad 0
  conv_param.dilations = std::make_shared<std::vector<int>>(2, 1);

  ce.SetParam(op_h, &conv_param);
  ce.Launch(op_h);

  outdata = y.data<float>();
  out_dim = y.dims();
  std::cout << "outdim: {" << out_dim[0] << ", " << out_dim[1] << ", "
            << out_dim[2] << ", " << out_dim[3] << "}\n";
  printf("1x1s1p0 sgemm out: \n");
  for (int i = 0; i < y.numel(); ++i) {
    printf("%.3f ", outdata[i]);
  }
  printf("\n");
  ce.ReleaseOpHandle(op_h);

  printf("--------conv 3x3s1p1 depthwise test--------\n");
  // TEST  3x3s1p1 depthwise
  op_h = ce.CreateOperator("conv2d");  // create a new conv2d is necessary
  x.Resize({1, 3, 5, 5});
  indata = x.mutable_data<float>();
  for (int i = 0; i < x.numel(); ++i) {
    indata[i] = 1.f;
  }
  filter.Resize({3, 1, 3, 3});
  filter_data = filter.mutable_data<float>();
  for (int i = 0; i < filter.numel(); ++i) {
    filter_data[i] = 1.f;
  }

  conv_param.paddings = std::make_shared<std::vector<int>>(4, 1);  // pad 1
  conv_param.groups = 3;  // for depthwise

  ce.SetParam(op_h, &conv_param);
  // Launch
  ce.Launch(op_h);
  outdata = y.data<float>();
  out_dim = y.dims();
  std::cout << "outdim: {" << out_dim[0] << ", " << out_dim[1] << ", "
            << out_dim[2] << ", " << out_dim[3] << "}\n";
  printf("3x3s1p1 depthwise out: \n");
  for (int i = 0; i < y.numel(); ++i) {
    printf("%.3f ", outdata[i]);
  }
  printf("\n");
  ce.ReleaseOpHandle(op_h);
  return 0;
}
