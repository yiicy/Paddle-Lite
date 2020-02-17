// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <string>
#include "lite/api/paddle_place.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {

//! one OpHandle will hold one kernel impl,
//! if you want to use two operators with the
//! same type, you must create two OpHandle
//! for example:
//! using namespace paddle::lite;
//! ComputeEngine ce;
//! OpHandle conv0_handle = ce.CreateOperator("conv2d");
//! OpHandle conv1_handle = ce.CreateOperator("conv2d");
//! ... ...
//! ce.ReleaseOpHandle(conv0_handle);
//! ce.ReleaseOpHandle(conv1_handle);
using OpHandle = void*;

// now ComputeEngine only support Target = Arm, Precison = kFloat
template <TargetType Type>
class ComputeEngine {
 public:
  ComputeEngine() = default;
  OpHandle CreateOperator(std::string op_type,
                          PrecisionType precision = PRECISION(kFloat),
                          DataLayoutType layout = DATALAYOUT(kNCHW)) {}
  void SetParam(OpHandle op_h, operators::ParamBase* param) {}
  void Launch(OpHandle op_h) {}
  void ReleaseOpHandle(OpHandle op_h) {}
  ~ComputeEngine() = default;
};

template <>
class ComputeEngine<TARGET(kARM)> {
 public:
  OpHandle CreateOperator(std::string op_type,
                          PrecisionType precision = PRECISION(kFloat),
                          DataLayoutType layout = DATALAYOUT(kNCHW));
  void SetParam(OpHandle op_h, operators::ParamBase* param);
  void Launch(OpHandle op_h);
  void ReleaseOpHandle(OpHandle op_h);
};

}  // namespace lite
}  // namespace paddle
