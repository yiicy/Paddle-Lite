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

#include "compute_api.h"  // NOLINT
#include <algorithm>
#include <utility>
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"

namespace paddle {
namespace lite {
OpHandle ComputeEngine<TARGET(kARM)>::CreateOperator(std::string op_type,
                                                     PrecisionType precision,
                                                     DataLayoutType layout) {
  LOG(INFO) << "Start Create Operator";
  auto op = LiteOpRegistry::Global().Create(op_type);
  CHECK(op) << "no Op found for " << op_type;
  LOG(INFO) << "Create Operator Success, Op Debug String: "
            << op->DebugString();
  lite_api::Place place(TARGET(kARM), precision, layout);
  auto kernels = op->CreateKernels({place});
  CHECK_GT(kernels.size(), 0) << "no kernel found for " << op_type;
  LOG(INFO) << "Create Kernels Success";
  auto& kernel = kernels.front();
  kernel->SetContext(ContextScheduler::Global().NewContext(kernel->target()));
  LOG(INFO) << "SetContext Success";
  auto ins_ptr = new Instruction(op, std::move(kernel));
  return ins_ptr;
}

void ComputeEngine<TARGET(kARM)>::SetParam(OpHandle op_h,
                                           operators::ParamBase* param) {
  auto ins = reinterpret_cast<Instruction*>(op_h);
  auto op = ins->mutable_op();
  CHECK(op) << "no Op found in instruction";
  auto kernel = ins->mutable_kernel();
  CHECK(kernel) << "no Kernel found in instruction";
  op->SetParam(param);
  op->AttachKernel(kernel);
  LOG(INFO) << "SetParam Success";
}

void ComputeEngine<TARGET(kARM)>::Launch(OpHandle op_h) {
  auto ins = reinterpret_cast<Instruction*>(op_h);
  ins->Run();
  LOG(INFO) << "Run Success";
}

void ComputeEngine<TARGET(kARM)>::ReleaseOpHandle(OpHandle op_h) {
  auto ins = reinterpret_cast<Instruction*>(op_h);
  delete ins;
}

}  // namespace lite
}  // namespace paddle
