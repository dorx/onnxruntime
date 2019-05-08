// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include "ITensorAllocator.h"
#include "mem_pattern.h"
#include "ml_value_patterns_planner.h"
#include "utils.h"

namespace onnxruntime {
class ExecutionProviders;

class SimpleTensorAllocator : public ITensorAllocator {
 private:
  MLValuePatternPlanner planner_;
  MemoryPatternGroup mem_patterns_;
  std::vector<BufferUniquePtr>& weights_buffers_;
  const ExecutionProviders& exec_providers_;
  const SequentialExecutionPlan& seq_plan_;

 private:
  std::unordered_map<int, const ONNX_NAMESPACE::TensorProto*> values_;

 public:
  SimpleTensorAllocator(const SequentialExecutionPlan& execution_plan, const ExecutionProviders& exec_providers,
                        std::vector<BufferUniquePtr>& weights_buffers)
      : planner_(execution_plan),
        weights_buffers_(weights_buffers),
        exec_providers_(exec_providers),
        seq_plan_(execution_plan) {}
  common::Status FinalizePlan() override { return Status::OK(); }
  common::Status GetPreallocatedBuffer(int mlvalue_index, const char* name, std::unique_ptr<MemBuffer>& out) override;
  common::Status Trace(int id, const ONNX_NAMESPACE::TensorProto* value) override;
};
}  // namespace onnxruntime
