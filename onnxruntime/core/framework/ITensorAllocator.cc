// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TensorAllocatorWithMemPattern.h"
#include "SimpleTensorAllocator.h"

namespace onnxruntime {

ITensorAllocator* ITensorAllocator::Create(bool enable_mem_pattern, const SequentialExecutionPlan& execution_plan,
                                           const ExecutionProviders& exec_providers,
                                           std::vector<BufferUniquePtr>& weights_buffers) {
  if (enable_mem_pattern) return new TensorAllocatorWithMemPattern(execution_plan, exec_providers, weights_buffers);
  return new SimpleTensorAllocator(execution_plan, exec_providers, weights_buffers);
}

}  // namespace onnxruntime