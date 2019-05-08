// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include <core/common/common.h>
#include <core/graph/onnx_protobuf.h>
#include <core/framework/allocator.h>
#include <core/framework/tensor.h>

namespace onnxruntime {
struct SequentialExecutionPlan;
class ExecutionProviders;
class MemBuffer;

class ITensorAllocator {
 public:
  virtual common::Status FinalizePlan() = 0;
  virtual common::Status GetPreallocatedBuffer(int mlvalue_index, const char* name,
                                               std::unique_ptr<MemBuffer>& out) = 0;
  virtual common::Status Trace(int id, const ONNX_NAMESPACE::TensorProto* value) = 0;
  virtual ~ITensorAllocator() = default;
  ITensorAllocator() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ITensorAllocator);
  /**
   *
   * Caller must delete the returned pointer
   */
  static ITensorAllocator* Create(bool enable_mem_pattern, const SequentialExecutionPlan& execution_plan,
                                  const ExecutionProviders& exec_providers,
                                  std::vector<BufferUniquePtr>& weight_buffers);
};

}  // namespace onnxruntime
